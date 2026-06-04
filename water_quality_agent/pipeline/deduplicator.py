"""
Station deduplication engine.

Detects and resolves duplicate stations across data sources using:
  1. Exact station-ID matching (within same source)
  2. Fuzzy name matching (Levenshtein / token-set similarity)
  3. Spatial proximity (Haversine distance < threshold)
  4. Combined scoring: name_sim × 0.4 + spatial_sim × 0.6

Outputs a canonical station registry mapping all source station IDs
to a single unified station ID.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from ..config.schema import CanonicalRecord
from ..utils.geo import haversine_km

logger = logging.getLogger(__name__)

# Spatial threshold: stations within this distance are candidates for merging
DISTANCE_THRESHOLD_KM = 2.0

# Name similarity threshold (0–1, where 1 = identical)
NAME_SIM_THRESHOLD = 0.6

# Combined score threshold for merging
MERGE_THRESHOLD = 0.7


@dataclass
class StationInfo:
    """Aggregated info about a station from one or more sources."""
    canonical_id: str
    names: List[str]
    latitudes: List[float]
    longitudes: List[float]
    rivers: List[str]
    states: List[str]
    source_ids: Dict[str, str]   # source_name → source_station_id
    record_count: int = 0

    @property
    def latitude(self) -> float:
        return sum(self.latitudes) / len(self.latitudes)

    @property
    def longitude(self) -> float:
        return sum(self.longitudes) / len(self.longitudes)

    @property
    def best_name(self) -> str:
        """Pick the most informative (longest) name."""
        return max(self.names, key=len) if self.names else ""

    @property
    def river(self) -> str:
        rivers = [r for r in self.rivers if r]
        return max(set(rivers), key=rivers.count) if rivers else ""

    @property
    def state(self) -> str:
        states = [s for s in self.states if s]
        return max(set(states), key=states.count) if states else ""


def _normalize_name(name: str) -> str:
    """Normalize station name for comparison."""
    s = name.lower().strip()
    # Remove common prefixes/suffixes
    s = re.sub(r"\b(station|monitoring|wq|water quality|cpcb|spcb)\b", "", s)
    # Remove punctuation
    s = re.sub(r"[^a-z0-9\s]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _token_set_similarity(a: str, b: str) -> float:
    """Token-set similarity: Jaccard index of word tokens."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _spatial_similarity(dist_km: float) -> float:
    """Convert distance to a 0–1 similarity score."""
    if dist_km <= 0.1:
        return 1.0
    if dist_km >= DISTANCE_THRESHOLD_KM * 2:
        return 0.0
    return max(0, 1 - dist_km / (DISTANCE_THRESHOLD_KM * 2))


class Deduplicator:
    """Station deduplication engine."""

    def __init__(
        self,
        distance_threshold_km: float = DISTANCE_THRESHOLD_KM,
        name_sim_threshold: float = NAME_SIM_THRESHOLD,
        merge_threshold: float = MERGE_THRESHOLD,
    ):
        self.distance_threshold = distance_threshold_km
        self.name_sim_threshold = name_sim_threshold
        self.merge_threshold = merge_threshold
        self.stations: Dict[str, StationInfo] = {}
        self._id_map: Dict[str, str] = {}  # source_station_id → canonical_id
        self._next_id = 1

    def _generate_id(self) -> str:
        cid = f"WQ_{self._next_id:05d}"
        self._next_id += 1
        return cid

    def build_station_registry(
        self, records: List[CanonicalRecord]
    ) -> Dict[str, StationInfo]:
        """Build a deduplicated station registry from records.

        Step 1: Group records by (source, source_station_id).
        Step 2: For each source station, find best match in existing registry.
        Step 3: If match score > threshold, merge. Otherwise, create new.
        """
        # Group by source station
        source_groups: Dict[str, List[CanonicalRecord]] = defaultdict(list)
        for rec in records:
            key = f"{rec.source}::{rec.source_station_id}"
            source_groups[key].append(rec)

        logger.info(f"Deduplicating {len(source_groups)} source stations...")

        for key, recs in source_groups.items():
            rep = recs[0]  # Representative record
            lat = rep.latitude
            lon = rep.longitude
            name = rep.station_name
            river = rep.river
            state = rep.state

            if lat == 0.0 and lon == 0.0:
                # No coordinates — can only match by name
                pass

            # Find best match in existing registry
            best_match_id = None
            best_score = 0.0

            norm_name = _normalize_name(name)

            for cid, station in self.stations.items():
                # Spatial similarity
                if lat and lon and station.latitude and station.longitude:
                    dist = haversine_km(lat, lon, station.latitude, station.longitude)
                    spatial_sim = _spatial_similarity(dist)
                else:
                    spatial_sim = 0.0

                # Name similarity
                for existing_name in station.names:
                    name_sim = _token_set_similarity(norm_name, _normalize_name(existing_name))
                    combined = name_sim * 0.4 + spatial_sim * 0.6
                    if combined > best_score:
                        best_score = combined
                        best_match_id = cid

            if best_score >= self.merge_threshold and best_match_id:
                # Merge into existing station
                station = self.stations[best_match_id]
                if name and name not in station.names:
                    station.names.append(name)
                if lat and lat not in station.latitudes:
                    station.latitudes.append(lat)
                if lon and lon not in station.longitudes:
                    station.longitudes.append(lon)
                if river:
                    station.rivers.append(river)
                if state:
                    station.states.append(state)
                station.source_ids[rep.source] = rep.source_station_id
                station.record_count += len(recs)
                canonical_id = best_match_id
                logger.debug(
                    f"Merged '{name}' ({rep.source}) → {best_match_id} "
                    f"(score={best_score:.2f})"
                )
            else:
                # Create new station
                canonical_id = self._generate_id()
                self.stations[canonical_id] = StationInfo(
                    canonical_id=canonical_id,
                    names=[name] if name else [],
                    latitudes=[lat] if lat else [],
                    longitudes=[lon] if lon else [],
                    rivers=[river] if river else [],
                    states=[state] if state else [],
                    source_ids={rep.source: rep.source_station_id},
                    record_count=len(recs),
                )

            self._id_map[key] = canonical_id

        logger.info(
            f"Deduplication result: {len(source_groups)} source stations → "
            f"{len(self.stations)} canonical stations"
        )
        return self.stations

    def apply_ids(self, records: List[CanonicalRecord]) -> List[CanonicalRecord]:
        """Replace station_id in records with canonical IDs."""
        for rec in records:
            key = f"{rec.source}::{rec.source_station_id}"
            if key in self._id_map:
                canonical_id = self._id_map[key]
                rec.station_id = canonical_id
                station = self.stations.get(canonical_id)
                if station:
                    rec.station_name = station.best_name
                    rec.river = station.river
                    rec.state = station.state
                    rec.latitude = station.latitude
                    rec.longitude = station.longitude
        return records
