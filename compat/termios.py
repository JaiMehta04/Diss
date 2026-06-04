"""Minimal Windows stub for termios to satisfy blessings on Earth Engine.
No-op implementations and placeholder constants.
"""
TIOCGWINSZ = 0


def tcgetattr(fd):
    return []


def tcsetattr(fd, when, attributes):
    return None


def ioctl(fd, request, arg=0, mutate_flag=True):
    return 0

# Placeholder control character values
VINTR = VQUIT = VERASE = VKILL = VEOF = VEOL = 0
VTIME = VMIN = VSWTC = VSTART = VSTOP = VSUSP = VEOL2 = 0
IGNBRK = BRKINT = IGNPAR = PARMRK = INPCK = ISTRIP = INLCR = IGNCR = ICRNL = IXON = 0
IXANY = IXOFF = IMAXBEL = IUTF8 = 0
OPOST = OLCUC = ONLCR = OCRNL = ONOCR = ONLRET = OFILL = OFDEL = NLDLY = CRDLY = 0
TABDLY = BSDLY = VTDLY = FFDLY = 0
CSIZE = CS5 = CS6 = CS7 = CS8 = CSTOPB = CREAD = PARENB = PARODD = HUPCL = CLOCAL = 0
ISIG = ICANON = ECHO = ECHOE = ECHOK = ECHONL = NOFLSH = TOSTOP = IEXTEN = 0
ECHOCTL = ECHOPRT = ECHOKE = FLUSHO = PENDIN = EXTPROC = 0
TCIFLUSH = TCOFLUSH = TCIOFLUSH = 0
TCSANOW = TCSADRAIN = TCSAFLUSH = 0
