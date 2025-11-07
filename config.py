#config.py

# flow definitions
MACH_NUMBER = 2.2  # Mach number for Student ID: 25211202
GAMMA = 1.667  # specified by Q. Helium gas

# numerical setup config
NUM_INTEGRATION_POINTS = int(1e6) + 1  # odd number, for numerical integration using Simpson's
NUM_IFT_ENTRIES = 10_000  # self-explanatory, NOTE: if being run for the first time, large value takes a while to calculate

# design setup
X_START, Y_START = 0, 1  # DO NOT CHANGE (x, y) location of throat
CENTRE_LINE_Y = 0  # DO NOT CHANGE, y-coordinate of the centreline (y=0)
NUM_EXPANSION_FANS = 5  # specified by Q, discretisation of infinite expansion fans, more=more accurate geometry
EXPANSION_FANS_FROM = 0.15  # degrees

# IFT
IFT_JSON = "IFT.json"  # same directory as codebase
OVERWRITE = False  # overwrite IFT?