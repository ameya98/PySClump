try:
    from .sclump import *
    from .utils import *
    from .PathSim.pathsim import *
except ImportError:
    from sclump import *
    from utils import *
    from PathSim.pathsim import *