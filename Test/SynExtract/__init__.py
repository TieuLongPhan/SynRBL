import sys
sys.path.append('../../')
import sys
from pathlib import Path

# Add the root directory to sys.path
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))

from SynRBL import *
