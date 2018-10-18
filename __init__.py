import sys
from os.path import dirname, realpath

dir_path = dirname(realpath(__file__))
project_path = realpath(dir_path + '/..')

# Adding where to find libraries and dependencies
sys.path.append(dir_path)
sys.path.append(project_path)

import preprocessing
import utils
