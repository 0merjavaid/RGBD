import cv2
import sys
import numpy as np
np.set_printoptions(threshold=np.nan)
sys.path.append(".")

import numpy as np
import cv2
import pytest
from utils import *

mask = cv2.imread('test_data/labels/ffmpeg_28.png',0)

contours = get_contours(mask)
with open('output.txt', 'w') as f:
	f.write(str(contours))
