import sys
sys.path.append(".")

import numpy as np
import cv2
import pytest
from utils import *


class TestUtils(object):

    @classmethod
    def setup_class(cls):
        cls.mask = cv2.imread('test_data/labels/ffmpeg_28.png',0)
        cls.image = cv2.imread('test_data/images/ffmpeg_28.png')
        cls.fake_mask = np.zeros((720, 1280))

    def test_get_contours(self):
    	contours = get_contours(self.mask)

    	with pytest.raises(AssertionError):
    		img_contours = get_contours(self.image)
    	
    	with pytest.raises(AttributeError):
    		img_contours = get_contours([1,2,3])

    	with pytest.raises(TypeError):
    		img_contours = get_contours()

    	assert get_contours(self.fake_mask) == []
    	assert len(contours) == 1
    	assert isinstance(contours, list)


    def test_get_contour_area(self):
    	contours = get_contours(self.mask)
    	contour = contours[0]
    	cnt_area = get_contour_area(contour)
    	
    	with pytest.raises(AssertionError):
    		get_contour_area([1,2,3])

    	with pytest.raises(TypeError):
    		get_contour_area()

    	assert isinstance(cnt_area, float)
    	assert np.isclose(cnt_area, 69203)

    def test_refine_contours(self):

    	contours = get_contours(self.mask)
    	bbox_out = np.array([537,  98, 891, 535])
    	bbox, hand_contour = refine_contours(contours)

    	with pytest.raises(AssertionError):
    		refine_contours(contours[0])

    	with pytest.raises(AssertionError):
    		refine_contours(contours, 'abc')

    	with pytest.raises(TypeError):
    		refine_contours()

    	assert isinstance(bbox, np.ndarray)
    	assert bbox.shape == (4,)
    	assert np.allclose(bbox, bbox_out)
    	assert isinstance(hand_contour, np.ndarray)
    	