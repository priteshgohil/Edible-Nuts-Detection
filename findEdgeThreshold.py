"""
This module contains a component that will generate new windows with slider,
to adjust the two threshold for edge detection the image. Later those value can be hardcoded in our
code.
"""
import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

class FindEdgeThreshold(object):
    """
    Slider of HSV lower and upper bound to easily find the mask.
    """
    def __init__(self):
        pass

    def nothing(self):
        pass

    def generate_window(self, window_name):
        """
        Generate opencv window with slider to adjust HSV value
        """
        cv2.namedWindow(window_name)
        cv2.createTrackbar("th1", window_name, 0, 255, self.nothing)
        cv2.createTrackbar("th2", window_name, 0, 255, self.nothing)

    def get_trackbar_pos(self, window_name):
        th1 = cv2.getTrackbarPos("th1", window_name)
        th2 = cv2.getTrackbarPos("th1", window_name)
        return [th1, th2]
