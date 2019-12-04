"""
This module contains a component that will generate new windows with slider,
to adjust the HSV mask for the image. Later those value can be hardcoded in our
code.
"""
import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np

class FindHSVValue(object):
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
        cv2.createTrackbar("LH", window_name, 0, 255, self.nothing)
        cv2.createTrackbar("LS", window_name, 0, 255, self.nothing)
        cv2.createTrackbar("LV", window_name, 0, 255, self.nothing)
        cv2.createTrackbar("UH", window_name, 255, 255, self.nothing)
        cv2.createTrackbar("US", window_name, 255, 255, self.nothing)
        cv2.createTrackbar("UV", window_name, 255, 255, self.nothing)

    def get_trackbar_pos(self, window_name):
        l_h = cv2.getTrackbarPos("LH", window_name)
        l_s = cv2.getTrackbarPos("LS", window_name)
        l_v = cv2.getTrackbarPos("LV", window_name)
        u_h = cv2.getTrackbarPos("UH", window_name)
        u_s = cv2.getTrackbarPos("US", window_name)
        u_v = cv2.getTrackbarPos("UV", window_name)
        return [l_h,l_s,l_v, u_h, u_s, u_v]
