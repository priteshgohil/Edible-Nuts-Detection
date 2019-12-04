import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

class TrackerGenerator(object):
    def __init__(self):
        pass

    def get_boosting_tracker(self):
        return cv2.TrackerBoosting_create()

    def get_MIL_tracker(self):
        return cv2.TrackerMIL_create()

    def get_KCF_tracker(self):
        return cv2.TrackerKCF_create()

    def get_TLD_tracker(self):
        return cv2.TrackerTLD_create()

    def get_CSRT_tracker(self):
        return cv2.TrackerCSRT_create()

    def get_MedianFlow_tracker(self):
        return cv2.TrackerMedianFlow_create()

    def get_GOTURN_tracker(self):
        return cv2.TrackerGOTURN_create()

    def get_MOSSE_tracker(self):
        return cv2.TrackerMOSSE_create()
