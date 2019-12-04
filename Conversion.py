import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

class ConvertFrames(object):
    def __init__(self):
        pass

    def convert_BGRtoHSV(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def convert_BGRtoRGB(self,frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def convert_BGRtoGrayscale(self,frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
