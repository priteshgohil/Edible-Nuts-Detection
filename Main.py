import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from Find_Mask import FindHSVValue
from Conversion import ConvertFrames


converter = ConvertFrames()
hsvSlider = FindHSVValue()
while True:
    frame = cv2.imread("myimage.png")
    #convert to hsv
    hsv_frame = converter.convert_BGRtoHSV(frame = frame)

    hsvSlider.generate_window("Tracker")
    lh,ls,lv,uh,us,uv = hsvSlider.get_trackbar_pos("Tracker")

    lower_mask = np.array([lh,ls,lv])
    upper_mask = np.array([uh,us,uv])

    hsv_mask = cv2.inRange(hsv_frame, lower_mask, upper_mask)
    masked_frame = cv2.bitwise_and(frame, frame, mask=hsv_mask)

    cv2.imshow("frame", frame)
    cv2.imshow("hsv_mask", hsv_mask)
    cv2.imshow("masked_frame", masked_frame)

    key = cv2.waitKey(1)
    if key=='q':
        break
cv2.destroyAllWindows()
