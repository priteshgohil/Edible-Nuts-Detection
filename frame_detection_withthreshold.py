#Detect still frame for each video in the given folder and save the result in specific folder.

import numpy as np
import argparse
import sys
if('/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path):
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import matplotlib.pyplot as plt

def get_frame_index(correlation, kldiv, total_frames):
    b = True #To run loop untill we get the frame number
    mul_factor = 1 #To amplify the kldiv by factor of 10 (only for too dark videos)
    kldiv = np.array(kldiv)
    correlation = np.array(correlation)
    frame_index = None
    FRAME_40_PER = (total_frames/2.5) # 40 % of total frame
    FRAME_20_PER = (total_frames/5) # 20% of the total frame
    while(b):
        diff = correlation - kldiv
        index = np.where(diff<0)[0]
        if(len(index)):
            if(mul_factor>1 and index[0] > FRAME_40_PER): #The index of the frame
                                #should be less than 40 % of the frames (if its more definately its wrong)
                kldiv = kldiv*10*mul_factor #Amplify the difference by 10,20,30,40 and so on
                mul_factor += mul_factor
                if(mul_factor>20): #try amplifying upto 200, still not found? then stop
                    print("cannot find the still frame index")
                    break
            else: #this means we got the frame in one shot and video is normal
                frame_index = index[0]
                if(frame_index<FRAME_40_PER): #if detected frame is less than 40% of the total frame then stop the detection.
                    b = False
                else: #if the frame index is more than 40 then repeat the process
                    mul_factor +=1
        else:
            kldiv = kldiv*10*mul_factor #Amplify the difference by 10,20,30,40 and so on
            mul_factor += mul_factor
            if(mul_factor>20): #try amplifying upto 200, still not found? then stop
                print("cannot find the still frame index")
                break
    return frame_index

def get_frame(vid_path, frame_num):
    cap = cv2.VideoCapture(vid_path)
    success, frame = cap.read() #read 1st frame
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def plot_for_report(vid_path, corr, kldiv,index, save_name, offset=12):
    im = get_frame(vid_path, index)
    fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(15,10))
    ax1.imshow(im)
    ax2.plot(kldiv)
    ax2.plot(corr)
    ax2.axvline(index,color='red',label=index)
    ax2.legend()

    detected_frame = index+offset
    im = get_frame(vid_path, detected_frame)
    ax3.imshow(im)
    ax4.plot(kldiv)
    ax4.plot(corr)
    ax4.axvline(detected_frame,color='green',label=detected_frame)
    ax4.legend()

    fig.savefig(save_name+"{}-{}.jpg".format(index, detected_frame))
    plt.clf()

def make_dataset(path):
    #default "../data/"
    data = []
    for root,d_names,f_names in os.walk(path):
        for f in f_names:
            if('.avi' in f):
                data.append(os.path.join(root,f))
    return data

def process_video(vid_path, result_folder):
    hist_correlation =[]
    hist_kldiv = []
    count = 0

    print("processing: {}".format(vid_path))
    cap = cv2.VideoCapture(vid_path)
    success, frame_prev = cap.read() #read 1st frame
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while cap.isOpened():
        success, frame_next = cap.read()
        if not success:
            break
        hist_prev = cv2.calcHist([frame_prev], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist_prev = cv2.normalize(hist_prev, hist_prev).flatten()

        hist_next = cv2.calcHist([frame_next], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist_next = cv2.normalize(hist_next, hist_next).flatten()
        hist_correlation.append(cv2.compareHist(hist_prev, hist_next, cv2.HISTCMP_CORREL))
        hist_kldiv.append(cv2.compareHist(hist_prev, hist_next, cv2.HISTCMP_KL_DIV))

        frame_prev = frame_next

    vid_frame = get_frame_index(hist_correlation,hist_kldiv, total_frames)

    #save the result
    video_name = vid_path.split("/")[-1]
    save_location = result_folder+video_name
    plot_for_report(vid_path,hist_correlation,hist_kldiv,vid_frame,save_location, offset=12)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/media/pritesh/Entertainment/cvData/", help="Path of the video or folder containing videos")
    parser.add_argument("--mode", type=int, default=2, help="mode 1 = Test video, mode 2 = Test the folder containing multiple videos")
    opt = parser.parse_args()
    print(opt)

    # folder to store the results
    result_folder = "../detected_frames/"
    result_folder2 = "../detected_frames2/"
    os.makedirs(result_folder2, exist_ok=True)

    if (opt.mode == 2):
        data = make_dataset("/media/pritesh/Entertainment/cvData/")
        for i,vid_path in enumerate(data):
            process_video(vid_path, result_folder2)
    else:
        process_video(opt.path, result_folder2)
