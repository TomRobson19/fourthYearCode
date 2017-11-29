import cv2
import os
import numpy as np
import csv
import math
from matplotlib import pyplot as plt

#####################################################################

# where is the data ? - set this to where you have it

master_path_to_dataset = "TTBB-durham-02-10-17-sub5"; # ** need to edit this **
directory_to_cycle = "left-images";     # edit this for left or right image set


#Is OK just to use this, don't need to mess with stereo
def getScaleFromGPS(index):
    gpsFile = open(master_path_to_dataset+"/GPS.csv") 
    previousImage = []
    currentImage = []
    for i, line in enumerate(gpsFile):
        if i == index:
            previousImage = line.split(",")
        elif i == index+1:
            currentImage = line.split(",")
            break
    gpsFile.close()

    radius = 6373.0

    previousLat = math.radians(float(previousImage[1]))
    previousLon = math.radians(float(previousImage[2]))
    currentLat = math.radians(float(currentImage[1]))
    currentLon = math.radians(float(currentImage[2]))

    dlon = currentLon - previousLon
    dlat = currentLat - previousLat

    a = math.sin(dlat / 2)**2 + math.cos(previousLat) * math.cos(currentLat) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = (radius * c)*1000.0

    return distance

def featureBinning(kp):
    bin_size = 100
    features_per_bin = 50

    bins_y = math.ceil(img.shape[0]/bin_size)
    bins_x = math.ceil(img.shape[1]/bin_size)

    number_of_bins = bins_x * bins_y

    temp_kp = [[] for _ in range(number_of_bins)]

    for i,p in enumerate(kp):
        bin_to_place = int(p.pt[0]//bin_size + bins_x*(p.pt[1]//bin_size))
        if len(temp_kp[bin_to_place]) < bin_size:
            if len(temp_kp[bin_to_place]) != 0:
                temp_kp[bin_to_place].append(kp[i])
            else:
                temp_kp[bin_to_place] = [kp[i]]

    kp = [item for sublist in temp_kp for item in sublist]
    return kp


#####################################################################

# full camera parameters - from camera calibration
# supplied images are stereo rectified

camera_focal_length_px = 399.9745178222656;  # focal length in pixels (fx, fy)
camera_focal_length_m = 4.8 / 1000;          # focal length in metres (4.8 mm, f)
stereo_camera_baseline_m = 0.2090607502;     # camera baseline in metres (B)
camera_height_above_wheelbase_m = (1608.0 + 31.75 + 90) / 1000; # in mm

optical_image_centre_h = 262.0;             # from calibration - cy
optical_image_centre_w = 474.5;             # from calibration - cx

image_height = 544;
image_width = 1024;

#####################################################################

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#####################################################################

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for images

full_path_directory =  os.path.join(master_path_to_dataset, directory_to_cycle);

previous_kp = None
previous_img = None

first_image = True

currentR = []
currentT = []

detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

for index, filename in enumerate(sorted(os.listdir(full_path_directory))):
    full_path_filename = os.path.join(full_path_directory, filename);

    img = cv2.imread(full_path_filename, cv2.IMREAD_COLOR)
    img = img[0:340, 0:image_width]

    if(first_image):
        first_image = False
        kp = detector.detect(img) 
        kp = featureBinning(kp)
        kp = np.array([x.pt for x in kp], dtype=np.float32)
    else:

        kp, st, err = cv2.calcOpticalFlowPyrLK(previous_img, img, previous_kp, None, **lk_params)
        st = st.reshape(st.shape[0])

        good_matches1 = (kp[st==1])
        good_matches2 = (previous_kp[st==1])

        if len(good_matches1) > 5:

            essential_matrix,_ = cv2.findEssentialMat(good_matches1,good_matches2,focal=camera_focal_length_px,pp=(optical_image_centre_w,optical_image_centre_h),method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _,R,t,_ = cv2.recoverPose(essential_matrix,good_matches1,good_matches2,focal=camera_focal_length_px,pp=(optical_image_centre_w,optical_image_centre_h))

            scale = getScaleFromGPS(index)

            if scale > 0.00001 or currentT == []:
                isForwardDominant = t[2] > t[0] and t[2] > t[1]
                if currentT == [] and currentR == []:
                    currentT = t*scale
                    currentR = R
                elif isForwardDominant:
                    currentR = R.dot(currentR)
                    currentT += scale*currentR.dot(t)
                    
                else:
                    print("Dominant motion not forward - ignored")
            else:
                print("Insufficient movement - assumed stationary")

            if len(good_matches2) < 1500:
                kp = detector.detect(img) 
                kp = featureBinning(kp)
                
                img2 = cv2.drawKeypoints(img,kp,img)
                kp = np.array([x.pt for x in kp], dtype=np.float32)
                cv2.imshow('input image',img2)
            else:
                cv2.imshow('input image',img)

            print(currentR)
            print(currentT)
            print(scale)








        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            print("Keyboard exit requested : exiting now - bye!")
            break; # exit
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
            print("pause")

    previous_kp = kp
    previous_img = img

# close all windows

cv2.destroyAllWindows()

#####################################################################

"""
TO DO:
PLOT THE FUCKER

WILL NEED TO CORRECT TO GROUND TRUTH, QUESTION IS HOW OFTEN?
"""