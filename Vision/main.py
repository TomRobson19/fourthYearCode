import cv2
import os
import numpy as np
import csv

#####################################################################

# where is the data ? - set this to where you have it

master_path_to_dataset = "TTBB-durham-02-10-17-sub5"; # ** need to edit this **
directory_to_cycle = "left-images";     # edit this for left or right image set



# unsure whether this is the correct way to get scale from gps, might be Breckon's answer in the FAQ
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
    return np.sqrt(((float(currentImage[1])-float(previousImage[1]))**2) + ((float(currentImage[2])-float(previousImage[2]))**2) + ((float(currentImage[3])-float(previousImage[3]))**2))
  

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

# set this to a file timestamp to start from (empty is first example - outside lab)
# e.g. set to 1506943191.487683 for the end of the Bailey, just as the vehicle turns

skip_forward_file_pattern = ""; # set to timestamp to skip forward to

pause_playback = False; # pause until key press after each image

#####################################################################

# resolve full directory location of data set for images

full_path_directory =  os.path.join(master_path_to_dataset, directory_to_cycle);

previous_kp = None
previous_des = None

first_image = True

currentR = []
currentT = []

thres = 5000
surf = cv2.xfeatures2d.SURF_create(thres)
index_params = dict(algorithm = 0, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)


for index, filename in enumerate(sorted(os.listdir(full_path_directory))):
    full_path_filename = os.path.join(full_path_directory, filename);
    # skip forward to start a file we specify by timestamp (if this is set)
    if(first_image):
        first_image = False
        img = cv2.imread(full_path_filename, cv2.IMREAD_COLOR)
        previous_kp, previous_des = surf.detectAndCompute(img,None)
        temp = (sorted(zip(previous_kp, previous_des), key=lambda pair: pair[0].response))

        previous_kp,previous_des = [list(t) for t in zip(*temp)]
        
    else:

        if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename)):
            continue;
        elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename)):
            skip_forward_file_pattern = "";

        # from image filename get the correspondoning full path

        img = cv2.imread(full_path_filename, cv2.IMREAD_COLOR)

        kp, des = surf.detectAndCompute(img,None)        

        temp = (sorted(zip(kp, des), key=lambda pair: pair[0].response))

        kp,des = [list(t) for t in zip(*temp)]

        matches = flann.knnMatch(np.asarray(des), np.asarray(previous_des), k=2)

        good_matches1 = []
        good_matches2 = []

        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:   #filter out 'bad' matches
                good_matches1.append(kp[m.queryIdx].pt)
                good_matches2.append(previous_kp[m.trainIdx].pt)

        old_good_matches1 = good_matches1
        old_good_matches2 = good_matches2
        
        #binning - bins are 68x128
        #size is 544 x 1024
        bin_size = 100

        no_bins = 64

        good_matches1 = [[] for _ in range(no_bins)]
        good_matches2 = [[] for _ in range(no_bins)]


        for i in old_good_matches1:
            bin_to_place = int(i[0]//128 + 8*(i[1]//68))
            if len(good_matches1[bin_to_place]) < bin_size:
                if len(good_matches1[bin_to_place]) != 0:
                    good_matches1[bin_to_place].append(i)
                else:
                    good_matches1[bin_to_place] = [i]

        for i in old_good_matches2:
            bin_to_place = int(i[0]//128 + 8*(i[1]//68))
            if len(good_matches2[bin_to_place]) < bin_size:
                if len(good_matches2[bin_to_place]) != 0:
                    good_matches2[bin_to_place].append(i)
                else:
                    good_matches2[bin_to_place] = [i]

        good_matches1 = [item for sublist in good_matches1 for item in sublist]
        good_matches2 = [item for sublist in good_matches2 for item in sublist]

        good_matches1 = np.array(good_matches1)
        good_matches2 = np.array(good_matches2)

        print(len(old_good_matches1),len(old_good_matches2))
        print(len(good_matches1),len(good_matches2))

        img2 = cv2.drawKeypoints(img,kp,img)

        essential_matrix,_ = cv2.findEssentialMat(good_matches1,good_matches2,focal=camera_focal_length_px,pp=(optical_image_centre_w,optical_image_centre_h),method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        _,R,t,_ = cv2.recoverPose(essential_matrix,good_matches1,good_matches2,focal=camera_focal_length_px,pp=(optical_image_centre_w,optical_image_centre_h))
        # print(R)
        # print(t)

        scale = getScaleFromGPS(index)
        # print(scale)

        if scale > 0.00001 or currentT == []:
            isForwardDominant = t[2] > t[0] and t[2] > t[1]
            if currentT == [] and currentR == []:
                currentT = t*scale
                currentR = R
            elif isForwardDominant:
                currentR = R.dot(currentR)
                currentT += scale*currentR.dot(t)
                
            else:
                print("Dominant motion not forward, ignored")
        else:
            print("Insufficient movement - assumed stationary")

        # print(currentR)
        # print(currentT)
        # print(scale)


        previous_kp = kp
        previous_des = des

        cv2.imshow('input image',img2)

        key = cv2.waitKey(40 * (not(pause_playback))) & 0xFF; # wait 40ms (i.e. 1000ms / 25 fps = 40 ms)
        if (key == ord('x')):       # exit
            print("Keyboard exit requested : exiting now - bye!")
            break; # exit
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
            print("pause")

# close all windows

cv2.destroyAllWindows()

#####################################################################

"""
TO DO:
Feature Binning
Start with initial GPS points and then plot them
Hope it works
"""