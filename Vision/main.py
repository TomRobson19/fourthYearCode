import cv2
import os
import numpy as np
import csv

#####################################################################

# where is the data ? - set this to where you have it

master_path_to_dataset = "TTBB-durham-02-10-17-sub5"; # ** need to edit this **
directory_to_cycle = "left-images";     # edit this for left or right image set

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

thres = 100

previous_image = None

first_image = True

for index, filename in enumerate(sorted(os.listdir(full_path_directory))):

    full_path_filename = os.path.join(full_path_directory, filename);
    # skip forward to start a file we specify by timestamp (if this is set)
    if(first_image):
        first_image = False
        previous_image = cv2.imread(full_path_filename, cv2.IMREAD_COLOR)
    else:

        if ((len(skip_forward_file_pattern) > 0) and not(skip_forward_file_pattern in filename)):
            continue;
        elif ((len(skip_forward_file_pattern) > 0) and (skip_forward_file_pattern in filename)):
            skip_forward_file_pattern = "";

        # from image filename get the correspondoning full path

        img = cv2.imread(full_path_filename, cv2.IMREAD_COLOR)

        #USE ORB NOT SURF
        surf = cv2.xfeatures2d.SURF_create(thres)
        kp1, des1 = surf.detectAndCompute(img,None)
        kp2, des2 = surf.detectAndCompute(previous_image,None)

        #img2 = cv2.drawKeypoints(img,kp,None,(255,0,0),4)

        index_params = dict(algorithm = 1, trees = 1)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        

        good_matches1 = []
        good_matches2 = []


        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:   #filter out 'bad' matches
                good_matches1.append(kp1[m.queryIdx].pt)
                good_matches2.append(kp2[m.trainIdx].pt)

        good_matches1 = np.array(good_matches1)
        good_matches2 = np.array(good_matches2)

        essential_matrix,mask = cv2.findEssentialMat(good_matches1,good_matches2)
        
        print(essential_matrix)

        _,R,t,mask = cv2.recoverPose(essential_matrix,good_matches1,good_matches2)

        print(R)
        print(t)


        previous_image = img

        cv2.imshow('input image',img)

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