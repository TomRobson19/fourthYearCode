import cv2
import os
import numpy as np
import csv
import math
from matplotlib import pyplot as plt
import geopy
import geopy.distance

#####################################################################

master_path_to_dataset = "TTBB-durham-02-10-17-sub5"
directory_to_cycle = "left-images"     # edit this for left or right image set

#Is OK just to use this, don't need to mess with stereo
def getScale(allGPS,index):

    previousImage = allGPS[index-1]
    currentImage = allGPS[index]

    previousLat = previousImage[0]
    previousLon = previousImage[1]
    currentLat = currentImage[0]
    currentLon = currentImage[1]

    dlon = currentLon - previousLon
    dlat = currentLat - previousLat

    distance = math.sqrt(dlon**2 + dlat**2)

    return distance

def GPSToXYZ():
    GPSXYZ = []
    gpsFile = open(master_path_to_dataset+"/GPS.csv") 
    for i, line in enumerate(gpsFile):
        if i != 0:
            temp = line.split(",")
            GPSXYZ.append([float(temp[1]), float(temp[2])])
    gpsFile.close()

    start = geopy.Point(GPSXYZ[0][0],GPSXYZ[0][1])

    for p in GPSXYZ:
        lat = geopy.Point(p[0],start.longitude)
        lon = geopy.Point(start.latitude,p[1])

        p[0] = geopy.distance.vincenty(start,lat).meters

        #change sign here to make go left
        if(p[0] < GPSXYZ[0][0]):
            p[0]*=-1
        p[1] = geopy.distance.vincenty(start,lon).meters
        if(p[1] < GPSXYZ[0][1]):
            p[1]*=-1

    return GPSXYZ

def XYZtoGPS(allGPS):
    start = geopy.point(allGPS[0][0],allGPS[0][1])
    for p in allGPS:
        d = geopy.distance.VincentyDistance(meters = p[0])
        newP = d.destination(point=start, bearing = 0)

        d = geopy.distance.VincentyDistance(meters = p[1])
        newP = d.destination(point=newP, bearing = 90)

        p[0] = newP.latitude
        p[1] = newP.longitude

    return allGPS

def featureBinning(kp):
    bin_size = 100
    features_per_bin = 50

    kp.sort(key=lambda x: x.response) 

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

def rotateFunct(pts_l, angle, degrees=False):
    """ Returns a rotated list(function) by the provided angle."""
    if degrees == True:
        theta = math.radians(angle)
    else:
        theta = angle

    R = np.array([ [math.cos(theta), -math.sin(theta)],
                   [math.sin(theta), math.cos(theta)] ])
    rot_pts = []
    for v in pts_l:
        v = np.array(v).transpose()
        v = R.dot(v)
        v = v.transpose()
        rot_pts.append(v)

    return rot_pts

def plotResults(allT,allGPS):
    angle = math.radians(-121.8)

    allGPS = allGPS[:len(allT)]

    newT = []
    for i,t in enumerate(allT):
        if i!=0:
            #remove minus to make go left
            newT.append([-t[0], t[2]])

    newT = np.array(newT)
    #allGPS = rotateFunct(allGPS,angle)

    plt.figure(1)
    GPS, = plt.plot(*zip(*allGPS), color='red', marker='o', label='GPS')
    pyMVO, = plt.plot(*zip(*newT), color='blue', marker='o',  label='py-MVO')
    plt.legend(handles=[pyMVO, GPS])
    # Set plot parameters and show it
    plt.axis('equal')
    plt.grid()
    plt.show()

#TO DO
def plotResultsOnMap():
    return None


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
lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

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
currentT = np.array([0,0,0])

allT = []

detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

allGPS = GPSToXYZ()

minFlowFeatures = 1500

for index, filename in enumerate(sorted(os.listdir(full_path_directory))[:52]):
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

            scale = getScale(allGPS,index)

            if scale > 0.00001:
                isForwardDominant = 100*t[2] > t[0]
                if currentR == []:
                    currentT = t*scale
                    currentR = R
                elif isForwardDominant:
                    currentR = R.dot(currentR)
                    currentT += scale*currentR.dot(t)
                    
                else:
                    print("Dominant motion not forward - ignored")
            else:
                print("Insufficient movement - assumed stationary")

            if len(good_matches1) < minFlowFeatures:
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
            break # exit
        elif (key == ord(' ')):     # pause (on next frame)
            pause_playback = not(pause_playback)
            print("pause")

    allT.append([currentT.item(0), currentT.item(1), currentT.item(2)])
    previous_kp = kp
    previous_img = img

# close all windows

plotResults(allT,allGPS)
cv2.destroyAllWindows()

#####################################################################

"""
TO DO:

WILL NEED TO CORRECT TO GROUND TRUTH, QUESTION IS HOW OFTEN?
"""