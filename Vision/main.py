import cv2
import os
import numpy as np
import csv
import math
from matplotlib import pyplot as plt
import geopy
import geopy.distance
import gmplot

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

def originalGPS():
    GPS = []
    gpsFile = open(master_path_to_dataset+"/GPS.csv") 
    for i, line in enumerate(gpsFile):
        if i != 0:
            temp = line.split(",")
            GPS.append([float(temp[1]), float(temp[2])])
    gpsFile.close()
    return GPS

def originalIMU():
    IMU = []
    imuFile = open(master_path_to_dataset+"/IMU.csv") 
    for i, line in enumerate(imuFile):
        if i != 0:
            temp = line.split(",")
            IMU.append([float(temp[1]), float(temp[2]), float(temp[3]), float(temp[4])])
    imuFile.close()
    return IMU


def GPSToXYZ():
    GPSXYZ = originalGPS()

    start = geopy.Point(GPSXYZ[0][0],GPSXYZ[0][1])

    newGPSXYZ = []

    for p in GPSXYZ:
        newP = [0,0]
        lat = geopy.Point(p[0],start.longitude)
        lon = geopy.Point(start.latitude,p[1])

        newP[0] = geopy.distance.vincenty(start,lat).meters

        if(p[0] > start[0]):
            newP[0]*=-1

        newP[1] = geopy.distance.vincenty(start,lon).meters
        if (p[1] < start[1]):
            newP[1]*=-1

        newGPSXYZ.append(newP)

    return newGPSXYZ

def XYZtoGPS(allGPS):
    temp = originalGPS()
    start = geopy.Point(temp[0][0],temp[0][1])
   
    for p in allGPS:
        d = geopy.distance.VincentyDistance(meters = p[0])
        newP = d.destination(point=start, bearing = 180)

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

def plotResults(allT,allGPS):
    allGPS = allGPS[:len(allT)]

    plt.figure(1)
    GPS, = plt.plot(*zip(*allGPS), color='red', marker='o', label='GPS')
    pyMVO, = plt.plot(*zip(*allT), color='blue', marker='o',  label='py-MVO')
    plt.legend(handles=[pyMVO, GPS])
    # Set plot parameters and show it
    plt.axis('equal')
    plt.grid()
    plt.show()


def plotResultsOnMap(allT):
    GPS = originalGPS()
    T = XYZtoGPS(allT)

    originalLat = []
    originalLon = []
    myLat = []
    myLon = []

    GPS = GPS[:len(T)]
    
    for i in range(len(T)):
        originalLat.append(GPS[i][0])
        originalLon.append(GPS[i][1])
        myLat.append(T[i][0])
        myLon.append(T[i][1])

    gmap = gmplot.GoogleMapPlotter(54.767093,-1.570038, 16)

    gmap.plot(originalLat, originalLon, 'red', edge_width=10)
    gmap.plot(myLat, myLon, 'cornflowerblue', edge_width=10)

    gmap.draw("mymap.html")

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

def correctToGroundTruth(allT, allGPS,frequency):
    correctionFrequency = frequency

    allT = np.array(allT)
    allGPS = np.array(allGPS)

    startIndex = 0
    newAllT = []
    angle = 0
    correction = 0

    for i,p in enumerate(allT):
        if i%correctionFrequency == 0 and i != 0:
            #find correct position 
            startIndex = i
            initialPoint = allT[i+5]-allT[i]
            angle = np.degrees(np.arctan2(initialPoint[0],initialPoint[1]))
            gpsInitialPoint = allGPS[i+5]-allGPS[i]
            gpsAngle = np.degrees(np.arctan2(gpsInitialPoint[0],gpsInitialPoint[1]))
            angle -= gpsAngle
            correction = allGPS[i] - p
        newT = p+correction
        initialPoint = allGPS[startIndex]
        newT = rotateFunct([newT-initialPoint], np.radians(angle))[0]+initialPoint
        newAllT.append(newT)

    return newAllT


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
lk_params = dict(winSize  = (3, 3), #default is 21
                #maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

full_path_directory =  os.path.join(master_path_to_dataset, directory_to_cycle);

previous_kp = None
previous_img = None

first_image = True

currentR = []
currentT = np.array([0,0,0],dtype=np.float64)

allT = []

detector = cv2.FastFeatureDetector_create(threshold=50, nonmaxSuppression=True)

allGPS = GPSToXYZ()

minFlowFeatures = 500

imu = originalIMU()

for index, filename in enumerate(sorted(os.listdir(full_path_directory))):#[:100]):
    full_path_filename = os.path.join(full_path_directory, filename)

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
            essential_matrix,_ = cv2.findEssentialMat(good_matches1,good_matches2,focal=camera_focal_length_px,pp=(optical_image_centre_w,optical_image_centre_h),method=cv2.RANSAC,prob=0.999,threshold=1.0)
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

        key = cv2.waitKey(1)  
        if (key == ord('x')):       
            print("Keyboard exit requested : exiting now - bye!")
            break # exit

    allT.append([currentT.item(0), currentT.item(1), currentT.item(2)])
    previous_kp = kp
    previous_img = img


newT = []
for i,t in enumerate(allT):
    newT.append([t[0], t[2]])

#correctedT = newT
correctedT = correctToGroundTruth(newT,allGPS,50)

# close all windows
plotResults(correctedT,allGPS)
plotResultsOnMap(correctedT)
cv2.destroyAllWindows()

#####################################################################