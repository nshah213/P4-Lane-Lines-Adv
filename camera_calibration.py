## Calibrate the camera
# 

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

# prepare object points
nx = 9 #The number of inside corners in x
ny = 6 #The number of inside corners in y

objpoints = []

objp = np.zeros((6*9,3),np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
#print(objp)

imgpoints = []
calImages =[]

# Make a list of calibration images
fnames = glob.glob("./camera_cal/*.jpg")
for fname in fnames:
	img = cv2.imread(fname)
	calImages.append(img)	
	# Convert to grayscale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#print(np.shape(gray))
	# Find the chessboard corners
	ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
	# If found, draw corners

	if ret == True:
		#print("Picture "+str(fname)+" Corners found:" + str(len(corners)))
		imgpoints.append(corners)
		objpoints.append(objp)

# Function that takes an image, object points, and image points
# performs the camera calibration, image distortion correction and 
# returns the undistorted image
def calibrate_camera(imgshape, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,imgshape,None,None)
    return mtx, dist

mtx, dist = calibrate_camera(calImages[0].shape[1::-1],objpoints, imgpoints)
"""
pickle_file = './camera_cal/camera.p'
print('Saving data to pickle file...')
try:
	with open('./camera_cal/camera.p', 'wb') as pfile:
		pickle.dump(
                {
                	'CameraMatrix': mtx,
			'Distortion': dist
                },
                pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
	print('Unable to save data to', pickle_file, ':', e)
	raise
"""
#Uncomment to plot undistorted calibrate camera images for writeup needs
#undist_images = []
f, ax = plt.subplots(1, 2, figsize=(8, 3),dpi = 100)
f.tight_layout()
offset = 7
for i in range(len(calImages)):
	undist = cv2.undistort(calImages[i], mtx, dist,None,mtx)
	cv2.imwrite("./output_images/camera_calibration/undistorted_image_"+str(i)+".png",undist)
	if (i== 7):
		ax[0].imshow(calImages[i])
		ax[0].set_title("Original Image "+str(i+1))
		ax[0].axis('off')

		ax[1].imshow(undist)
		ax[1].set_title("Undistorted Image "+str(i+1))
		ax[1].axis('off')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig("./output_images/camCal_Images_and_Results.png")
plt.show()

f, ax = plt.subplots(1, 2, figsize=(8, 3),dpi = 100)
f.tight_layout()
offset = 0
# Make a list of calibration images
fnames = glob.glob("./test_images/*.jpg")
for i in range(len(fnames)):
	img = cv2.imread(fnames[i])
	undist = cv2.undistort(img, mtx, dist,None,mtx)
	cv2.imwrite("./output_images/undistorted_test_image_"+str(i)+".png",undist)
	
	if (i== offset):
		ax[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
		ax[0].set_title("Original Image "+str(i+1))
		ax[0].axis('off')

		ax[1].imshow(cv2.cvtColor(undist,cv2.COLOR_BGR2RGB))
		ax[1].set_title("Undistorted Image "+str(i+1))
		ax[1].axis('off')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
plt.savefig("./output_images/camCal_Images_and_Results.png")
plt.show()

