## Calibrate the camera
# 

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import pickle

with open('./camera_cal/camera.p', mode='rb') as f:
    data = pickle.load(f)
    
mtx, dist = data['CameraMatrix'], data['Distortion']
"""
# Uncomment this to run one time to manually collect the 4 points of interest for perspective transform
# Manually select points that will be used for perspective transform, 
#image ./straight_lines1.jpg will be used for this purpose
fname = './test_images/straight_lines2.jpg'
img = cv2.imread(fname)
undist = cv2.undistort(img, mtx, dist,None,mtx)
refPts = []
def select_ROI_points(event, x, y, flags, param):
	global refPts
	if event == cv2.EVENT_LBUTTONDOWN:
		pass
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPts.append([np.float32(x), np.float32(y)])
		print(refPts)


cv2.imshow("Select perspective points", undist)
cv2.setMouseCallback("Select perspective points", select_ROI_points)
wait = True	
while(wait):
	key = cv2.waitKey(1) & 0xFF	
	if len(refPts) == 4:
		wait = False

print("Selected points are - ")
print(refPts)
cv2.destroyAllWindows()
# output when run once is - [[240,696],[622,432],[663,432],[1078,692]]
src = np.asarray(refPts)
print(src)
"""
## Calculate M and Minv for applying perspective transform
# Load the points of interest for perspective transform
# uncomment next line if we want to recollect new points for creating the transform
#src = np.float32([[251,689],[586,455],[689,455],[1070,689]])
#src = np.float32([[251,689],[586,455],[689,455],[1070,689]])
#src = np.float32([[251,689],[590,455],[700,455],[1020,689]]) # current ok performance- scope for improvement
src = np.float32([[203,720],[585,460],[695,460],[1027,720]]) # From writeup template - uncanny similarity with selected points :)
#create destination points assuming the order of selection of points is clockwise,
# starting from bottom left point
#dst = np.float32([src[0],[src[0][0],src[1][1]],[src[3][0],src[2][1]],src[3]])
#dst = np.float32([[251,720],[250,0],[970,0],[1000,720]])
dst = np.float32([[320,720],[320,0],[960,0],[960,720]])
print(dst)

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)



## Code to enable thresholding
def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255), is_gray = False):
	if orient == 'x':
		sobel = cv2.Sobel(gray, cv2.CV_64F,1,0)
	if orient == 'y':
		sobel = cv2.Sobel(gray, cv2.CV_64F,0,1)
	# 3) Take the absolute value of the derivative or gradient
	abs_sobel = np.absolute(sobel)
	#print(np.shape(abs_sobel))
	# 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
	scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
	# 5) Create a mask of 1's where the scaled gradient magnitude 
	binary_output = np.zeros_like(gray)
	#print(np.shape(gray))
	binary_output[(scaled_sobel > thresh[0]) & (scaled_sobel < thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	return binary_output

def mag_thresh(gray, sobel_kernel=3, thresh=(0, 255), is_gray = False):
    # Calculate gradient magnitude
    # Apply threshold
	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
	# 3) Calculate the magnitude
	mag = (sobelx**2 + sobely**2)**0.5
	# 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
	scaled_mag = np.uint8(mag/np.max(mag)*255.)
	# 5) Create a binary mask where mag thresholds are met
	binary_output = np.zeros_like(gray)
	binary_output[(scaled_mag > thresh[0])&(scaled_mag < thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	return binary_output

def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2),is_gray = False):
	# Calculate gradient direction
	# Apply threshold
	# 2) Take the gradient in x and y separately
	sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
	sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
	# 3) Take the absolute value of the x and y gradients
	abs_sobelx = np.absolute(sobelx)
	abs_sobely = np.absolute(sobely)
	# 4) Use  to calculate the direction of the gradient
	dirImg = np.arctan2(abs_sobely, abs_sobelx)
	# 5) Create a binary mask where direction thresholds are met
	binary_output = np.zeros_like(gray)
	binary_output[(dirImg > thresh[0])&(dirImg < thresh[1])] = 1
	# 6) Return this mask as your binary_output image
	return binary_output, dirImg

def singlech_threshold(channel, thresh=(0, 127)):
	# Accept 1 channel image input and apply threshold provided as input
	# Apply threshold
	binary_output = np.zeros_like(channel)
	binary_output[(channel >= thresh[0]) & (channel <= thresh[1])] = 1
	return binary_output


### Current level of masking includes 
def laneLineMasking(img, hist_flag = False):
	
	#img = np.copy(img)
	# Convert to grayscale for gradient thresholding
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#grayImage = grayscale(image)
	clahe = cv2.createCLAHE(tileGridSize =(50,50))
	histImage = clahe.apply(gray)
	#plt.imshow(histImage, cmap = 'Greys_r')

	src_int = np.array([[[100,696],[548,433],[800,433],[1240,696]]],dtype = np.int32)
	region_masked = region_of_interest(img,src_int)#np.array([src], dtype =np.int32)

	# Convert to HLS color space and separate the V channel
	hls = cv2.cvtColor(region_masked, cv2.COLOR_BGR2HLS).astype(np.float)
	hsv = cv2.cvtColor(region_masked, cv2.COLOR_BGR2HSV).astype(np.float)
	h_channel = hls[:,:,0]
	l_channel = (hls[:,:,1] - np.average(hls[:,:,1]))*255/((np.max(hls[:,:,1]))-np.average(hls[:,:,1]))
	s_channel = (hls[:,:,2] - np.average(hls[:,:,2]))*255/(np.max(hls[:,:,2]))
	#print(hls)
	sv_channel = hsv[:,:,1]
	v_channel = (hsv[:,:,2] - np.average(hsv[:,:,2]))*255/(np.max(hsv[:,:,2])- np.average(hsv[:,:,2]))
	#if best_fit_flag == False:
	# Threshold H value for detecting yellow
	h_mask = singlech_threshold(h_channel, [98, 103])
	l_mask = singlech_threshold(l_channel,[235,255])
	# Threshold S to detect yellow and white
	v_mask = singlech_threshold(v_channel, [190,255])
	s_mask = singlech_threshold(s_channel, [200,255])
	sv_mask = singlech_threshold(sv_channel, [200,255])
	# Calculate gradient based masks
	#sobelx_mask = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(60, 100))
	#print(np.shape(sobelx_mask))
	#maskedHist = region_of_interest(histImage,src_int)
	#sobeldir_mask, dirImage = dir_threshold(gray, sobel_kernel=3, thresh=(0.9, 1.1))
	#print(np.shape(sobeldir_mask))    
	# Threshold x gradient
	sobelmag_mask = mag_thresh(gray,3, (35, 70))
	#else:
		# Threshold H value for detecting yellow
	#	h_mask = singlech_threshold(h_channel, [98, 103])
	#	l_mask = singlech_threshold(l_channel,[235,255])
		# Threshold S to detect yellow and white
	#	v_mask = singlech_threshold(v_channel, [140,255])
	#	s_mask = singlech_threshold(s_channel, [180,255])
	#	sv_mask = singlech_threshold(sv_channel, [180,255])
		# Calculate gradient based masks
	#	sobelx_mask = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(40, 80))
		#print(np.shape(sobelx_mask))
	#	maskedHist = region_of_interest(histImage,src_int)
	#	sobeldir_mask, dirImage = dir_threshold(gray, sobel_kernel=3, thresh=(0.9, 1.1))
	#	#print(np.shape(sobeldir_mask))    
		# Threshold x gradient
	#	sobelmag_mask = mag_thresh(gray,3, (20, 50))

		#print(np.shape(sobelmag_mask))
		# Threshold color chan	nel
		# Stack each chaannel
		# Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
		# be beneficial to replace this channel with something else.
	#color_binary_hsl = np.uint8(np.dstack((s_channel, l_channel, h_channel))*255)
	#color_binary_sobel = np.uint8(np.dstack((sobelx_mask, h_mask,sobeldir_mask))*255)
	result_binary = np.zeros_like(h_mask)
	if hist_flag == True:
		vnew_mask = singlech_threshold(v_channel, [99,255])
		result_binary[((sobelmag_mask == 1)&(vnew_mask == 1))] = 1
	else:
		result_binary[(v_mask == 1)| (s_mask == 1)| (sv_mask == 1)] = 1	#| (s_mask == 1)|(l_mask == 1)| (s_mask == 1)|(h_mask == 1)|((sobelx_mask==1)|(sobelmag_mask == 1)
	result_binary = np.uint8(result_binary)	
	document = False	
	if (document):	
		f, ([ax1, ax2],[ax3, ax4]) = plt.subplots(2, 2, figsize=(8, 6), dpi = 100)
		f.tight_layout()
		ax1.imshow(v_channel,cmap = 'gray')
		ax1.set_title('v_channel')
		ax1.axis('off')
		ax2.imshow(sobelmag_mask,cmap = 'gray')
		ax2.set_title('sobel_magmask')
		ax2.axis('off')
		ax3.imshow(s_channel, cmap = 'gray')
		ax3.set_title('s_channel_hls')
		ax3.axis('off')
		ax4.imshow(result_binary,cmap = 'gray')
		ax4.set_title("binary_result")
		ax4.axis('off')
		#ax5.imshow(dirImage,cmap = 'gray')
		#ax5.set_title('Sobel direction')
		plt.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02)
		document = False # Set off to stop saving images
		if document:
			plt.savefig('./output_images/current_masking_pipeline.png')
		plt.show()
	
	return result_binary

class Line():
	def __init__(self):
		# was the line detected in the last iteration?
		self.detected = False  
		# x values of the last n fits of the line
		self.recent_xfitted = []
		self.hist_length = 0
		#average x values of the fitted line over the last n iterations
		self.bestx = None     
		#polynomial coefficients averaged over the last n iterations
		self.best_fit = None  
		#polynomial coefficients for the most recent fit
		self.current_fit = [np.array([False])]  
		#radius of curvature of the line in some units
		self.radius_of_curvature = None 
		#distance in meters of vehicle center from the line
		self.line_base_pos = [] 
		#difference in fit coefficients between last and new fits
		self.diffs = np.array([0,0,0], dtype='float') 
		#x values for detected line pixels
		self.allx = [] 
		#y values for detected line pixels	
		self.ally = []
		
		# try to sweep height of image scanned for the lines, some videos may do better if only lower part of image is used for lane detection in  windowing
		self.image_height_crop = 0
		self.best_fit_available = False		
		self.best_base_pos = 0
		self.cnt_missed = 0
		self.cnt_detected = 0
		self.top_xposition = 0
		self.convolve_search = True


def window_mask(width, height, img_ref, center,level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	return output

## This function returns a list of windows that we should look for lane line pizels in for one left and one right lane line
## It expects global Line() to initialized for right and left lane lines

def find_window_centroids(image, window_width, window_height, margin):
	#print("Entering find_window_centroids:")
	window_centroids = [] # Store the (left,right) window centroid positions per level
	windows_left = [] # Store the left window centroid positions per level
	windows_right = [] # Store the right window centroid positions per level
	window = np.ones(window_width)
	#print(np.shape(window))
	global line_left
	global line_right
	global even_harder_challenge
	# Create our window template that we will use for convolutions
	multiplier = 3./4.
	if(even_harder_challenge):
		multiplier = 9./10.
	if (line_left.best_fit_available == True)&(line_left.detected == True):
		l_center = line_left.line_base_pos[line_left.hist_length - 1]
		print("left line - Using best fit position")		
		print(l_center)
	else:
		l_sum = np.sum(image[int(multiplier*image.shape[0]):,:int(image.shape[1]/3)], axis=0)
		l_center = np.argmax(np.convolve(window,l_sum))-window_width/3
		l_center = max(l_center,int(window_width/3))
		print("Recalculating left base position")
		print(l_center)
	if (line_right.best_fit_available == True)&(line_right.detected == True):
		r_center = line_right.line_base_pos[line_right.hist_length - 1]
		print("right line - Using best fit position")
		print(r_center)
	else:				
		r_sum = np.sum(image[int(multiplier*image.shape[0]):,int(2*image.shape[1]/3):], axis=0)
		r_center = np.argmax(np.convolve(window,r_sum))-window_width/3+int(2*image.shape[1]/3)
		r_center = min(r_center,np.shape(image)[1]-int(window_width/3))
		print("Recalculating right base position")
		print(r_center)
	#windows_left.append(l_center)
	#windows_right.append(r_center)
	delta_r_center = 0
	delta_l_center = 0
	left_stop = False
	right_stop = False
	search_region_left = margin
	search_region_right = margin
	line_left.convolve_search = False
	line_right.convolve_search = False
	convolve_search_left = False
	convolve_search_right = False
	
	missedCompensation = 5
	if (line_left.best_fit_available == True)&(line_left.detected == True):
		for level in range(0,(int)((image.shape[0] - line_left.image_height_crop)/window_height)):
			y = int(image.shape[0]-level*window_height)		
			l_center = int(line_left.best_fit[0]*(y)**2 + line_left.best_fit[1]*(y)+ line_left.best_fit[2])
			windows_left.append(l_center)
		#if (even_harder_challenge):
			#line_left.convolve_search = True
			#windows_left = []
	else:
		convolve_search_left = True # To be deleted
		line_left.convolve_search = True

	if (line_right.best_fit_available == True)&(line_right.detected == True):
		for level in range(0,(int)((image.shape[0] - line_right.image_height_crop)/window_height)):
			y = int(image.shape[0]-level*window_height)		
			r_center = int(line_right.best_fit[0]*(y)**2 + line_right.best_fit[1]*(y)+ line_right.best_fit[2])
			windows_right.append(r_center)
		#if (even_harder_challenge):
			#line_right.convolve_search = True
			#windows_right = []
	else:
		convolve_search_right = True
		line_right.convolve_search = True
	# Go through each layer looking for max pixel locations
	if (line_left.convolve_search | line_right.convolve_search):
		for level in range(0,(int)((image.shape[0] - line_left.image_height_crop)/window_height)):
			# convolve the window into the vertical slice of the image
			layer_sum = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
			layer_total_pixels = np.sum(layer_sum)		
		
			conv_signal = np.convolve(window, layer_sum)
			offset = int(window_width/2)		
	
			if(line_left.convolve_search):
				l_min_index = int(max(l_center-search_region_left,0))		
				l_max_index = int(min(l_center+search_region_left,image.shape[1]))
				nonzero_left = layer_sum[l_min_index:l_max_index].nonzero
				pixels_left = list(np.shape(nonzero_left()))[1]
				#print(search_region_left)
				if (left_stop == False):
					if (pixels_left > 0):
						# Find the best left centroid by using past left center as a reference
						# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
						l_center_new = np.argmax(conv_signal[l_min_index+offset:l_max_index+offset])+l_min_index#-offset
						delta_l_center = 0
						if even_harder_challenge:
							delta_l_center = max(min(2,l_center_new - l_center),-2)
						l_center = l_center_new
						search_region_left -= missedCompensation
						search_region_left = max(margin, search_region_left) 
					else :				
						l_center += delta_l_center
						search_region_left += missedCompensation
						search_region_left = min(search_region_left, 150)
					#if l_center < search_region_left:
					if l_center < 0:
						left_stop = True
						left_center = search_region_left
					#if l_center > image.shape[1] - search_region_left:
					if l_center > image.shape[1]:
						left_stop = True
						left_center = image.shape[1] -search_region_left
					windows_left.append(l_center)

			if(line_right.convolve_search):		
				r_min_index = int(max(r_center-search_region_right,0))
				r_max_index = int(min(r_center+search_region_right,image.shape[1]))
				nonzero_right = layer_sum[r_min_index:r_max_index].nonzero	
				pixels_right = list(np.shape(nonzero_right()))[1]
		
				if (right_stop == False):
					if (pixels_right > 0):
					# Find the best left centroid by using past left center as a reference
					# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
					#print(r_min_index, r_max_index, offset)				
						r_center_new = np.argmax(conv_signal[r_min_index+offset:r_max_index+offset])+r_min_index#-offset
						delta_r_center = 0#max(min(5,r_center_new - r_center),-5)
						if even_harder_challenge:
							delta_r_center = max(min(2,r_center_new - r_center),-2)
						#print(delta_r_center)
						r_center = r_center_new
						search_region_right -= missedCompensation
						search_region_right = max(margin, search_region_right) 
					else :
						r_center += delta_r_center
						search_region_right += missedCompensation
						search_region_left = min(search_region_left, 150)
					if r_center < 0:
						right_stop = True
						right_center = search_region_right
					if r_center > image.shape[1]:
						right_stop = True
						right_stop = image.shape[1] -search_region_right
				#print(windows_right)	
				windows_right.append(r_center)
		
	return windows_left, windows_right

# Expects global lane lines to be initialized and iteration to be set up for naming purposes
def findLaneLinesConv(image):
	#print(np.shape(warped_binary))
	#img = cv2.imread(fnames[i])
	#test_images.append(img)
	global line_left
	global line_right
	global iteration
	global even_harder_challenge
	#if (debug_perspectivePoints):
	#	getPerspectivePoints(image)
	iteration += 1
	undist=cv2.undistort(image, mtx, dist,None,mtx)
	[h,w] = np.shape(image)[:2]
	masked_binary = laneLineMasking(undist,False)# even_harder_challenge)
	#src_int = np.array([[[100,696],[628,433],[665,433],[1120,696]]],dtype = np.int32)
	#region_masked = region_of_interest(masked_binary,src_int)#np.array([src], dtype =np.int32)
	#apply perspective
	warped_binary =  cv2.warpPerspective(masked_binary, M, (w,h))
	#masked_binary = laneLineMasking(destination)
	
	# window settings
	window_width = 80
	window_height = 10		 # Break image into 9 vertical layers since image height is 720
	search_margin = 50 # How much to slide left and right for searching
	#if ((line_left.best_fit_available == True)&(line_right.best_fit_available == True)):
		#window_width = 50
		#search_margin = 150
	windows_left, windows_right = find_window_centroids(warped_binary, window_width, window_height, search_margin)
	#print(windows_left)
	#print(windows_right)	
	# If we found any window centers
	output = np.dstack((warped_binary, warped_binary, warped_binary))*255
	window_img = np.zeros_like(output)
	lane_image = np.zeros_like(output)
	result = np.zeros_like(image)

	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/710 # meters per pixel in x dimension

	document = False

	left_lane_inds = []
	if len(windows_left) > 0:
		# Points used to draw all the left and right windows
		l_points = np.zeros_like(warped_binary)
		# Go through each level and select image pixels in the area of the windows
		row_offset = line_left.image_height_crop
		nonzero = warped_binary[row_offset:,:].nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		ploty = np.linspace(row_offset, warped_binary.shape[0]-1, warped_binary.shape[0])
		good_left_inds = []
		margin = int(window_width/2)
		if(line_left.convolve_search == False):
			margin += 0 # need additional margin if orientation of vehicle changing with respect to lane lines
		for level in range(0,max(len(windows_left),0)):

			#print(level)
			# Window_mask is a function to draw window areas
			l_mask = window_mask(window_width,window_height,warped_binary,windows_left[level],level)
			# Identify window boundaries in x and y (and right and left)
			leftx_current = windows_left[level]
			#print(leftx_current)
			
			if(line_left.convolve_search == False):
				margin += 0
			win_y_low = warped_binary.shape[0]-row_offset - (level+1)*window_height
			win_y_high = warped_binary.shape[0]-row_offset - level*window_height
			win_xleft_low = int(leftx_current - margin)
			win_xleft_low = max(win_xleft_low,0)
			win_xleft_high = int(leftx_current + margin)
			win_xleft_high = min(win_xleft_high,warped_binary.shape[1])
			if document:
				cv2.rectangle(output,(win_xleft_low,win_y_low + row_offset),(win_xleft_high,win_y_high+row_offset),(0,125,0),2)
			# Draw the windows on the visualization image
			#cv2.rectangle(output,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0),2)
			#cv2.rectangle(output,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0),2)
			# Identify the nonzero pixels in x and y within the window
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
			
			threshold = 0
			if (len(good_left_inds)> threshold):
				left_lane_inds.append(good_left_inds)
		
		#print(good_left_inds)
		y_eval = np.max(ploty)
		if (left_lane_inds != []):				
			left_lane_inds = np.concatenate(left_lane_inds)
			#print("len(left_lane_inds = ",len(left_lane_inds))
			leftx = nonzerox[left_lane_inds]
			lefty = nonzeroy[left_lane_inds]
			lefty = lefty + row_offset
			
			left_fit, left_res, _,_,_ = np.polyfit(lefty, leftx, 2, full = True)
			print("Left Fit Details ",len(leftx),' ',left_res,' ',left_res/len(leftx))
			
			print("Left_Fit = ", left_fit)
			print(line_left.cnt_missed)
			#print("Len of line_left.recent_xfitted = ",len(line_left.recent_xfitted))
			y_eval = np.max(ploty)
			left_fitx = left_fit[0]*(ploty)**2 + left_fit[1]*(ploty)+ left_fit[2]			
			#print("New fit = ",left_fit)
			#print(ploty)
			if ((left_res/len(leftx) < 1200)):
				line_left.detected = True
				if (line_left.best_fit_available):
					delta_topx = np.abs(left_fitx[0] - line_left.top_xposition)
					delta_botx = np.abs(left_fitx[len(left_fitx)-1] - line_left.best_base_pos)
					print(delta_topx)
					print(line_left.best_fit_available)
					if (delta_topx > 300)|(delta_botx > 50):
						line_left.detected = False
						line_left.cnt_missed += 1
					else:
						line_left.line_base_pos.append(leftx[0])
						line_left.recent_xfitted.append(left_fit)
						line_left.hist_length += 1
						#line_left.cnt_missed -= 1
						line_left.cnt_detected += 1
				else:
					line_left.line_base_pos.append(leftx[0])
					line_left.recent_xfitted.append(left_fit)
					line_left.hist_length += 1
					#line_left.cnt_missed -= 1
					line_left.cnt_detected += 1
				# line_left.cnt_detected = min(10,line_left.cnt_detected)
				
				if line_left.cnt_missed > 10:
					line_left.best_fit_available = False
					line_left.recent_xfitted = []
					line_left.line_base_pos = []
					line_left.hist_length = 0
					line_left.cnt_missed = 0
					line_left.cnt_detected = 0
				print(line_left.hist_length, len(line_left.recent_xfitted))
				if line_left.cnt_detected > 2:
					line_left.best_fit = []
					for i in range (len(left_fit)):
						sum_fit = 0
						sum_basepos = 0
						for j in range(3):
							print(line_left.cnt_detected)
							sum_fit +=  line_left.recent_xfitted[line_left.hist_length-1 - j][i]
														
							if (i==0):
								sum_basepos += (line_left.line_base_pos[line_left.hist_length -1 -j])
								#print(line_left.line_base_pos[line_left.hist_length -1 -j])
								#print(sum_basepos)
								line_left.best_base_pos = sum_basepos/3.
								print(line_left.best_base_pos)
						line_left.best_fit.append(sum_fit/3)
					print(line_left.best_fit)
					line_left.best_fit_available = True
					line_left.top_xposition = left_fitx[0]

				else:
					line_left.best_fit_available = False
					#line_left.cnt_missed = 0
					#line_left.best_fit = []
					#line_left.hist_length = 0
			else:
				line_left.detected = False
				line_left.cnt_missed += 1
				line_left.cnt_missed = min(10,line_left.cnt_missed)
				if (line_left.hist_length > 0):
					left_fitx = line_left.recent_xfitted[line_left.hist_length-1][0]*(ploty)**2 + line_left.recent_xfitted[line_left.hist_length-1][1]*(ploty)+ line_left.recent_xfitted[line_left.hist_length-1][2]				
				print("line_left = False")	
			
			output[lefty, leftx] = [255, 0, 0]
				#lane_image[left_fitx, ploty]= [255,255,0]
				#if(line_left.best_fit_available == False):
				#	#left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
				#left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])			
			if(line_left.best_fit_available == False):
				left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-5, ploty]))])
				left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+5,ploty])))])
				left_line_pts = np.hstack((left_line_window1, left_line_window2))	
				cv2.fillPoly(lane_image, np.int_(left_line_pts), (255,255, 0))
				result = cv2.addWeighted(output, 1, lane_image, 1, 0)
		else:
			line_left.detected = False
			line_left.cnt_missed += 1

		if(line_left.best_fit_available == True):
			left_fitx = line_left.best_fit[0]*(ploty)**2 + line_left.best_fit[1]*(ploty)+ line_left.best_fit[2]
			left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-5, ploty]))])
			left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+5,ploty])))])
			left_line_pts = np.hstack((left_line_window1, left_line_window2))	
			cv2.fillPoly(lane_image, np.int_(left_line_pts), (0,255, 255))
			result = cv2.addWeighted(output, 1, lane_image, 1, 0)
			left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
			left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
			print("Averaged radius of curvature for left lane = ",left_curverad," m")
	else:
		line_left.best_fit_available = False

	#lane_image = np.zeros_like(output)
	right_lane_inds = []
	if len(windows_right) > 0:
		# Points used to draw all the right windows
		r_points = np.zeros_like(warped_binary)
		# Go through each level and select image pixels in the area of the windows 
		good_right_inds = []
		row_offset = line_right.image_height_crop
		nonzero = warped_binary[row_offset:,:].nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		ploty = np.linspace(row_offset, warped_binary.shape[0]-1, warped_binary.shape[0])
		#print(len(windows_right))
		margin = int(window_width/2)
		if(line_right.convolve_search == False):
			margin += 0
		for level in range(0,np.max(len(windows_right),0)):
			# Window_mask is a function to draw window areas
			r_mask = window_mask(window_width,window_height,warped_binary,windows_right[level],level)
			#print("Flag right cnv search = ", line_right.convolve_search)
			if(line_right.convolve_search == False):
				margin += 0 
			# Identify window boundaries in x and y (and right and left)
			rightx_current = windows_right[level]
			win_y_low = int(warped_binary.shape[0]- row_offset - (level+1)*window_height)
			win_y_high = int(warped_binary.shape[0]- row_offset - level*window_height)
			win_xright_low = int(rightx_current - margin)
			win_xright_low = max(win_xright_low,0)
			win_xright_high = int(rightx_current + margin)
			win_xright_high = min(win_xright_high,warped_binary.shape[1])
			#print(win_xright_low,win_y_low,win_xright_high,win_y_high)
			if(document):
				# Draw the windows on the visualization image
				cv2.rectangle(output,(win_xright_low,win_y_low+row_offset),(win_xright_high,win_y_high+row_offset),(0,125,0),2)
			# Identify the nonzero pixels in x and y within the window
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
			#print(win_y_low, win_y_high, win_xright_low, win_xright_high)
			#print(good_right_inds)			
			# Append these indices to the lists
			threshold = 0
			if (len(good_right_inds)> threshold):
				right_lane_inds.append(good_right_inds)
		
		#print(np.shape(right_lane_inds))
		if (right_lane_inds != []):				
			right_lane_inds = np.concatenate(right_lane_inds)
			
			rightx = nonzerox[right_lane_inds]
			righty = nonzeroy[right_lane_inds]
			righty = righty + row_offset
			
			#print(rightx)
			#print(righty)
			#print(len(righty))
			right_fit, right_res, _,_,_ = np.polyfit(righty, rightx, 2, full = True)
			print("Right Res per N = ",len(rightx),' ',right_res,' ',right_res/len(rightx))
			#print(rightx[0])			
			#print()
			#print(len(ploty))
			#print(/)
			print("Right_Fit = ", right_fit)
			print(line_right.cnt_missed)
			y_eval = np.max(ploty)
			y_eval = np.max(ploty)
			right_fitx = right_fit[0]*(ploty)**2 + right_fit[1]*(ploty)+ right_fit[2]
			if ((right_res/len(rightx) < 1200)): #(len(rightx) > 1000)&			
				line_right.detected = True
				
				if (line_right.best_fit_available):
					delta_topx = np.abs(right_fitx[0] - line_right.top_xposition)
					#print(delta_topx)
					#if (delta_topx > 600):
					delta_botx = np.abs(right_fitx[len(right_fitx)-1] - line_right.best_base_pos)
					#print(delta_topx)
					#print(line_right.best_fit_available)
					if (delta_topx > 300)|(delta_botx > 50):
						line_right.detected = False
						line_right.cnt_missed += 1
					else:
						line_right.line_base_pos.append(rightx[0])
						line_right.recent_xfitted.append(right_fit)
						line_right.hist_length += 1
						line_right.cnt_detected += 1
				else:
					line_right.line_base_pos.append(rightx[0])
					line_right.recent_xfitted.append(right_fit)
					line_right.hist_length += 1
					line_right.cnt_detected += 1
				#line_right.cnt_detected = min(10,line_right.cnt_missed)	
				if line_right.cnt_missed > 10:
					line_right.best_fit_available = False
					line_right.recent_xfitted = []
					line_right.line_base_pos = []
					line_right.hist_length = 0
					line_right.cnt_missed = 0
					line_right.cnt_detected = 0
				
				if line_right.cnt_detected > 2:
					line_right.best_fit = []
					for i in range (len(right_fit)):
						sum_fit = 0
						sum_basepos = 0
						for j in range(3):
							sum_fit +=  line_right.recent_xfitted[line_right.hist_length-1 - j][i]
							if (i==0):
								sum_basepos += line_right.line_base_pos[line_right.hist_length -1 -j]
								line_right.best_base_pos = sum_basepos/3.
						line_right.best_fit.append(sum_fit/3)
					line_right.best_fit_available = True
					line_right.top_xposition = right_fitx[0]
				else:
					line_right.best_fit_available = False
					#line_right.cnt_missed = 0

			else:
				line_right.detected = False
				line_right.cnt_missed += 1
				line_right.cnt_missed = min(3,line_right.cnt_missed)
				if (line_right.hist_length > 0):
					right_fitx = line_right.recent_xfitted[line_right.hist_length-1][0]*(ploty)**2 + line_right.recent_xfitted[line_right.hist_length-1][1]*(ploty)+ line_right.recent_xfitted[line_right.hist_length-1][2]				
				print("line_right = False")	
			
			output[righty, rightx] = [0, 0, 255]
			#lane_image[right_fitx, ploty]= [255,255,0]
			if(line_right.best_fit_available == False):
				#right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
				#right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])			
				right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-5, ploty]))])
				right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+5,ploty])))])
				right_line_pts = np.hstack((right_line_window1, right_line_window2))	
				cv2.fillPoly(lane_image, np.int_(right_line_pts), (255,255, 0))
				result = cv2.addWeighted(output, 1, lane_image, 1, 0)
		else:
			line_right.detected = False
			line_right.cnt_missed += 1

		if(line_right.best_fit_available == True):
			right_fitx = line_right.best_fit[0]*(ploty)**2 + line_right.best_fit[1]*(ploty)+ line_right.best_fit[2]
			right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-5, ploty]))])
			right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+5,ploty])))])
			right_line_pts = np.hstack((right_line_window1, right_line_window2))	
			cv2.fillPoly(lane_image, np.int_(right_line_pts), (0,255, 255))
			result = cv2.addWeighted(output, 1, lane_image, 1, 0)
			right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
			right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
			print("Averaged radius of curvature for right lane = ",right_curverad," m")
	else:
		line_right.best_fit_available = False			

	
	if((right_lane_inds != [])&(left_lane_inds != [])):
		right_line_main = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
		left_line_main = np.array([np.flipud(np.transpose(np.vstack([left_fitx,ploty])))])
		combined_pts = np.hstack((left_line_main, right_line_main))		
		#right_line_pt = np.flipud(right_line_pts[0,:,:])
		#print(right_line_pt)
		#print(np.shape(right_line_pt))
		#pts_combined = np.array(np.vstack((left_line_pts[0,:,:],right_line_pt)),dtype = np.int32)
		#print((combined_pts))
		color = [0,127,0]
		if ((line_left.best_fit_available)&(line_right.best_fit_available)):
			color = [0,0,127]
		cv2.fillPoly(window_img, np.int_(combined_pts), color)
		result = cv2.addWeighted(result, 1, window_img, 0.5, 0)

	#if (line_left.detected & line_right.detected):		
	#	print(left_curverad, 'm', right_curverad, 'm')
	if (document):
		plt.imshow(result)
		#plt.plot(left_fitx, ploty, color='yellow')
		#plt.plot(right_fitx, ploty, color='yellow')
		plt.xlim(0, 1280)
		plt.ylim(720, 0)
		plt.axis('off')
		
		if document:
			plt.savefig("./output_images/lane_detection_mask_warped.png")
		#plt.plot(left_fitx, ploty, color='yellow')
		#plt.plot(right_fitx, ploty, color='yellow')
		#plt.xlim(0, 1280)
		#plt.ylim(720, 0)
		plt.show()
	unwarped = cv2.warpPerspective(result, Minv, (w,h))
	#result = cv2.warpPerspective(result, Minv, (w,h))
	final_result = cv2.addWeighted(image, 0.8, unwarped, 1, 0)

	### Print the calculated curvature on the return image frame
	font = cv2.FONT_HERSHEY_SIMPLEX
	if(left_lane_inds != []):
		if(line_left.best_fit_available):
			text_left = "Radius of curvature left lane = "+str(left_curverad)+" m"
			cv2.putText(final_result, text_left ,(20,30), font, 1,(255,255,255),2,cv2.LINE_AA)
	if(right_lane_inds != []):
		if(line_right.best_fit_available):
			text_right = "Radius of curvature right lane = "+str(right_curverad)+" m"				
			cv2.putText(final_result, text_right ,(20,60), font, 1,(255,255,255),2,cv2.LINE_AA)
	if((left_lane_inds != [])&(right_lane_inds != [])):
		if(line_left.best_fit_available&line_right.best_fit_available):	
			left_position = left_fitx[len(left_fitx)-1]
			right_position = right_fitx[len(right_fitx)-1]
			#right_position = line_right.best_base_pos
			#print(left_position, right_position, (right_position - left_position)/2)
			offset_lane_center = ((left_position+(right_position - left_position)/2) - 640)* xm_per_pix
			#print(offset_lane_center)
			if (offset_lane_center >= 0):
				text_offset = "Left of center of lane by = "+str(offset_lane_center)+" m"
			if (offset_lane_center < 0):
				text_offset = "Right of center of lane by = "+str(offset_lane_center*-1)+" m"
			cv2.putText(final_result, text_offset ,(20,90), font, 1,(255,255,255),2,cv2.LINE_AA)
	if document:
		cv2.imwrite("./output_images/lane_line_detection.jpg",cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR))
		temporary =  cv2.warpPerspective(image, M, (w,h))
		cv2.imwrite("./output_images/perspective_transform_result.jpg",cv2.cvtColor(temporary, cv2.COLOR_RGB2BGR))
	return final_result



def region_of_interest(img, vertices):
	"""
	Applies an image mask.
	Only keeps the region of the image defined by the polygon
	formed from `vertices`. The rest of the image is set to black.
	"""
	#defining a blank mask to start with
	mask = np.zeros_like(img)   
	#defining a 3 channel or 1 channel color to fill the mask with depending on the input image
	if len(img.shape) > 2:
		channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
		ignore_mask_color = (255,) * channel_count
	else:
		ignore_mask_color = 255
        
	#filling pixels inside the polygon defined by "vertices" with the fill color    
	cv2.fillPoly(mask, vertices, ignore_mask_color)
	#plt.imshow(img)
	#plt.show()
	#returning the image only where mask pixels are nonzero
	masked_image = cv2.bitwise_and(img, mask)
	return masked_image

from moviepy.editor import VideoFileClip
fnames = []
fnames = glob.glob("*.mp4")

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
line_left = Line()
line_right = Line()
iteration = 0

# Loop to iterate through all the videos .mp4 file present in the folder
for file_num in range(0,len(fnames)):
	clip1 = VideoFileClip(fnames[file_num])
	#reset lines for the new video
	even_harder_challenge = False
	line_right.image_height_crop = 0
	line_left.image_height_crop = 0

	if file_num == 1:
		line_right.image_height_crop = 400
		line_left.image_height_crop = 400
		even_harder_challenge = True
	if file_num == 2:
		line_right.image_height_crop = 200
		line_left.image_height_crop = 200

	line_left.best_fit_available = False		
	line_left.best_base_pos = 0
	line_left.cnt_missed = 0
	line_left.cnt_detected = 0
	line_left.top_xposition = 0
	line_left.convolve_search = True

	line_right.best_fit_available = False		
	line_right.best_base_pos = 0
	line_right.cnt_missed = 0
	line_right.cnt_detected = 0
	line_right.top_xposition = 0
	line_right.convolve_search = True


	line_left.detected = False
	line_right.detected = False
	line_left.recent_xfitted = []
	line_left.hist_length = 0
	line_left.best_fit = None  
	line_left.current_fit = [np.array([False])]  
	line_left.radius_of_curvature = None 
	line_left.line_base_pos = [] 
	line_left.allx = [] 
	line_left.ally = []
	line_left.best_fit_available = False		
	line_left.best_base_pos = 0
	line_right.recent_xfitted = []
	line_right.hist_length = 0
	line_right.best_fit = None  
	line_right.current_fit = [np.array([False])]  
	line_right.radius_of_curvature = None 
	line_right.line_base_pos = [] 
	line_right.allx = [] 
	line_right.ally = []
	line_right.best_fit_available = False		
	line_right.best_base_pos = 0
	print(fnames[file_num])
	white_output = './result/tmp'+str(file_num)+'.mp4'
	white_clip = clip1.fl_image(findLaneLinesConv)  #NOTE: this function expects color images!!
	white_clip.write_videofile(white_output, audio=False)

"""
# Read test images first and undistort them
#fnames = []
#fnames = glob.glob("./test_images/*.jpg")
test_images = []
undist_images = []
for i in range(0,5):
	img = cv2.imread(fnames[i])
	test_images.append(img)
	undist=cv2.undistort(img, mtx, dist,None,mtx)
	undist_images.append(undist)
	# convert to grayscale
	gray = cv2.cvtColor(undist,cv2.COLOR_BGR2GRAY)
	[h,w] = np.shape(gray)
	masked_binary = laneLineMasking(undist)
	print([h,w])
	#src_int = np.array([[[100,696],[628,433],[665,433],[1120,696]]],dtype = np.int32)
	#region_masked = region_of_interest(masked_binary,src_int)#np.array([src], dtype =np.int32)
	#apply perspective
	warped_binary =  cv2.warpPerspective(masked_binary, M, (w,h))
	#masked_binary = laneLineMasking(destination)
	warped_detectedlanes = findLaneLinesConv(warped_binary)#[250:650,:]
	#warped_detectedlanes = initializeLaneLines(warped_binary[250:650,:])
	f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(24, 9))
	f.tight_layout()
	ax1.imshow(undist)
	ax1.set_title(fnames[i], fontsize=50)
	ax2.imshow(warped_binary)
	ax2.set_title('Image corrected for perspective', fontsize=5)
	ax3.imshow(warped_detectedlanes)
	ax3.set_title('After Minv', fontsize=5)
	plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.1)
	plt.show()
"""


