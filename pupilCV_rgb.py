import cv2
import time as t

from pupil_tools.colour_tools import *
from pupil_tools.data_tools import *
from pupil_tools.signal_tools import *

import csv
import numpy as np
from os.path import expanduser

cv2.ocl.setUseOpenCL(True)
home = expanduser("~")


##### cofig #####

data_source=home+"/recordings/10"
data_source=home+"/Dropbox/recordings/2"

#data_source=home+"/Desktop/20190323124155554"
video_source=data_source+"/world.mp4"


cap = cv2.VideoCapture(video_source)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

video_w=1280
video_h=720

##### end cofig #####


##### read recond info #####

info = {}

with open(data_source+"/info.csv") as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	firstLine=True
	
	for row in csvReader:
		if firstLine:
			firstLine=False
		else:
	
			info[row[0]]= row[1]


try:
	video = info["World Camera Resolution"].split("x")

	video_w,video_h= int(video[0]),int(video[1])

	print (video_w,video_h)
except Exception as e:
	pass






export_source=data_source+"/exports/000"



##### read pupil_positions.csv #####

gaze_positions=[]

with open(export_source+"/gaze_positions.csv") as csvDataFile:
	csvReader = csv.reader(csvDataFile)
	firstLine = True
	for row in csvReader:
		if firstLine: #skip fisrstline (column declaration)
			firstLine=False
		else:

			gaze_positions.append(row)
			
csvDataFile.close()	
			  
##### end read pupil_positions.csv #####


prev_frame_index = 0
gaze_pix_positions=[]
gaze_frame_list_x=[]
gaze_frame_list_y=[]
gaze_frame_list_time=[]

prev_frame_x= 0
prev_frame_y= 0

start_time = t.time()

for gaze_sample in gaze_positions:

	frame_index =  int(gaze_sample[1])
	frame_time =  float(gaze_sample[0])


	if frame_index != prev_frame_index:

		gaze_frame_list_x=np.clip(gaze_frame_list_x, 0, video_w-1)
		gaze_frame_list_y=np.clip(gaze_frame_list_y, 0, video_h-1)

		gaze_pix_positions.append( (frame_index,gaze_frame_list_x,gaze_frame_list_y,gaze_frame_list_time) )
		gaze_frame_list_x=[]
		gaze_frame_list_y=[]
		gaze_frame_list_time=[]

	if float(gaze_sample[2])>0.6:

	
		gaze_frame_list_x.append(int( float(gaze_sample[3]) * video_w))
		gaze_frame_list_y.append(int( (1-float(gaze_sample[4])) * video_h))
	
		prev_frame_x=int( float(gaze_sample[3]) * video_w)
		prev_frame_i=int( (1-float(gaze_sample[4])) * video_h)

	else:
		gaze_frame_list_x.append(prev_frame_x)
		gaze_frame_list_y.append(prev_frame_y)



	gaze_frame_list_time.append(float(frame_time))
 

	prev_frame_index = frame_index


##### end read pupil_positions.csv #####

frame_index=-1
frame_index_alt=0
correction = 0


first_row = True

gaze_positions_x = []
gaze_positions_y = []

row = ["frame_index","time","AVGlum","SpotLum"]

# Check if came=[]ra opened successfully
if (cap.isOpened()== False): 
	print("Error opening video stream or file")

with open(data_source+'/outputFromVideo.csv', 'w') as csvFile:
	writer = csv.writer(csvFile)
	if first_row:
		writer.writerow(row)
		first_row=False
	
	# Read until video is completed
	while(cap.isOpened()):

		# Capture frame-by-frame
		ret, frame = cap.read()
		frame_index=frame_index+1
		


		if ret == True:

			gaze_frame_n = gaze_pix_positions[frame_index_alt][0]

			if frame_index+1 > gaze_frame_n and frame_index_alt+2 < len(gaze_pix_positions) :
				frame_index_alt=frame_index_alt+1
				gaze_frame_n = gaze_pix_positions[frame_index_alt][0]

			if gaze_frame_n == frame_index+1 :

				gaze_frame_n = gaze_pix_positions[frame_index_alt][0]
	
				gaze_frame_n = gaze_pix_positions[frame_index_alt][0]
	
	
				gaze_positions_x = gaze_pix_positions[frame_index_alt][1]
				gaze_positions_y = gaze_pix_positions[frame_index_alt][2]
				gaze_positions_time = gaze_pix_positions[frame_index_alt][3]
	
	

				frame_r , frame_g , frame_b = np.dsplit(frame, 3)
				R = np.average(frame_r)
				G = np.average(frame_g)
				B = np.average(frame_b)

				lum = relativeLuminanceClac(R,G,B)

				
				frame_blurr=cv2.GaussianBlur(frame,(15,15),cv2.BORDER_DEFAULT)
				#frame_blurr=cv2.blur(frame,(11,11),cv2.BORDER_DEFAULT)
				#frame_blurr=cv2.blur(frame_blurr,(21,21),cv2.BORDER_DEFAULT)
				#frame_blurr=cv2.blur(frame_blurr,(41,41),cv2.BORDER_DEFAULT)
				#frame_blurr=cv2.blur(frame_blurr,(81,81),cv2.BORDER_DEFAULT)


				frame_r_blurr,frame_g_blurr,frame_b_blurr = np.dsplit(frame_blurr, 3)

				lumBlurr = relativeLuminanceClac(R,G,B)

				R_blurr = np.average(frame_r_blurr)
				G_blurr = np.average(frame_g_blurr)
				B_blurr = np.average(frame_b_blurr)

				lumBlurr = relativeLuminanceClac(R_blurr,G_blurr,B_blurr)

				lumCoeff = lum/lumBlurr

				



				
	
				for i in range(0,len(gaze_positions_time)):

					R_pixval  = float(frame_r_blurr[gaze_positions_y [i],gaze_positions_x [i]])
					G_pixval  = float(frame_g_blurr[gaze_positions_y [i],gaze_positions_x [i]])
					B_pixval  = float(frame_b_blurr[gaze_positions_y [i],gaze_positions_x [i]])

					pixval = relativeLuminanceClac(R_pixval,G_pixval,B_pixval)
	
					row = [frame_index,gaze_positions_time [i],lum,lumCoeff*pixval]
		
					writer.writerow(row)
	

				if frame_index % 1000 == 0:
					print ( round((frame_index/length)*100),"%" )
					print(round((frame_index / (t.time() - start_time))))

			correction = gaze_frame_n - frame_index
		else:
			csvFile.close()	
			break

cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()