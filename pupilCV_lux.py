
import csv
import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np
from datetime import *
from pupil_tools.data_tools import *
from pupil_tools.colour_tools import *
from pupil_tools.signal_tools import *
import scipy.fftpack
from collections import OrderedDict
from scipy.signal import freqz

##### deifne the user home folder path
from os.path import expanduser
home = expanduser("~")


##### cofig #####

##### define the recordings folder

#data_source="/Users/giovanni/Desktop/local_recording/20190225123115316
#recording_source=home+"/recordings/"
recording_source=home+"/Dropbox/recordings/"
#recording_source=home+"/Desktop/"

#recording_source="/Users/Giovanni/Desktop/local_recording"

recording_name="20190323124155554"
recording_name="3"
age 		  = 61
data_source=recording_source+recording_name


export_source=data_source+"/exports/000"
export_source_alt=home+"/Dropbox/recordings/all_data"



##### unified pupil size #####


fig, ax = plt.subplots(figsize=(20,10))
ax.set_ylim(-1, 10)
age = 26
referenceAge = 28.58 
nOfEye = 2
fieldAngle = 167

##### unified pupil size #####


pupil_data_mm = False
wl_baseline =0
useLux = True
useCamera = False
verbose = True
confidence_treshold = 0.6
filterForConf = True


##### end cofig #####







timelag = 0
#timelag = 321+1.408 # resincronize data from the external lux loggher
distSampleLenght = 10#(eye_frames 120fps)
sampleFreq = 120
sampleFreqCamera = 60
drawDist = True
distanceType = "euclidean"# "DTW" "euclidean" "sqeuclidean"
drawConfidence = False
exportCsv = True
exportDistance = False
drawSections =True

bandstop_filter = False
##### read recond info #####


if pupil_data_mm :
	pupil_coulmn = 13 # 13 in mm 6 in px
	pupil_offset = 0
	

else:
	pupil_coulmn = 6 # 13 in mm 6 in px
	pupil_offset = 0


pupilData = readPupil(export_source)
recordingInfo = readInfo(data_source)

#get Time from the info file

recStartTime = datetime.fromtimestamp(float(recordingInfo["Start Time (System)"]))
recStartTimeAlt = float(recordingInfo["Start Time (Synced)"])
bootTime = datetime.fromtimestamp(float(recordingInfo["Start Time (System)"])-recStartTimeAlt)
timeFromBoot= recStartTime-bootTime
recDuration = recordingInfo["Duration Time"].split(":")
recDurationSeconds = timedelta( seconds= ( int(recDuration[0])* 60 + int(recDuration[1])) * 60 + int(recDuration[2]))
recEndTime = recStartTime + recDurationSeconds

if verbose:
	print ("Reconding started at :", recStartTime )
	print ("Computer booted  at :", bootTime )
	print ("It was on for :" , timeFromBoot )
	print ("The recording lasted :" , recDuration )



recPupilValues,recTimeStamps,recFrames,recSimpleTimeStamps,recConfidence = processPupil(pupilData,pupil_coulmn,recStartTimeAlt,filterForConf,confidence_treshold)



recPupilValues = interpnan(recPupilValues)#remove nan form the pupil arrary

if bandstop_filter:
	order=3
	fs = sampleFreq
	lowcut = 0.047
	highcut = 0.5
	data = recPupilValues

	recPupilValues_filter_bandstop = butter_bandstop_filter(data, lowcut, highcut, fs, order)


#order = 1
#fs = sampleFreq     # sample rate, Hz
#cutoff = 0.047  # desired cutoff frequency of the filter, Hz
#data = recPupilValues

# Filter the data, and plot both the original and filtered signals.
#recPupilValues_filter = butter_lowpass_filter(data, cutoff, fs, order)

recPupilValues_filter = savgol_filter(recPupilValues, 1*sampleFreq+1, 2)

recPupilValues = savgol_filter(recPupilValues, int(sampleFreq/10)+1, 6)

recConfidence = savgol_filter(recConfidence, int(sampleFreq/10)+1, 6)







if useLux:

	luxTimeStamps,luxValues = readLux(recording_source,data_source,recStartTime,recEndTime)
	luxTimeStamps=[x - timelag for x in luxTimeStamps]
	luxValues = signal.savgol_filter(interpnan(luxValues), 10+1, 6) # filtered set of lux (10fps)

	luxValues = upsampleLux(luxTimeStamps,luxValues,recTimeStamps,recordingInfo,True)
	
	luxPupilValues = interpnan(calcPupil(luxValues,age,referenceAge,nOfEye,fieldAngle))

	meanLux = np.nanmean(luxPupilValues, axis=0)
	meanRec = np.nanmean(recPupilValues_filter, axis=0)

	stdLux = np.nanstd(luxPupilValues)
	stdRec = np.nanstd(recPupilValues_filter)
	
	pupil_coeff = meanLux/ meanRec

	pupil_coeff = ( meanLux-stdLux )/ (meanRec - stdRec ) 

	if verbose:
		print("pupil_coeff=",pupil_coeff )

	recPupilValues_scaled = [x * pupil_coeff  for x in recPupilValues]
	recPupilValues_translated = [x * pupil_coeff +  wl_baseline for x in recPupilValues]

	recPupilValues_filter_scaled = [x * pupil_coeff  for x in recPupilValues_filter]
	recPupilValues_filter_translated = [x * pupil_coeff +  wl_baseline for x in recPupilValues_filter]
	
	if bandstop_filter:
		recPupilValues_filter_bandstop_scaled = [x * pupil_coeff  for x in recPupilValues_filter_bandstop]
		recPupilValues_filter_bandstop_translated = [x * pupil_coeff - meanLux for x in recPupilValues_filter_bandstop]
		recSimpleTimeStamps_bandstop= [x - 7.2  for x in recSimpleTimeStamps]
 
		graphPlot(recSimpleTimeStamps_bandstop,recPupilValues_filter_bandstop_translated ,"gray",0.8,"Bandstop Filter Cognitive Wl")

	

	graphPlot(recSimpleTimeStamps,luxPupilValues ,"blue",0.8,"Sensor Calculated Pupil")
	
	if not useCamera:
		if verbose:
			graphPlot(recSimpleTimeStamps,recPupilValues_translated ,"gray",0.5,"Raw EyeTracker Pupil")
		
		graphPlot(recSimpleTimeStamps,recPupilValues_filter_translated,"black",0.8,"Smoothed EyeTracker Pupil")


if useCamera:

	indexLum , timeStampsLum , avgLum , spotLum  = readCamera(data_source)
	
	#spotLum = signal.savgol_filter(interpnan(interpzero(spotLum)), sampleFreqCamera*2+1 , 2)
	#avgLum = signal.savgol_filter(interpnan(interpzero(avgLum)), sampleFreqCamera*2+1  , 2)


	avgLum = upsampleLux(timeStampsLum , avgLum , recTimeStamps , recordingInfo , False)
	spotLum = upsampleLux(timeStampsLum , spotLum , recTimeStamps , recordingInfo , False)

	#graphPlot(recSimpleTimeStamps,spotLum,"purple",1,"Camera Calculated Pupil")
	#graphPlot(recSimpleTimeStamps,avgLum,"green",1,"Camera Calculated Pupil")


	scaledSpotLum=[]
	for i in range(0,len(recTimeStamps)):

		sensorLux = luxValues[i]
		cameraALum= avgLum[i] 
		cameraSLum= spotLum[i]

		cameraLum_min= sensorLux / (cameraALum *1000+1)
		cameraLum_max= cameraLum_min * 1001 

		#scaledSpot = (sensorLux * cameraSLum )/ cameraALum # proportion method

		scaledSpot = ((cameraLum_max * cameraSLum)+ (cameraLum_min * (1-cameraSLum)) )/2 # linear interpolation method

		scaledSpotLum.append(scaledSpot)

	scaledSpotLum = signal.savgol_filter(interpnan(interpzero(scaledSpotLum )), sampleFreq*3+1  , 1)

	#graphPlot(recSimpleTimeStamps,scaledSpotLum,"green",1,"scaledSpotLum.")

	spotPupilValues = calcPupil(scaledSpotLum,age,referenceAge,nOfEye,fieldAngle )

	meanLum = np.nanmean(spotPupilValues, axis=0)
	meanRec = np.nanmean(recPupilValues_filter, axis=0)

	stdLum= np.nanstd(spotPupilValues)
	stdRec = np.nanstd(recPupilValues_filter)
	pupilLum_coeff = meanLum/meanRec
	pupilLum_coeff = ( meanLum-stdLum )/ (meanRec - stdRec )
	
	if verbose:
		print("pupilLum_coeff=",pupilLum_coeff )

	recPupilValues_filter_scaled_Lum = [x * pupilLum_coeff for x in recPupilValues_filter]

	graphPlot(recSimpleTimeStamps,spotPupilValues,"orange",1,"Camera Calculated Pupil")

	graphPlot(recSimpleTimeStamps,recPupilValues_filter_scaled_Lum,"black",0.8,"Smoothed EyeTracker Pupil")






if drawDist:

	if useCamera:
		distanceVal , distanceTime = drawDistance( recPupilValues_filter_scaled_Lum, spotPupilValues, recSimpleTimeStamps, distSampleLenght, distanceType)
	else:
		distanceVal , distanceTime = drawDistance( recPupilValues_filter_translated, luxPupilValues, recSimpleTimeStamps, distSampleLenght, distanceType)

	
	## Number of samplepoints
	#N = len(distanceVal)
	## sample spacing

	#T = (1.0 / sampleFreq ) * distSampleLenght

	#x = distanceTime

	#y = distanceVal

	#yf = scipy.fftpack.fft(y)
	#xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
	#
	#fig, ax = plt.subplots()
	#ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
	#plt.show()


if drawConfidence:
	plt.plot(recTimeStamps, recConfidence,marker='o', markerfacecolor='blue', markersize=0.5, color='red', linewidth=0.5,label="Confidence")







if drawSections:
	#sections = [0,61,83,142,178,238,299,360,420,460] #orignal frode 
	timeCorrection=recStartTimeAlt+bootTime.timestamp()

	sections  = loadSections(data_source)
	if len(sections)==0:
		step=60

		for x in range(0,int(int(recDurationSeconds.total_seconds())/step)):

			beg = timeCorrection + x*step
			end = beg + step

			if x% 2 == 0: diff=1
			else: diff=1

			sections.append((beg,end,diff,"none"))

	drawRecSections(sections,timeCorrection,distanceVal,distanceTime)
	
	if exportCsv: 
		distanceSectionList,distanceNameList,distanceConditionList = findDistanceSection(sections,distanceTime,timeCorrection)

		








handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())

plt.xlabel('Time s')
plt.ylabel('Pupil diameter mm')
plt.title("CW"+ recording_name)

plt.savefig(export_source+'/plot'+recording_name+'.pdf', bbox_inches='tight')
plt.savefig(export_source_alt+'/plot_'+recording_name+'.pdf', bbox_inches='tight')

if exportCsv:
	csv_header = ["timestamp_unix","timestamp_relative","frame_n","confidence","mm_pupil_diameter_scaled","mm_pupil_diameter_calc_lux","px_pupil_diameter_raw","recording_name","age"]
	csv_rows   = [recTimeStamps,recSimpleTimeStamps,recFrames,recConfidence,recPupilValues_filter_translated,luxPupilValues,recPupilValues,recording_name,age]

	if useCamera:
		csv_header.append("mm_pupil_diameter_calc_camera")
		csv_rows.append(spotPupilValues)
		
	saveCsv(export_source,"pupilOutput.csv",csv_header,csv_rows)
	saveCsv(export_source_alt,recording_name+"_pupilOutput.csv",csv_header,csv_rows)


	if exportDistance:
		csv_header = ["distanceVal","distanceTime","recording_name","age"]
		csv_rows   = [distanceVal,distanceTime,recording_name,age]

		if drawSections:
			csv_header.append("section")
			csv_header.append("section_name")
			csv_header.append("light condition")

			csv_rows.append(distanceSectionList)
			csv_rows.append(distanceNameList)
			csv_rows.append(distanceConditionList)


	saveCsv(export_source,"pupilOutputDistance.csv",csv_header,csv_rows)
	saveCsv(export_source_alt,recording_name+"_pupilOutputDistance.csv",csv_header,csv_rows)

plt.show()





