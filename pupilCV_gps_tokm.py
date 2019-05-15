import matplotlib.pyplot as plt
import xml.dom.minidom as mnd
import math
import datetime
import numpy as np
from bisect import bisect_left
import csv
import calendar

import matplotlib.cm as cm
from adjustText import adjust_text
import colorsys
cmap = cm.jet

import time as tm
from pupil_tools.data_tools import *
from pupil_tools.colour_tools import *
from pupil_tools.signal_tools import *


##### deifne the user home folder path
from os.path import expanduser
home = expanduser("~")
export_source_alt=home+"/recordings/all_data/maps"


##### cofig #####

##### define the recordings folder


#recording_source=home+"/recordings/Navigator/"
recording_source=home+"/recordings/Assistant/"

recording_name="A5A"
recording_nameB="A5B"

dataB =True

data_source=recording_source+recording_name
data_sourceB=recording_source+recording_nameB
export_source=data_source+"/exports/000"
export_sourceB=data_sourceB+"/exports/000"

data=open(data_source+'/gps_track.gpx')

#READ GPX FILE

xmldoc = mnd.parse(data)
track = xmldoc.getElementsByTagName('trkpt')
elevation=xmldoc.getElementsByTagName('ele')
datetime=xmldoc.getElementsByTagName('time')
n_track=len(track)



first_row = True
#Read distance
distanceVal=[]
rawDistanceVal=[]

#Read distance
distanceTime=[]
rawDistanceTime=[]
BDistanceTime=0
with open(export_source+"/pupilOutputDistance.csv") as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    for row in csvReader:
        if first_row:
            first_row=False
        else:
            rawDistanceVal.append(float(row[0]))
            rawDistanceTime.append(float(row[4]))

    csvDataFile.close() 

if dataB:

    first_row = True
    with open(export_sourceB+"/pupilOutputDistance.csv") as csvDataFile:
        csvReader = csv.reader(csvDataFile)
    
        for row in csvReader:
            if first_row:
                first_row=False

            else:
                if BDistanceTime==0 :
                    BDistanceTime=float(row[4])

                rawDistanceVal.append(float(row[0]))
                rawDistanceTime.append(float(row[4]))
    
        csvDataFile.close() 


for i in range( int(( rawDistanceTime[-1]- rawDistanceTime[0])/2)):
    time_r= rawDistanceTime[0]+i*2


    val = findClosestsAndIterpolate( time_r , rawDistanceTime, rawDistanceVal)

    distanceVal.append(val)
    distanceTime.append(time_r)




#PARSING GPX ELEMENT
lon_list=[]
lat_list=[]
h_list=[]
time_list=[]
epoch_list=[]


for s in range(n_track):
    lon,lat=track[s].attributes['lon'].value,track[s].attributes['lat'].value
    elev=elevation[s].firstChild.nodeValue
    lon_list.append(float(lon))
    lat_list.append(float(lat))
    h_list.append(float(elev))
    # PARSING TIME ELEMENT
    dt=datetime[s].firstChild.nodeValue

    time_split=dt.split('T')
    hms_split=time_split[1].split(':')
    time_hour=int(hms_split[0])
    time_minute=int(hms_split[1])
    time_second=int(hms_split[2].split('Z')[0])
    'Jul 9, 2009 @ 20:02:58 UTC'
    time_secondEph = calendar.timegm(tm.strptime(dt, '%Y-%m-%dT%H:%M:%SZ'))
    #print(time_secondEph)
    #print(dt)


    total_second=time_hour*3600+time_minute*60+time_second
    time_list.append(total_second)
    epoch_list.append(int(time_secondEph))




#GEODETIC TO CARTERSIAN FUNCTION
def geo2cart(lon,lat,h):
    a=6378137 #WGS 84 Major axis
    b=6356752.3142 #WGS 84 Minor axis
    e2=1-(b**2/a**2)
    N=float(a/math.sqrt(1-e2*(math.sin(math.radians(abs(lat)))**2)))
    X=(N+h)*math.cos(math.radians(lat))*math.cos(math.radians(lon))
    Y=(N+h)*math.cos(math.radians(lat))*math.sin(math.radians(lon))
    return X,Y

#DISTANCE FUNCTION
def distance(x1,y1,x2,y2):
    d=math.sqrt((x1-x2)**2+(y1-y2)**2)
    return d

#SPEED FUNCTION
def speed(x0,y0,x1,y1,t0,t1):
    d=math.sqrt((x0-x1)**2+(y0-y1)**2)
    delta_t=t1-t0
    if delta_t==0:
        delta_t=0.001
    s=float(d/delta_t)
    return s

#POPULATE DISTANCE AND SPEED LIST
d_list=[0.0]
speed_list=[0.0]
l=0
start_dist=0
for k in range(n_track-1):
    if k<(n_track-1):
        l=k+1
    else:
        l=k
    XY0=geo2cart(lon_list[k],lat_list[k],h_list[k])
    XY1=geo2cart(lon_list[l],lat_list[l],h_list[l])

    d=distance(XY0[0],XY0[1],XY1[0],XY1[1])
    sum_d=d+d_list[-1]
    d_list.append(sum_d)
    
    if start_dist==0:
        if lat_list[k]<60.372644:
            start_dist=sum_d


    
    #SPEED
    s=speed(XY0[0],XY0[1],XY1[0],XY1[1],time_list[k],time_list[l])
    speed_list.append(s)

d_list= [x - start_dist  for x in d_list]
#PLOT TRACK
#f,(track,speed,elevation)=plt.subplots(3,1)
f,(track)=plt.subplots(1,1,figsize=(6,11))
#f.set_figheight(8)
#f.set_figwidth(2)

#plt.subplots_adjust(hspace=0.5)
track.set_aspect(0.5)
img = plt.imread("/Users/giovanni/Desktop/AAA_filed_test/map.jpg")
track.imshow(img, zorder=0, extent=[5.1272, 5.2515, 60.2750, 60.4057])
track.plot(lon_list,lat_list,'k')
track.set_ylabel("Latitude")
track.set_xlabel("Longitude")

track.set_title(recording_name+" Track Plot")
if dataB:
    track.set_title(recording_name+"_"+recording_nameB+" Track Plot")



track.set_aspect(2.1663)
track.set_xlim((5.1272, 5.22))
track.set_ylim((60.2750, 60.38))

#PLOT SPEED
#speed.bar(d_list,speed_list,30,color='w',edgecolor='w')
#speed.set_title("Speed")
#speed.set_xlabel("Distance(m)")
#speed.set_ylabel("Speed(m/s)")

#PLOT ELEVATION PROFILE
base_reg=0
#elevation.plot(d_list,h_list)
#elevation.fill_between(d_list,h_list,base_reg,alpha=0.1)
#elevation.set_title("Elevation Profile")
#elevation.set_xlabel("Distance(m)")
#elevation.set_ylabel("GPS Elevation(m)")
#elevation.grid()

#ANIMATION/DYNAMIC PLOT

distanceVal_std = np.nanstd(distanceVal)
distanceVal_mean = np.nanmean(distanceVal)


distanceVal_min=min(distanceVal)
distanceVal_max=max(distanceVal)
distanceVal_var=distanceVal_max-distanceVal_min


for i in range(len(distanceTime)-1):

    lon_closestValue = findClosestsAndIterpolate( distanceTime[i] , epoch_list, lon_list )
    lat_closestValue = findClosestsAndIterpolate( distanceTime[i] , epoch_list, lat_list )

    track.plot(lon_closestValue,lat_closestValue ,marker='o',
                    markersize=8, mfc=(1,1,1,1), mec=(1,1,1,0.0))



texts=[]
distance_interv=500


dist_step_srt=[]
dist_step_num=[]
dist_step_end=[]
wl_val=[]



#distance tf
for i in range( int(d_list[-1]/distance_interv  )-1):

    d_beg= i*distance_interv
    d_mid= d_beg+distance_interv/2
    d_end= d_beg+distance_interv

    d_beg_t= findClosestsAndIterpolate( d_beg ,  d_list , epoch_list)
    d_end_t= findClosestsAndIterpolate( d_end ,  d_list , epoch_list)
    d_mid_t= (d_beg_t+d_end_t)/2


    lon_closestValue_beg = findClosestsAndIterpolate( d_beg_t , epoch_list, lon_list )
    lat_closestValue_beg = findClosestsAndIterpolate( d_beg_t , epoch_list, lat_list )
    
    lon_closestValue = findClosestsAndIterpolate( d_mid_t , epoch_list, lon_list )
    lat_closestValue = findClosestsAndIterpolate( d_mid_t , epoch_list, lat_list )

    lon_closestValue_end = findClosestsAndIterpolate( d_end_t , epoch_list, lon_list )
    lat_closestValue_end = findClosestsAndIterpolate( d_end_t , epoch_list, lat_list )


    index_beg= findClosest( d_beg_t, distanceTime )[1]
    index_end= findClosest( d_end_t, distanceTime )[1]
   
    if index_beg!=index_end:
      

        meanWl= np.nanmean(distanceVal[index_beg:index_end])



        c = (meanWl + distanceVal_std*1.5) /((distanceVal_std*1.5)*2)
    #c = (distanceVal[i] - distanceVal_min) /(distanceVal_var)
        if c<0:
            c=0
        if c>1:
            c=1
        c=0.4-c*0.4

        c=colorsys.hsv_to_rgb(c,1,1)


        r=c[0]
        g=c[1]
        b=c[2]

        dist_step_num.append(i)
        dist_step_srt.append(d_beg/1000)
        dist_step_end.append(d_end/1000)
        wl_val.append(meanWl)
    
       

        track.plot(lon_closestValue_end,lat_closestValue_end ,marker='.',
                        markersize=1, mfc=(0,0,0,1), mec=(0,0,0,1))
        track.plot(lon_closestValue_beg,lat_closestValue_beg ,marker='.',
                        markersize=1, mfc=(0,0,0,1), mec=(0,0,0,1))
        track.plot(lon_closestValue,lat_closestValue ,marker='o',
                        markersize=8.1, mfc=(r,g,b,1), mec=(0,0,0,0.0))
        texts.append(track.text(lon_closestValue+0.001,lat_closestValue, str(i)+" "+str( round(  d_beg/1000,1) )+" to "+str( round(  d_end/1000,1) )+"km", fontsize=8))
        #texts.append(track.text(lon_closestValue,lat_closestValue, str(i), fontsize=8))


adjust_text(texts, only_move={'texts':'x'})
print(wl_val)

csv_header = ["dist_step_num","partecipant","dist_step_str","dist_step_end","wl_val","wl_percived"]

csv_rows = [dist_step_num,recording_name,dist_step_srt,dist_step_end,wl_val,0]

saveCsv(export_source_alt,recording_name+"_wl_dist.csv",csv_header,csv_rows)




plt.savefig(export_source_alt+'/plot_alt'+recording_name+'.pdf', bbox_inches='tight')
plt.show()