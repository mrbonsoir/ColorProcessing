'''        
Created on 16.01.2013
here I want to display the chromaticity diagram of CIE 1931

I did not find a ways to close all the figure window when starting the script or to refresh only 
the selected figure.
'''

from colorTools import *
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

# Create a 3D grid of RGB data

#vecGrid = np.hstack([np.arange(0,255,10),255])
#R,G,B = create3Dgrid(vecGrid)
#RGB = np.transpose(np.hstack([R,G,B]))/255
#print RGB, np.shape(RGB)

RGB = np.array([[255,  255,   255,     0,   255,     0,     0,  50, 0],
                [255,  255,     0,   255,     0,   255,     0, 150, 0],
                [255,    0,   255,   255,     0,     0,   255,  50, 0]])
#print RGB, np.shape(RGB)

# Convert the RGB data to XYZ
XYZ = conversionRGB2XYZ(RGB)
print XYZ
# Convert the XYZ data to xyz 
xyz = conversionXYZ2xyz(XYZ)

# Display the chromaticity data xy 
displayChroma_xy(xyz[0,:],xyz[1,:],'Une bien belle figure CIE xy',1)

# Convert the XYZ data to Lab 
Lab = conversionXYZ2Lab(XYZ, 'D50_31')

# Display the chromaticity ab 
displayChroma_ab(Lab[1,:],Lab[2,:],'Une bien belle figure CIE ab',2)

# Don't forget this line to show all the figures
plt.show()

# To Do:
# - add color to the graph point like the corresponding rgb color
# - add convexhull of of Lab gamut
# - add gamut intersection function for xy diagram
# - add printer model
# - add projector model
# -  add metrics functions

'''
data = np.loadtxt("AllData_xyz1964.txt")
fileNameOutput = "tempData"
f = open('C:/WorkInPython/ColorProcessing/src/'+fileNameOutput+'.txt', 'w')
f.write('data=np.array([')
for i in np.arange(0,data.shape[0])-1:
    l = '['+'{:0.0f}'.format(data[i,0])+',\t'+'{:0.7f}'.format(data[i,1])+',\t'+'{:0.7f}'.format(data[i,2])+',\t'+'{:0.7f}'.format(data[i,3])+',\t'+'{:0.7f}'.format(data[i,4])+'],\n' 
    f.write(l)
l = '['+'{:0.0f}'.format(data[-1,0])+',\t'+'{:0.7f}'.format(data[-1,1])+',\t'+'{:0.7f}'.format(data[-1,2])+',\t'+'{:0.7f}'.format(data[-1,3])+',\t'+'{:0.7f}'.format(data[-1,4])+']'
f.write(l)
f.write('])')
f.close()'''