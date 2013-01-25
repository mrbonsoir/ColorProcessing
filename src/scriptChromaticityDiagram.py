'''        
Created on 16.01.2013
here I want to display the chromaticity diagram of CIE 1931
'''

from colorTools import *
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

## And now we do something
vecGrid = np.hstack([np.arange(0,255,50),255])
# print vecGrid
r,g,b = create3Dgrid(vecGrid)
rgb = np.transpose(np.hstack([r,g,b]))/255.
XYZ = conversionRGB2XYZ(rgb)
xyz = conversionXYZ2xyz(XYZ)
print 'size of the data converted: '+str(np.shape(xyz)) 
displayChroma_xy(xyz[0,:],xyz[1,:],'Une bien belle figure')
print "hello I'm back again!"


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