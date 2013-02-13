'''
Created on 16.01.2013
This thing should help to make any kind of color conversion, like XYZ to xyz...
@author: gerjer
'''

import math
import numpy as np
#from colorTools import allData_nmXYZxy1931, allData_nmXYZxy1964

def conversionRGB2XYZ(dataRGB):
    """
    RGB to XYZ conversion. Expects 0-255 RGB values.
    Also expect 3xn data to be converted and gives XYZ as 3xN
    For now sRGB and D50 is assumed to be the basis
    """
    
    if dataRGB.max() > 1:
        dataRGB = dataRGB/255.
        
    # To avoid having a point with [X Y Z] = [0 0 0] replace this vector by something
    # just above 0
    sumRGB = dataRGB[0,:] + dataRGB[1,:] + dataRGB[2,:]    
    ind = np.nonzero(sumRGB == 0)
    dataRGB[:,ind] = 0.000001

    for i in np.arange(0,np.shape(dataRGB)[0]):
        for j in np.arange(0,np.shape(dataRGB)[1]):
            if dataRGB[i,j] > 0.04045:
                dataRGB[i,j] = math.pow((dataRGB[i,j] + 0.055) / 1.055, 2.4)
            else:
                dataRGB[i,j] = dataRGB[i,j] / 12.92
   
    # do the dataXYZ = M x dataRGB for conversion for illuminant D50
    M = ([[0.4360747,  0.3850649,  0.1430804],
          [0.2225045,  0.7168786,  0.0606169],
          [0.0139322,  0.0971045,  0.7141733]])
    dataXYZ = np.dot(M, dataRGB) * 100
    return dataXYZ

def conversionXYZ2xyz(XYZ):
    """
    The function make the conversion XYZ to xyz where x,y,z = X,Y,Z / (X + Y + Z) 
    """
    sumXYZ     = XYZ[0,:] + XYZ[1,:] + XYZ[2,:]
    ind        = np.nonzero(sumXYZ > 0) # check that no element is divided by 0
    sumXYZ     = np.tile(sumXYZ,(3,1))
    xyz        = sumXYZ
    xyz[:,ind] = XYZ[:,ind] / sumXYZ[:,ind]
    return xyz

def conversionXYZ2xyY(XYZ):
    """
    The function make the conversion XYZ to xyY. 
    """
    sumXYZ     = XYZ[0,:] + XYZ[1,:] + XYZ[2,:]
    ind        = np.nonzero(sumXYZ > 0) # check that no element is divided by 0
    sumXYZ     = np.tile(sumXYZ,(3,1))
    xyY        = sumXYZ
    xyY[:,ind] = XYZ[:,ind] / sumXYZ[:,ind]
    xyY[2,:]   = XYZ[1,:]
    return xyY

def conversionXYZ2Lab(XYZ, XYZw): 
    '''
    XYZ are the 3 x N data to be converted
    XYZw is the XYZ of the white point reference for the transformation,
    if XYZr = 'A_31' then it is the XYZ of illuminant A and CMF CIE1931
    if XYTr = D55_64' then it is the XYZ of illuminant D55 and CMF CIE1964
    '''
    Lab = np.ones(np.shape(XYZ))
    
    # set the white point
    if XYZw == 'A_31':
        XYZr = np.array([109.85, 100, 35.58])
    elif XYZw == 'A_64':
        XYZr = np.array([111.14, 100, 35.20])
    elif XYZw == 'D65_31':
        XYZr = np.array([95.04, 100, 108.88])
    elif XYZw == 'D65_64':
        XYZr = np.array([94.81, 100, 107.32])
    elif XYZw == 'C_31':
        XYZr = np.array([98.07, 100, 118.22])
    elif XYZw == 'C_64':
        XYZr = np.array([97.29, 100, 116.14])
    elif XYZw == 'D50_31':
        XYZr = np.array([96.42, 100, 82.51])
    elif XYZw == 'D50_64':
        XYZr = np.array([96.72, 100, 81.43])
    elif XYZw == 'D55_31':
        XYZr = np.array([95.68, 100, 92.14])
    elif XYZw == 'D55_64':
        XYZr = np.array([95.80, 100, 90.93])
    elif XYZw == 'D75_31':
        XYZr = np.array([94.97, 100, 122.61])
    elif XYZw == 'D75_64':
        XYZr = np.array([94.42, 100, 120.64])
    elif XYZw == 'Uniform':
        XYZr = np.array([100, 100, 100])
    elif XYZw == 'CIC2010':
        XYZr = np.array([99.8082, 100, 99.5881])
    else: # a special white point 
        XYZr = np.array([122.9, 127.4, 98.5])
         
    # reshape the vector of white point
    XYZr         = np.tile(XYZr,XYZ.shape[1]).reshape(XYZ.shape[1],XYZ.shape[0]).transpose()
    varXYZr      = XYZ / XYZr

    # do the CIELab conversion function    
    for i in np.arange(0,np.shape(varXYZr)[0]):
        for j in np.arange(0,np.shape(varXYZr)[1]):
            if (varXYZr[i,j] <= 0.008856):
                varXYZr[i,j] = 7.787*varXYZr[i,j] + 16.0/116.0
            else:
                varXYZr[i,j] = math.pow(varXYZr[i,j],(1.0/3.0))

    # last step to obtain the CIELlab values
    Lab[0,:] = (116 * varXYZr[1,:]) - 16
    Lab[1,:] = 500 * (varXYZr[0,:] - varXYZr[1,:])
    Lab[2,:] = 200 * (varXYZr[1,:] - varXYZr[2,:])

    return Lab