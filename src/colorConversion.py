'''
Created on 16.01.2013
This thing should help to make any kind of color conversion, like XYZ to xyz...
@author: gerjer
'''

import math
import numpy as np

def conversion_RGB_to_XYZ(data_RGB):
    """ Conversion RGB data to XYZ using matrix operation.
    For now sRGB and D50 is assumed to be the basis.

    Args: 
        data_RGB (float or [floats]): 3 x n RGB point

    Output:
        data_XZY (float or [floats]): 3 x n XYZ point
    """
    
    if data_RGB.max() > 1:
        data_RGB = data_RGB/255.
        
    # To avoid having a point with [X Y Z] = [0 0 0] we will replace 
    # this vector by something just above 0:
    sum_RGB                = data_RGB[0,:] + data_RGB[1,:] + data_RGB[2,:]    
    index_zero             = np.nonzero(sum_RGB == 0)
    data_RGB[:,index_zero] = 0.000001

    for i in np.arange(0,np.shape(data_RGB)[0]):
        for j in np.arange(0,np.shape(data_RGB)[1]):
            if data_RGB[i,j] > 0.04045:
                data_RGB[i,j] = math.pow((data_RGB[i,j] + 0.055) / 1.055, 2.4)
            else:
                data_RGB[i,j] = data_RGB[i,j] / 12.92
   
    M = ([[0.4124564,  0.3575761,  0.1804375],
          [0.2126729,  0.7151522,  0.0721750],
          [0.0193339,  0.1191920,  0.9503041]])

    data_XYZ = np.dot(M, data_RGB) * 100
    
    return data_XYZ

def conversion_XYZ_to_xyz(XYZ):
    """ Conversion XYZ data to xyz as follows:
        x, y, z = X, Y, Z / sum(X,Y,Z)

    Args:
        XYZ (float or [floats]): chromaticyties XYZ of size 3 x n

    Output:
        xyz (float or [floats]): chromaticyties xyz of size 3 x m

    """
    # here we sum X + Y + Z:
    sum_XYZ       = XYZ[0, :] + XYZ[1, :] + XYZ[2, :] 
    index_sum_XYZ = np.nonzero(sum_XYZ > 0)
    sum_XYZ       = np.tile(sum_XYZ, (3, 1))

    # here we normalize the XYZ by its sum:
    xyz           = XYZ[:, index_sum_XYZ] / sum_XYZ[:, index_sum_XYZ] 

    return xyz

def conversion_XYZ_to_xyY(XYZ):
    """ Conversion XYZ to xyY

    Args:
        XYZ (float or [floats]) of size 3 x n
    
    Output:
        xyY (float or [floats]) of size 3 x n

    """
    sum_XYZ       = XYZ[0,:] + XYZ[1,:] + XYZ[2,:]
    index_sum_XYZ = np.nonzero(sum_XYZ > 0) 
    sum_XYZ       = np.tile(sum_XYZ,(3,1))
    xyY           = sum_XYZ
    xyY[:,ind]    = XYZ[:,index_sum_XYZ] / sum_XYZ[:,index_sum_XYZ]
    xyY[2,:]      = XYZ[1,:]

    return xyY

def conversion_XYZ_to_Lab(XYZ, type_white_point_XYZw='D50_31'): 
    ''' Conversion XYZ to Lab.

    Args: 
        XYZ (float [floats]) of size 3 x n points
    
    kwargs:
        type_point_shape (str): white point and CMF as follows
        if type_white_point_XYZw = 'A_31' then it is illuminant A and CMF CIE1931
        if type_white_point_XYZw = 'D55_64' then it is the illuminant D55 and CMF CIE1964
        and it works for illuminant A, D50, D55, D65, D75, Uniform

    Output:
        Lab (float [floats]) of size 3 x n points

    '''

    Lab = np.ones(np.shape(XYZ))
    
    # set the white point
    if type_white_point_XYZw == 'A_31':
        XYZr = np.array([109.85, 100, 35.58])
    elif type_white_point_XYZw == 'A_64':
        XYZr = np.array([111.14, 100, 35.20])
    elif type_white_point_XYZw == 'D65_31':
        XYZr = np.array([95.04, 100, 108.88])
    elif type_white_point_XYZw == 'D65_64':
        XYZr = np.array([94.81, 100, 107.32])
    elif type_white_point_XYZw == 'C_31':
        XYZr = np.array([98.07, 100, 118.22])
    elif type_white_point_XYZw == 'C_64':
        XYZr = np.array([97.29, 100, 116.14])
    elif type_white_point_XYZw == 'D50_31':
        XYZr = np.array([96.42, 100, 82.51])
    elif type_white_point_XYZw == 'D50_64':
        XYZr = np.array([96.72, 100, 81.43])
    elif type_white_point_XYZw == 'D55_31':
        XYZr = np.array([95.68, 100, 92.14])
    elif type_white_point_XYZw == 'D55_64':
        XYZr = np.array([95.80, 100, 90.93])
    elif type_white_point_XYZw == 'D75_31':
        XYZr = np.array([94.97, 100, 122.61])
    elif type_white_point_XYZw == 'D75_64':
        XYZr = np.array([94.42, 100, 120.64])
    elif type_white_point_XYZw == 'Uniform':
        XYZr = np.array([100, 100, 100])
    elif type_white_point_XYZw == 'CIC2010':
        XYZr = np.array([99.8082, 100, 99.5881])
    else: # a special white point 
        XYZr = np.array([122.9, 127.4, 98.5])
         
    # reshape the vector of white point
    XYZr          = np.tile(XYZr,XYZ.shape[1]).reshape(XYZ.shape[1],XYZ.shape[0]).transpose()
    var_XYZr      = XYZ / XYZr

    # do the CIELab conversion function    
    for i in np.arange(0,np.shape(var_XYZr)[0]):
        for j in np.arange(0,np.shape(var_XYZr)[1]):
            if (var_XYZr[i,j] <= 0.008856):
                var_XYZr[i,j] = 7.787*var_XYZr[i,j] + 16.0/116.0
            else:
                var_XYZr[i,j] = math.pow(var_XYZr[i,j],(1.0/3.0))

    # last step to obtain the CIELlab values
    Lab[0,:] = (116 * var_XYZr[1,:]) - 16
    Lab[1,:] = 500 * (var_XYZr[0,:] - var_XYZr[1,:])
    Lab[2,:] = 200 * (var_XYZr[1,:] - var_XYZr[2,:])

    return Lab
