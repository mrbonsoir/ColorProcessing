'''
Created on 16.01.2013
This thing should store all functions for computing color differences such as CIE Delta E and so...
@author: gerjer
'''

import numpy as np

def funDifferenceLab(Lab1, Lab2):
    '''
    Lab1 and lab2 are 3xn size
    The output is 1xn size
    '''
    diffLab = np.sqrt(np.sum((Lab1-Lab2)*(Lab1-Lab2)))
    return diffLab

#def funDifferenceL(L1, L2):
	#'''
    #L1 and L are 1xn size
    #The output is 1xn size
    #'''
    #print L1, L2
    #diffL = L1#diffL = np.sqrt(np.sum((L1-L2)*(L1-L2)))
    #return diff