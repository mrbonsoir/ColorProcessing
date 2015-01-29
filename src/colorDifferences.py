'''
Created on 16.01.2013
This thing should store all functions for computing color differences such as
CIE Delta 76, 94 and so...
Jeremie Gerhardt
'''

import numpy as np
import math

def fun_difference_CIELab76(Lab1, Lab2):
    ''' Compute the Delta CIE 76 color difference between Lab1 and Lab2.

    Args:
        Lab1, Lab2 (float [floats]): two Lab values or 2 sets of Lab x np

    Output:
        difference_Lab (float [floats]): vector of size n.

    '''
    difference_Lab = np.sqrt(np.sum((Lab1-Lab2)*(Lab1-Lab2)))
    
    return difference_Lab