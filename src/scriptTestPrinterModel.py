'''
Created on 18.02.2013
    In that script I'm testing various printer models. But the main idea is to compare
    NG cellular model and the one developed by the swedish people not using n factor, but the
    NG primaries, the Demichel model and an effective coverage map.
    In both cases, what I need are grid data in CMY to XYZ space. I need to build the relationship
    between the colorant values and their effective coverage values.
@author: gerjer
'''
from colorTools import *
from colorConversion import *
from colorPrinterModel import *
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

# create the colorant grid for test
vecGrid = np.hstack([np.arange(0,100,50),100])
C,M,Y = create3Dgrid(vecGrid)
CMY = np.transpose(np.hstack([C,M,Y]))/100


print vecGrid, CMY, np.shape(CMY), np.shape(CMY)

# test Demichel
w = Demichel3Colorants(CMY)
print w