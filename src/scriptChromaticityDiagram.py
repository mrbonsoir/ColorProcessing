'''     
18.12.2013   
A tiny script tp test the various functions in colorTools and colorConversion 
modules.
Jeremie Gerhardt
'''

from colorTools import *
from colorConversion import *
import matplotlib.pyplot as plt
import numpy as np
import math
import sys

# Create a 3D grid of RGB data
vec_grid = np.hstack([np.arange(0,255,92),255])
R,G,B    = create_3D_grid(vec_grid)
data_RGB = np.transpose(np.hstack([R,G,B]))/255
print 'Your grid of data is made of '+str(np.shape(data_RGB)[1])+' points.'
print 'And each point has '+str(np.shape(data_RGB)[0])+' coordinates.'

# Convert the RGB data to XYZ
data_XYZ = conversion_RGB_to_XYZ(data_RGB)

# Convert the XYZ data to xyz 
data_xyz = conversion_XYZ_to_xyz(data_XYZ)

# Display the chromaticity data xy 
fig = plt.figure()
plot_spectrum_locus_31()
plot_chroma_xy(data_xyz[0,:], data_xyz[1,:],['Rec709', 'Rec2020', 'ACES'], color_points="magenta", color_background="red", fill=True)
plt.savefig("chroma_diagram_xy.png")

# Convert the XYZ data to Lab 
data_Lab = conversion_XYZ_to_Lab(data_XYZ, 'D65_31')

# Display the chromaticity ab 
fig = plt.figure()
displayChroma_ab(data_Lab[1,:],data_Lab[2,:])
plt.savefig("chroma_diagram_ab.png")
plt.show()
