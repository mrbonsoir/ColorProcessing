# Filename: colorTools.py
'''
Created on 24.01.2013
My module for color tools. Create grid for test data...
@author: gerjer
'''
import math
import numpy as np
import matplotlib.pyplot as plt


ACES_x = np.array([0.7347, 0, 0.0001])
ACES_y = np.array([0.26539, 1, -0.077])

REC709_x = np.array([0.64, 0.3, 0.15])
REC709_y = np.array([0.33, 0.6, 0.06])

DCIP3_x = np.array([0.68, 0.265, 0.15])
DCIP3_y = np.array([0.32, 0.69, 0.06])

REC2020_x = np.array([0.708, 0.170, 0.131])
REC2020_y = np.array([0.292, 0.797, 0.046])


def create_3D_grid(x):
    """ Create a 3D grid from a vector of data in one dimension.
    The idea is to do someting equivalent as meshgrid, but for 3D and
    not only 2D.

    Args:
        x (float or [floats]): vector points

    Output:
        u, v, w (float or [floats]): coords

    """
   
    [u, v] = np.meshgrid(x, x)
    w = np.tile(np.ones(np.shape(u)),((np.size(x)),1))
    u = np.tile(np.reshape(u,(np.size(u),1)),(np.size(x),1)) 
    v = np.tile(np.reshape(v,(np.size(v),1)),(np.size(x),1))
               
    for ii in np.arange(0,np.size(x)):
        block = w[:,ii] * x[ii]
        w[: ,ii] = np.transpose(block)

    w = np.reshape(np.transpose(w),(np.size(u),1))
    
    return u, v, w

def plot_chroma_xy(x, y, types_standard_gamut=None, type_point_shape='o', color_points='gray', color_background='white',  fill=False, draw_lines=True):
    """ Plot chromaticities xy in chromaticity diagram

    Args:
        x,y (float or [floats]): coords

    kwargs:
        type_point_shape (str): matplotlib type. Ex: 'o', 'r+'
    
        color_points (str): matplotlib color. Ex: green, cyan

        color_background (str): color of the gamut triangle(s) background
           
        types_standard_gamut list(str): gamuts to display
    
        Available types_standard_gamut : Rec709, ACES, DCIP3, Rec2020
    
        drawLines (bool): draw outline

        fill (bool): fill triangle

    """
    if types_standard_gamut != None: 
        plot_standard_gamuts(types_standard_gamut, color_background, draw_lines, fill)
    
    plot_points(x, y, type_point_shape, color_points)
        
    # some options
    plt.xlabel('chromaticity x')
    plt.ylabel('chromaticity y')
    plt.grid(True)
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    plt.draw()

def plot_standard_gamuts(types, color=None, draw_lines=True, fill=False):
    """ Plot standard gamuts

    Args:
        types list(str): gamuts to display
        Available types : Rec709, ACES, Res2020, DCIP3

    kwargs:
        color (str): if none, 1st point--> red, 2nd --> green, 3rd -> blue

        drawLines (bool): draw outline

        fill (bool): fill triangle

    """
    for type in types:
        if type == "Rec709":
            plot_triangle(REC709_x, REC709_y, color, draw_lines, fill)
        elif type == "ACES":
            plot_triangle(ACES_x, ACES_y, color, draw_lines, fill)
        elif type == "DCIP3":
            plot_triangle(DCIP3_x, DCIP3_y, color, draw_lines, fill)
        elif type =="Rec2020":
            plot_triangle(REC2020_x, REC2020_y, color, draw_lines, fill)

def plot_points(x, y, type='o', color='gray'):
    """plot an xy points

    Args:
        x,y (float or [floats]): coords

    kwargs:
        type (str): matplotlib type. Ex: 'o', 'r+'

        color (str): matplotlib color. Ex: green, cyan

    """
    plt.plot(x, y, type, color=color)

def plot_triangle(x, y, color=None, draw_lines=True, fill=False):
    """plot an rgb triangle in xy

    Args:
        x,y (numpy.array): [r, g, b] coords

    kwargs:
        color (str): if none, 1st point--> red, 2nd --> green, 3rd -> blue

        drawLines (bool): draw outline

        fill (bool): fill triangle

    """
    if fill:
        plt.fill(x, y, color='grey', alpha='0.5')
    if draw_lines:
        indexVal = np.hstack([np.arange(x.size), 0])
        plt.plot(x[indexVal], y[indexVal], '-k')

    if color:
        plt.plot(x[0], y[0], 'o', x[1], y[1], 'o', x[2], y[2], 'o',
                 color=color)
    else:
        plt.plot(x[0], y[0], 'or', x[1], y[1], 'og', x[2], y[2], 'ob')

def plot_spectrum_locus(data_path):
    """ Plot standard spectrum locus

    Args:
        data_path (str): path to a file containing xyz data

    """
    data = np.loadtxt(data_path)
    xL = data[:, 1] / (data[:, 1] + data[:, 2] + data[:, 3])
    yL = data[:, 2] / (data[:, 1] + data[:, 2] + data[:, 3])
    plt.plot(xL, yL, 'k-')
    plt.plot(xL[[0, xL.size-1]], yL[[0, yL.size-1]], 'k:')

def plot_spectrum_locus_31():
    """ Plot CIE1931 spectrum locus

    """
    plot_spectrum_locus("AllData_xyz1931.txt")

def plot_spectrum_locus_64():
    """ Plot CIE1964 spectrum locus

    """
    plot_spectrum_locus("AllData_xyz1964.txt")


def displayChroma_ab(a, b, type_point_shape='o', color_points='gray'):
    """ Plot chromaticities ab in ab chromaticity diagram

    Args:
        a, b (float or [floats]): coords

    kwargs:
        type_point_shape (str): matplotlib type. Ex: 'o', 'r+'
    
        color_points (str): matplotlib color. Ex: green, cyan
    
    """
    plot_points(a, b, type_point_shape, color_points)   
        
    # some options
    plt.xlabel('chromaticity a')
    plt.ylabel('chromaticity b')
    plt.grid(True)
    plt.axis([-180, 180, -180, 180])
    plt.draw()
