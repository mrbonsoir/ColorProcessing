# Filename: colorTools.py
'''
Created on 24.01.2013
My module for color tools. Create grid for test data, plot data...
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

sRGB_x = np.array([0.64, 0.3, 0.15])
sRGB_y = np.array([0.33, 0.6, 0.06])

white_point_D65_x = np.array([0.3128])
white_point_D65_y = np.array([0.3290])

data_AllData_xyz1931 = np.loadtxt("/Users/jeremie/dev/myPythonModules/ColorProcessing/data/AllData_xyz1931.txt")
data_AllData_xyz1964 = np.loadtxt("/Users/jeremie/dev/myPythonModules/ColorProcessing/data/AllData_xyz1964.txt")

co_punctual_point_protan_xy = np.array([0.7635, 0.2365])
co_punctual_point_deuteran_xy = np.array([1.4, -0.4])
co_punctual_point_tritan_xy = np.array([0.1748,  0])

# In order for protan, deuteran, tritan
spec_JND_count = np.array([21, 31, 44])


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

def plot_spectrum_locus(data_name):
    """ Plot standard spectrum locus

    Args:
        data_path (str): path to a file containing xyz data

    """
    data = data_name
    xL = data[:, 1] / (data[:, 1] + data[:, 2] + data[:, 3])
    yL = data[:, 2] / (data[:, 1] + data[:, 2] + data[:, 3])
    plt.plot(xL, yL, 'k-')
    plt.plot(xL[[0, xL.size-1]], yL[[0, yL.size-1]], 'k:')

def plot_spectrum_locus_31():
    """ Plot CIE1931 spectrum locus

    """
    plot_spectrum_locus(data_AllData_xyz1931)

def plot_spectrum_locus_64():
    """ Plot CIE1964 spectrum locus

    """
    plot_spectrum_locus(data_AllData_xyz1964)


def displayChroma_ab(a, b, type_point_shape='o', color_points='gray'):
    """ Plot chromaticities ab in ab chromaticity diagram

    Args:
        a, b (float or [floats]): coords

    kwargs:
        type_point_shape (str): matplotlib type. Ex: 'o', 'r+'
<<<<<<< HEAD

        color_points (str): matplotlib color. Ex: green, cyan

=======

        color_points (str): matplotlib color. Ex: green, cyan

>>>>>>> 19ef348d1e94136d2ddd84264f07d361254b1197
    """
    plot_points(a, b, type_point_shape, color_points)

    # some options
    plt.xlabel('chromaticity a')
    plt.ylabel('chromaticity b')
    plt.grid(True)
    plt.axis([-180, 180, -180, 180])
    plt.draw()

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

def fun_plot_two_xy_datasets(data1, data2, marker1 = 'o', marker2 = 's', arrow=False):
    """
    The function displays two datasets of the same size.

    In:
        data1 (n x 2 float): xy first data set.
        data2 (n x 2 float) xy second data set.
    """

    # plot the 1931 locus
    plot_spectrum_locus_31()

    # plot the data
    plt.plot(data1[:,0], data1[:,1], marker1)
    plt.plot(data2[:,0], data2[:,1], marker2)

    if arrow == True:
        for i in np.arange(0,np.shape(data1)[0]):
            plt.plot([data1[i,0], data2[i,0]], [data1[i,1], data2[i,1]],'-k')


    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel("chroma x")
    plt.ylabel("chroma y")
    plt.grid()
    plt.draw()

def fun_plot_two_ab_datasets(data1, data2, marker1 = 'o', marker2 = 's', arrow=False):
    """
    The function displays two datasets of the same size.

    In:
        data1 (n x 2 float): ab first data set.
        data2 (n x 2 float) ab second data set.
    """
    # plot the data
    plt.plot(data1[:,0], data1[:,1], marker1)
    plt.plot(data2[:,0], data2[:,1], marker2)

    if arrow == True:
        for i in np.arange(0,np.shape(data1)[0]):
            plt.plot([data1[i,0], data2[i,0]], [data1[i,1], data2[i,1]],'-k')

    plt.xlim([-120, 120])
    plt.ylim([-120, 120])
    plt.xlabel("chroma a")
    plt.xlabel("chroma b")
    plt.grid()
    plt.draw()

def fun_plot_confusion_line(type_cvd = "deuteran"):
    """The function display the confusion lines for a chosen
    type of CVD.
    In:
        type_cvd (str): deuteran (default), protan or tritan

    """
    plot_spectrum_locus_31()

    # plot white points
    plt.plot(white_point_D65_x, white_point_D65_y,'xk')

    # plot co-punctual plot_point
    if type_cvd == "deuteran":

        plt.plot([white_point_D65_x, co_punctual_point_deuteran_xy[0]],
                 [white_point_D65_y, co_punctual_point_deuteran_xy[1]],':')

        # index 380nm to 780nm with delta (780 - 380) / nb_JND
        point_locus_xy = fun_get_index_for_JND(data_AllData_xyz1931[:,0:4], spec_JND_count[1]);
        for i in np.arange(spec_JND_count[1]):
            plt.plot([co_punctual_point_deuteran_xy[0], point_locus_xy[i,0]] ,
                     [co_punctual_point_deuteran_xy[1], point_locus_xy[i,1]],':k')

    elif type_cvd == "protan":

        plt.plot([white_point_D65_x, co_punctual_point_protan_xy[0]],
                 [white_point_D65_y, co_punctual_point_protan_xy[1]],':')

        # index 380nm to 780nm with delta (780 - 380) / nb_JND
        point_locus_xy = fun_get_index_for_JND(data_AllData_xyz1931[:,0:4], spec_JND_count[0]);
        for i in np.arange(spec_JND_count[0]):
            plt.plot([co_punctual_point_protan_xy[0], point_locus_xy[i,0]] ,
                     [co_punctual_point_protan_xy[1], point_locus_xy[i,1]],':k')


    elif type_cvd == "tritan":

        plt.plot([white_point_D65_x, co_punctual_point_tritan_xy[0]],
                 [white_point_D65_y, co_punctual_point_tritan_xy[1]],':')

        # index 380nm to 780nm with delta (780 - 380) / nb_JND
        point_locus_xy = fun_get_index_for_JND(data_AllData_xyz1931[:,0:4], spec_JND_count[2]);
        for i in np.arange(spec_JND_count[2]):
            plt.plot([co_punctual_point_tritan_xy[0], point_locus_xy[i,0]],
                     [co_punctual_point_tritan_xy[1], point_locus_xy[i,1]],':k')
    else:
        print "I'm not color blind obviously..."

    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])

def fun_get_index_for_JND(data_nm_X_Y_Z, JND_COUNT):
    """The function get the xy coordinates for a given wavelength nm on the locus
    border.
    """

    #step_nm = np.round(np.linspace(380, 780, JND_COUNT))
    step_nm = np.linspace(380, 780, JND_COUNT);

    # resample the X and Y curves at 1nm
    new_X = np.interp(step_nm, data_nm_X_Y_Z[:,0],data_nm_X_Y_Z[:,1])
    new_Y = np.interp(step_nm, data_nm_X_Y_Z[:,0],data_nm_X_Y_Z[:,2])
    new_Z = np.interp(step_nm, data_nm_X_Y_Z[:,0],data_nm_X_Y_Z[:,3])

    new_data_xy = np.zeros((JND_COUNT,2))
    new_data_xy[:,0] = new_X / (new_X + new_Y + new_Z);
    new_data_xy[:,1] = new_Y / (new_X + new_Y + new_Z);
    return new_data_xy

def fun_plot_Lab(Lab, fig, marker_shape = 'o'):
    """ The function display 3D CIELab data in a 3D diagram.
    The data Lab are assumed to be of shape (n x 3)"""
    #fig = plt.figure(figsize=(12,12))
    ax = fig.gca(projection='3d')
    #ax.scatter3D(Lab[:,1], Lab[:,2], Lab[:,0], marker_shape)
    ax.plot3D(Lab[:,1], Lab[:,2], Lab[:,0], marker_shape)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('L')

    ax.set_xlim3d(-120, 120)
    ax.set_ylim3d(-120, 120)
    ax.set_zlim3d(0,100)
    ax.axis("square")

def fun_create_Lab_circle(L_max, C, angle_step):
    """The function creates a circle in Lab space with constant Chroma and L."""
    angle = np.pi * np.arange(0,360, angle_step) / 180
    L = L_max * np.ones((len(angle),1))
    a = C*np.cos(angle)
    b = C*np.sin(angle)
    Lab = np.hstack([L, a.reshape(len(angle),1), b.reshape(len(angle),1)])
    return Lab


def fun_compute_most_important_color(img, angle_step = 90):
    """The function computes the most important colors in an image.
    To do so the image - assumed to be an RGB image - is converted to a cylindrical
    color shape.

    In:
        img (n x m x 3): probably float type of data.
        angle_step (uint8): a valut to divide the angle color value (defaut = 90
        for a division of 360 / 90)

    Out:
        tab_mvp (3 x p) for the p most important colors, at least 3 values.
    """

    # convert img_rgb to img_cylindric

    # count pixel into color slices
