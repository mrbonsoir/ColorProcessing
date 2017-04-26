#
# Created 25th April 2017 - Jeremie Gerhardt
#
# A few function to simulate how CVD people see images.

import numpy as np
import matplotlib.image as image
import colorspacious as cs

# Transformation matrix for Protanope
lms2lms_transform = np.array([[0,2.02344,-2.52581], [0,1,0], [0,0,1]])

# Transformation matrix for Deuteranope
lms2lms_transform = np.array([[1,0,0], [0.494207,0,1.24827], [0,0,1]])

# Transformation matrix for Tritanope
lms2lms_transform = np.array([[1,0,0], [0,1,0], [-0.395913,0.801109,0]])

# I doubt we will end up here, but if we do nothing should happen.
lms2lms_transform = np.eye(3)

# Transformation RGB to LMS and LMS to RGB
rgb2lms = np.array([[17.8824,43.5161,4.11935],
                [3.45565,27.1554,3.86714],
                [0.0299566,0.184309,1.46709]])
lms2rgb = np.linalg.inv(rgb2lms)

# A very Simple Method for CVD simulation
def fun_simulate_cvd(img_rgb, type_cvd = "p"):
    """ The function transforms the input image to simulate its color rendering
    for the chosen type of cvd.

    To be a bit faster the image data is reshape from (m x n x 3) to (3 x (m x n))

    Then only vector x (3 x 3) operations are applied.

    Finally the matrix for computation is reshape into the original image shape.
    """

    img_rgb = img_rgb.astype("float") / 255

    if type_cvd == "p":
        # Transformation matrix for Protanope (another form of red/green color deficit)
        lms2lms_transform = np.array([[0,2.02344,-2.52581], [0,1,0], [0,0,1]])

    elif type_cvd == "d":
        # Transformation matrix for Deuteranope (a form of red/green color deficit)
        lms2lms_transform = np.array([[1,0,0], [0.494207,0,1.24827], [0,0,1]])

    elif type_cvd == "t":
        # Transformation matrix for Tritanope (a blue/yellow deficit - very rare)
        lms2lms_transform = np.array([[1,0,0], [0,1,0], [-0.395913,0.801109,0]])

    else:
        # I doubt we will end up here, but if we do nothing should happen.
        lms2lms_transform = np.eye(3)

    rgb2lms = np.array([[17.8824,43.5161,4.11935],
                        [3.45565,27.1554,3.86714],
                        [0.0299566,0.184309,1.46709]])
    lms2rgb = np.linalg.inv(rgb2lms)

    # reshape image as vector 3 x m
    R = img_rgb[:,:,0].flatten()
    G = img_rgb[:,:,1].flatten()
    B = img_rgb[:,:,2].flatten()
    img_as_vector = np.vstack([R, G, B])

    # RGB to LMS
    LMS = np.dot(rgb2lms, img_as_vector)

    # LMS to CVD
    _LMS = np.dot(lms2lms_transform, LMS)

    # _LMS to _RGB
    _RGB = np.dot(lms2rgb, _LMS)

    # reshape the image
    img_cvd = np.zeros_like(img_rgb)
    img_cvd[:,:,0] = _RGB[0,:].reshape((np.shape(img_rgb)[0],np.shape(img_rgb)[1]))
    img_cvd[:,:,1] = _RGB[1,:].reshape((np.shape(img_rgb)[0],np.shape(img_rgb)[1]))
    img_cvd[:,:,2] = _RGB[2,:].reshape((np.shape(img_rgb)[0],np.shape(img_rgb)[1]))

    # Save daltonized image
    img_cvd = img_cvd * 255
    img_cvd = np.clip(img_cvd, 0, 255)
    img_cvd = img_cvd.astype("uint8")

    return img_cvd

# A more advanced solution using Machado Method
def fun_simulate_cvd_Machado(img_rgb, type_cvd = "p", severity_factor = 100):
    """The function use the code from the module colorspacious where the Method
    from Machado had been implemented
    """

    if type_cvd == "p":
        cvd_space = {"name": "sRGB1+CVD",
                             "cvd_type": "protanomaly",
                             "severity": severity_factor}
    if type_cvd == "d":
        cvd_space = {"name": "sRGB1+CVD",
                             "cvd_type": "deuteranomaly",
                             "severity": severity_factor}
    if type_cvd == "t":
        cvd_space = {"name": "sRGB1+CVD",
                             "cvd_type": "tritanomaly",
                             "severity": severity_factor}

    img_cvd_sRGB = cs.cspace_convert(img_rgb, cvd_space, "sRGB1")
    img_cvd_sRGB = np.clip(img_cvd_sRGB,0, 255)
    img_cvd_sRGB = img_cvd_sRGB.astype("uint8")
    return img_cvd_sRGB
