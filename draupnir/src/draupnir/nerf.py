'''
Implementation of nerf method
to calculate coords of a 4th
atom given the coords of the
first 3, the bond length (l),
bond angle (theta), and torsion
angle (chi)
Rohit Bhattacharya
rohit.bhattachar@gmail.com
'''

#ARTICLE: https://onlinelibrary.wiley.com/doi/10.1002/jcc.20237
# imports
import numpy as np

def nerf(a, b, c, l, theta, chi):
    '''
    Nerf method of finding 4th coord (d)
    in cartesian space
    Params:
    a, b, c : coords of 3 points
    l : bond length between c and d
    theta : bond angle between b, c, d (in degrees)
    chi : dihedral using a, b, c, d (in degrees)
    Returns:
    d : tuple of (x, y, z) in cartesian space
    '''

    # calculate unit vectors AB and BC
    ab_unit = (b-a)/np.linalg.norm(b-a)
    bc_unit = (c-b)/np.linalg.norm(c-b)

    # calculate unit normals n = AB x BC
    # and p = n x BC
    n_unit = np.cross(ab_unit, bc_unit)
    n_unit = n_unit/np.linalg.norm(n_unit)
    p_unit = np.cross(n_unit, bc_unit)

    # create rotation matrix [BC; p; n] (3x3)
    M = np.array([bc_unit, p_unit, n_unit]).T

    # convert degrees to radians
    theta = np.pi/180 * theta
    chi = np.pi/180 * chi

    # calculate coord pre rotation matrix
    d2 = [-l*np.cos(theta), l*np.sin(theta)*np.cos(chi), l*np.sin(theta)*np.sin(chi)]

    # calculate with rotation as our final output
    return c + np.dot(M, d2)