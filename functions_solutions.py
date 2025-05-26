import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
# from scipy.ndimage import convolve
import cv2
from cv2 import filter2D
from skimage.feature import peak_local_max

#################### Week 2 ####################
def getGaussDerivative(t):
    '''
    Computes kernels of Gaussian and its derivatives.
    Parameters
    ----------
    t : float
        Vairance - t.

    Returns
    -------
    g : numpy array
        Gaussian.
    dg : numpy array
        First order derivative of Gaussian.
    ddg : numpy array
        Second order derivative of Gaussian
    dddg : numpy array
        Third order derivative of Gaussian.

    '''

    kSize = 5
    s = np.sqrt(t)
    x = np.arange(int(-np.ceil(s*kSize)), int(np.ceil(s*kSize))+1)
    x = np.reshape(x,(-1,1))
    g = np.exp(-x**2/(2*t))
    g = g/np.sum(g)
    dg = -x/t*g
    ddg = -g/t - x/t*dg
    dddg = -2*dg/t - x/t*ddg
    return g, dg, ddg, dddg

def get_gauss_derivatives(t, trunc=5):
    """
    Returns Gaussian kernel and its derivatives up to the third order.

    Parameters:
    t (float): The variance of the Gaussian.
    trunc (float, optional): The truncation value for the kernel size. Defaults to 5.

    Returns:
    tuple: A tuple containing the kernels as numpy arrays.
    """
    s = np.sqrt(t)
    r = np.ceil(trunc * s)
    x = np.arange(-r, r + 1)
    g = np.exp(-x**2 / (2 * t))
    g = g / np.sum(g)
    dg = -(x / t) * g
    ddg = -g / t - (x / t) * dg
    dddg = -2 * dg / t - (x / t) * ddg
    return g, dg, ddg, dddg


def separable_filtering(image, kernelx, kernely):
    """
    Applies separable filtering to an image.

    Args:
        image: The input image to be filtered.
        kernelx: The kernel for first axis.
        kernely: The kernel for second axis.

    Returns:
        The filtered image.
    """
    kernelx = kernelx.reshape(1, -1)
    kernely = kernely.reshape(-1, 1)
    return filter2D(filter2D(image, -1, kernelx), -1, kernely)

def detect_fibers(im, diameter_limits, nr_steps, t_detection, thres, detect_min=False):
    """
    Detects fibers in images by finding maxima of Gaussian smoothed image.

    Parameters:
    - im: numpy array, input image
    - diameter_limits: tuple, min and max of the fiber diameter (in pixels)
    - nr_steps: int, number of steps for scale variation
    - t_detection: float, scale parameter for fiber center detection
    - thres: float, blob magnitude threshold for fiber detection

    Returns:
    - coord: numpy array, coordinates of detected fiber centers
    - scale: numpy array, scale of detected fibers
    """
        
    t_values = np.linspace(diameter_limits[0]**2 / 8, 
                           diameter_limits[1]**2 / 8, nr_steps, endpoint=True)
    
    L_blob_vol = np.zeros(im.shape + (len(t_values),))
    for i, t in enumerate(t_values):
        g, dg, ddg, dddg = get_gauss_derivatives(t)
        L_blob_vol[:, :, i] = t * (separable_filtering(im, g, ddg) +  
                                   separable_filtering(im, ddg, g))
    # Detect fibre centers
    g, dg, ddg, dddg = get_gauss_derivatives(t_detection)
    Lg = separable_filtering(im, g, g)
    if detect_min:
        coord = peak_local_max(-Lg, threshold_abs=thres)
    else:
        coord = peak_local_max(Lg, threshold_abs=thres)
    # Find coordinates and size (scale) of fibres
    magnitudeIm = L_blob_vol.min(axis = 2)
    scaleIm = L_blob_vol.argmin(axis = 2)
    
    scales = scaleIm[coord[:,0], coord[:,1]]
    magnitudes = - magnitudeIm[coord[:,0], coord[:,1]]
    idx = np.where(magnitudes > thres)[0]
    coord = coord[idx]
    scale = t_values[scales[idx]]
    return coord, scale

def get_circles(coord, radii):
    """
        Compute coordinates for drawing circles around detected blobs.

    Parameters:
    coord (numpy.ndarray): Array of shape (n, 2) with the circle centers.
    scale (numpy.ndarray): Array of length n with circle radii.

    Returns:
    numpy.ndarray: Array of shape (n, 91) with the x-coordinates n circles.
    numpy.ndarray: Array of shape (n, 91) with the y-coordinates of n circles.
    """
    
    theta = np.linspace(0, 2 * np.pi, 91, endpoint=True)
    circ = np.array((np.cos(theta), np.sin(theta)))
    circ_y = np.outer(circ[0], radii) + coord[:, 0]
    circ_x = np.outer(circ[1], radii) + coord[:, 1]

    return circ_x, circ_y










######################## Week 3 ########################

def get_transformation(p, q):
    '''
    Compute the transformation parameters of the equation:
        q = s * R @ p + t

    Parameters
    ----------
    p : numpy array
        2 x n array of points.
    q : numpy array
        2 x n array of points. p and q corresponds.

    Returns
    -------
    R : numpy array
        2 x 2 rotation matrix.
    t : numpy array
        2 x 1 translation matrix.
    s : float
        scale parameter.

    '''
    m_p = np.mean(p,axis=1, keepdims=True)
    m_q = np.mean(q,axis=1, keepdims=True)
    s = np.linalg.norm(q - m_q, axis=0).sum() / np.linalg.norm(p - m_p, axis=0).sum()
    C = (q - m_q) @ (p - m_p).T
    U, S, V = np.linalg.svd(C)
    
    R_ = U @ V
    R = R_ @ np.array([[1, 0],[0, np.linalg.det(R_)]])
    
    t = m_q - s * R @ m_p
    return R, t, s

# Robust transformation

def get_robust_transformation(p, q, thres = 3):
    '''
    Compute the transformation parameters of the equation:
        q = s * R @ p + t

    Parameters
    ----------
    p : numpy array
        2 x n array of points.
    q : numpy array
        2 x n array of points. p and q corresponds.

    Returns
    -------
    R : numpy array
        2 x 2 rotation matrix.
    t : numpy array
        2 x 1 translation matrix.
    s : float
        scale parameter.
    idx : numpy array
        index of the points in p and q that are inliers in the robust match

    '''
    
    R,t,s = get_transformation(p, q)

    q_1 = s * R @ p + t
    d = np.linalg.norm(q - q_1, axis=0)
    idx = np.where(d < thres)[0]
    print(idx)
    R,t,s = get_transformation(p[:,idx], q[:,idx])
    
    return R, t, s, idx


def match_SIFT(im1, im2, thres = 0.6):
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # Apply ratio test
    good_matches = []
    
    for m,n in matches:
        if m.distance/(n.distance+10e-10) < thres:
            good_matches.append([m])
    
    # Find coordinates
    pts_im1 = [kp1[m[0].queryIdx].pt for m in good_matches]
    pts_im1 = np.array(pts_im1, dtype=np.float32).T
    pts_im2 = [kp2[m[0].trainIdx].pt for m in good_matches]
    pts_im2 = np.array(pts_im2, dtype=np.float32).T
    return pts_im1, pts_im2


def find_nearest(p,q,dd=3):
    idx_pq = np.zeros((p.shape[0]), dtype=np.int_)
    d_pq = np.zeros((p.shape[0])) + 10e10
    idx_qp = np.zeros((q.shape[0]), dtype=np.int_)
    for i in range(p.shape[0]):
        d = np.sum((q-p[i,:])**2,axis=1)
        idx_pq[i] = np.argmin(d)
        d_pq[i] = d[idx_pq[i]]
    for i in range(q.shape[0]):
        d = np.sum((p-q[i,:])**2,axis=1)
        idx_qp[i] = np.argmin(d)
    
    p_range = np.arange(0,p.shape[0])
    match = idx_qp[idx_pq] == p_range
    idx_p = p_range[match*(d_pq < dd**2)]
    
    idx_q = idx_pq[match*(d_pq < dd**2)]
    return idx_p, idx_q

