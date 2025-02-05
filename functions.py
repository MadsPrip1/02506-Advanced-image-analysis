import skimage.io
import numpy as np
from scipy.ndimage import convolve
from scipy.linalg import circulant

############################ Week 1 ##########################################

def get_gray_image(path):
    """
    Reads an image from the specified file path and converts it to a grayscale image.

    Parameters:
        path (str): The file path to the image.

    Returns:
        numpy.ndarray: The grayscale image as a float array.
    """
    image_original = skimage.io.imread(path)
    return skimage.img_as_float(image_original)

# 1.1.1 Task

def gaussian_1D_kernel(n, t):
    """
    Computes a 1D Gaussian kernel.

    Parameters:
        sigma (float): The standard deviation of the Gaussian distribution.
        t (float): The variance of the Gaussian distribution.

    Returns:
        numpy.ndarray: The normalized 1D Gaussian kernel.
    """
    sigma = np.sqrt(t)
    RADIUS = np.ceil(n * sigma)
    x = np.arange(-RADIUS, RADIUS + 1)

    kernel = np.exp(-x**2/(2*t))
    return kernel/np.sum(kernel)

def gaussian_1D_grad(n, t):
    """
    Compute the gradient of a 1D Gaussian function.

    Parameters:
        x (float or numpy.ndarray): The input value(s) at which to evaluate the gradient.
        t (float): The variance of the Gaussian distribution.

    Returns:
        float or numpy.ndarray: The gradient of the 1D Gaussian function evaluated at x.
    """
    sigma = np.sqrt(t)
    RADIUS = np.ceil(n * sigma)
    x = np.arange(-RADIUS, RADIUS + 1)

    return - x/t * gaussian_1D_kernel(n, t)


# 1.1.2 Task 

def segmentation_length(image):
    """
    Calculate the segmentation length of a binary image.

    The segmentation length is defined as the total number of transitions 
    between 0 and 1 (or 1 and 0) along both the vertical and horizontal 
    directions in the image.

    Parameters:
        image (numpy.ndarray): A 2D binary numpy array representing the image.

    Returns:
        int: The total number of transitions in the image.
    """
    count_vertically = np.sum(image[1:, :] != image[:-1,:])
    count_horizontally = np.sum(image[:, 1:] != image[:,:-1])
    return count_vertically + count_horizontally

# 1.1.3 Task

def smooting_normal(data, LAMBDA, N=1):
    """
    Apply smoothing to the input data using a specified smoothing equation: (I + λL)X
    
    Parameters:
        data (numpy.ndarray): The input data to be smoothed (the point should be at the row so the shape is (x,2)).
        LAMBDA (float): The smoothing parameter that controls the amount of smoothing.
        N (int, optional): The number of iterations to apply the smoothing. Default is 1.

    Returns:
        numpy.ndarray: The smoothed data after applying the smoothing equation.
    """
    len_data = len(data)
    I = np.eye(len_data)
    vec = np.zeros(len_data)
    vec[-1], vec[0], vec[1] = 1, -2, 1 

    # create matrix L and print for sanity check
    L = circulant(vec)
    matrix = (I + LAMBDA*L)
    image_smoothed = matrix @ data.copy()

    for i in range(N-1):
        image_smoothed = matrix @ image_smoothed

    return image_smoothed
    


def smooting_inv(data, LAMBDA, N=1):
    """
    Apply smoothing to the input data using a specified smoothing equation: (I - λL)^(-1)X
    
    Parameters:
        data (numpy.ndarray): The input data to be smoothed (the point should be at the row so the shape is (x,2)).
        LAMBDA (float): The smoothing parameter that controls the amount of smoothing.
        N (int, optional): The number of iterations to apply the smoothing. Default is 1.

    Returns:
        numpy.ndarray: The smoothed data after applying the smoothing equation.
    """
    len_data = len(data)
    I = np.eye(len_data)
    vec = np.zeros(len_data)
    vec[-1], vec[0], vec[1] = 1, -2, 1 

    # create matrix L and print for sanity check
    L = circulant(vec)
    matrix = np.linalg.inv(I - LAMBDA*L)
    image_smoothed = matrix @ data.copy()

    for i in range(N-1):
        image_smoothed = matrix @ image_smoothed

    return image_smoothed
    


def smooting_alpha_beta(data, alpha, beta, N=1):
    """
    Apply smoothing to the input data using a specified smoothing equation: (I - λL)^(-1)X
    
    Parameters:
        data (numpy.ndarray): The input data to be smoothed (the point should be at the row so the shape is (x,2)).
        alpha (float): Controls the elasticity (or length minimization).
        beta (float): Controls the rigidity (or curvature minimization)
        N (int, optional): The number of iterations to apply the smoothing. Default is 1.

    Returns:
        numpy.ndarray: The smoothed data after applying the smoothing equation.
    """
    len_data = len(data)
    I = np.eye(len_data)
    alpha_vec = np.zeros(len_data)
    alpha_vec[-1], alpha_vec[0], alpha_vec[1] = 1, -2, 1 

    beta_vec = np.zeros(len_data)
    beta_vec[-2], beta_vec[-1], beta_vec[0], beta_vec[1], beta_vec[2] = -1, 4, -6, 4, -1

    A = circulant(alpha_vec)
    B = circulant(beta_vec)
    matrix = np.linalg.inv(I - alpha*A - beta*B) 
    image_smoothed = matrix @ data.copy()

    for i in range(N-1):
        image_smoothed = matrix @ image_smoothed

    return image_smoothed
    

def smoothing_matrix(alpha, beta, N):
    """
    Generates a smoothing matrix using circulant matrices based on the given parameters.

    Parameters:
        alpha (float): The weight for the first circulant matrix.
        beta (float): The weight for the second circulant matrix.
        N (int): The size of the matrix.
    Returns:
        numpy.ndarray: The resulting smoothing matrix of size NxN.
        
    The function constructs two circulant matrices A and B using the provided alpha and beta values.
    It then computes the smoothing matrix as the inverse of (I - alpha*A - beta*B), where I is the identity matrix.
    """


    # Make a vector to be used in circulate
    alpha_vec = np.zeros(N)
    alpha_vec[-1], alpha_vec[0], alpha_vec[1] = 1, -2, 1 

    beta_vec = np.zeros(N)
    beta_vec[-2], beta_vec[-1], beta_vec[0], beta_vec[1], beta_vec[2] = -1, 4, -6, 4, -1

    A = circulant(alpha_vec)
    B = circulant(beta_vec)

    I = np.eye(N)
    return np.linalg.inv((I - alpha*A - beta*B))

# quiz 

def convolve_2d(image, kernel):
    """
    Applies a 2D convolution to an image using the specified kernel.

    Parameters:
        kernel (numpy.ndarray): The convolution kernel to be applied with shape (x,).
        image (numpy.ndarray): The input image to be convolved.

    Returns:
        numpy.ndarray: The convolved image.

    Note:
    This function performs the convolution in two steps:
    1. Convolves the image with the kernel reshaped to a column vector.
    2. Convolves the result with the kernel reshaped to a row vector.
    """
    image_blurred = convolve(image, kernel.reshape(-1,1))
    return convolve(image_blurred, kernel.reshape(1,-1))
