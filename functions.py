import skimage.io
import numpy as np
import matplotlib.pyplot as plt
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
        n (int): The factor to determine the radius of the kernel.
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
        n (int): The factor to determine the radius of the kernel.
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

def smoothing_explicit(data, LAMBDA, N=1):
    """
    Explicit (forward) smooting using the equation: (I - λL)X
    
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
    


def smoothing_implicit(data, LAMBDA, N=1):
    """
    Apply implicit smoothing to the input data using a specified smoothing equation: (I - λL)^(-1)X
    
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
    


def smoothing_alpha_beta(data, alpha, beta, N=1):
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

def curve_length(data):
    """
    Calculate the length of a curve given its data points.

    Parameters:
        data (numpy.ndarray): A 2D array of shape (n, 2) representing the coordinates of the curve points.

    Returns:
        float: The total length of the curve.
    """
    d = (np.sqrt(((data - np.roll(data, shift=1, axis=0))**2).sum(axis=1))).sum()
    return d



################################## Week 2 ###########################################

# 2.2.1  Computing Gaussian and its second order derivative

def gaussian_2grad(n, t):
    """
    Computes the second derivative of a Gaussian function.

    Parameters:
        n (int): The factor to determine the radius of the kernel.
        t (float): The variance (sigma squared) of the Gaussian function.

    Returns:
        numpy.ndarray: The second derivative of the Gaussian function evaluated at each point in x.
    """

    RADIUS = np.ceil(n*np.sqrt(t))
    x = np.arange(-RADIUS, RADIUS + 1)
    assert RADIUS > 3*np.sqrt(t), 'RADIUS must be larger than 3*sqrt(t)'

    kernel = (1/t + x**2/t**2) * np.exp(-x**2/(2*t))
    return kernel/np.sum(kernel)

# 2.1.2 Detecting blobs at one scale

def Laplacian(image, t, n=5):
    """
    Applies the Laplacian of Gaussian (LoG) operator to an image.

    Parameters:
        image (ndarray): The input image on which the Laplacian of Gaussian is to be applied.
        t (float): The standard deviation of the Gaussian kernel.
        n (int, optional): The size of the kernel. Default is 5.

    Returns:
    tuple: A tuple containing two ndarrays:
        - L_xx: The result of convolving the image with the second derivative of the Gaussian in the x direction.
        - L_yy: The result of convolving the image with the second derivative of the Gaussian in the y direction.
        
    """
    kernel_2grad = gaussian_2grad(n, t)
    kernel_gaussian = gaussian_1D_kernel(n, t)

    L_xx = convolve(image, kernel_2grad.reshape(1,-1))
    L_xx = convolve(L_xx, kernel_gaussian.reshape(-1,1))

    L_yy = convolve(image, kernel_2grad.reshape(-1,1))
    L_yy = convolve(L_yy, kernel_gaussian.reshape(1,-1))
    return L_xx, L_yy


def plot_image_with_circles(image, local_max=None, local_min=None, t=None, fig_size=(7,7), circles=True, linewidth=1, ax=None):
    """
    Plots an image with detected blobs highlighted by circles.
    Local maxima are plotted in red and local minima in blue.

    Parameters:
        image (ndarray): The input image to be displayed.
        local_max (ndarray or list, optional): Coordinates of the local maxima.
        local_min (ndarray or list, optional): Coordinates of the local minima.
        t (float or list): The variance(s) used for the detected blobs.
        fig_size (tuple, optional): The size of the figure to be created. Default is (7, 7).
        circles (bool, optional): Whether to draw circles around blobs. Default is True.
        linewidth (int, optional): Width of the circles. Default is 1.
        ax (matplotlib.axes.Axes, optional): The axis on which to plot. Default is None.

    Returns:
        None
    """
    # Check if ax is provided, otherwise create a new figure and axis
    if ax is None:
        fig, ax = plt.subplots(figsize=fig_size)
    
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.axis('off')

    # Plot local maxima (in red)
    if local_max is not None:
        if isinstance(local_max, list):
            for (points, scale) in zip(local_max, t):
                for maxima in points:
                    y, x = maxima
                    if circles:
                        c = plt.Circle((x, y), np.sqrt(2*scale), color='r', linewidth=linewidth, fill=False)
                        ax.add_patch(c)
                    else:
                        ax.plot(x, y, 'ro', markersize=linewidth)
        else:
            for maxima in local_max:
                y, x = maxima
                if circles:
                    c = plt.Circle((x, y), np.sqrt(2*t), color='r', linewidth=linewidth, fill=False)
                    ax.add_patch(c)
                else:
                    ax.plot(x, y, 'ro', markersize=linewidth)

    # Plot local minima (in blue)
    if local_min is not None:
        if isinstance(local_min, list):
            for (points, scale) in zip(local_min, t):
                for minima in points:
                    y, x = minima
                    if circles:
                        c = plt.Circle((x, y), np.sqrt(2*scale), color='b', linewidth=linewidth, fill=False)
                        ax.add_patch(c)
                    else:
                        ax.plot(x, y, 'bo', markersize=linewidth)
        else:
            for minima in local_min:
                y, x = minima
                if circles:
                    c = plt.Circle((x, y), np.sqrt(2*t), color='b', linewidth=linewidth, fill=False)
                    ax.add_patch(c)
                else:
                    ax.plot(x, y, 'bo', markersize=linewidth)

    # Title adjustment
    title = f"Detected blobs with variance: {t}\n"
    if local_max is not None and local_min is None:
        title += " (Maxima in Red)"
    elif local_min is not None and local_max is None:
        title += " (Minima in Blue)"
    elif local_max is not None and local_min is not None:
        title += " (Maxima in Red, Minima in Blue)"
    
    
    
    ax.set_title(title)

    # Only show the plot if ax is None (i.e., a new figure was created)
    if ax is None:
        plt.show()


def Laplacian(image, t, normalize=True, n=5):
    """
    Applies the Laplacian of Gaussian (LoG) operator to an image.

    Parameters:
        image (ndarray): The input image on which the Laplacian of Gaussian is to be applied.
        t (float): The standard deviation of the Gaussian kernel.
        n (int, optional): The size of the kernel. Default is 5.

    Returns:
        ndarray: The result of applying the Laplacian of Gaussian to the image.
    """
    kernel_2grad = gaussian_2grad(n, t)
    kernel_gaussian = gaussian_1D_kernel(n, t)

    L_xx = convolve(image, kernel_2grad.reshape(1,-1))
    L_xx = convolve(L_xx, kernel_gaussian.reshape(-1,1))

    L_yy = convolve(image, kernel_2grad.reshape(-1,1))
    L_yy = convolve(L_yy, kernel_gaussian.reshape(1,-1))

    L = L_xx + L_yy
    if normalize:
        return t*L
    else:
        return L

################################## Week 3 ###########################################

def Rotation_translation_scale(P, Q):
    """
    Calculate the rotation matrix, translation vector, and scaling factor 
    that transforms point set P to point set Q, considering both R and R.T.

    Parameters:
        P (numpy.ndarray): Original point set of shape (2, n)
        Q (numpy.ndarray): Transformed point set of shape (2, n)

    Returns:
        R (numpy.ndarray): Rotation matrix of shape (2, 2)
        t (numpy.ndarray): Translation vector of shape (2, 1)
        s (float): Scaling factor
    """
    
    # Calculate the scale
    mu_Q = np.mean(Q, axis=1).reshape(-1, 1) # mean of Q
    mu_P = np.mean(P, axis=1).reshape(-1, 1) # mean of P
    s = np.linalg.norm(Q - mu_Q) / np.linalg.norm(P - mu_P)

    # Calculate the rotation
    H = (Q - mu_Q) @ (P - mu_P).T
    U, _, Vt = np.linalg.svd(H)     # SVD decomposition and it will return the transpose of V therefore Vt
    R = Vt.T @ U.T
    R_T = R.T                       # Also consider the transpose of R

    # Calculate the translation for both rotations (this is done to ensure that we get the best rotational matrix)
    t = mu_Q - s * R @ mu_P
    t_T = mu_Q - s * R_T @ mu_P

    # Calculate the reconstructed points for both R and R_T
    Q_est = s * R @ P + t
    Q_est_T = s * R_T @ P + t_T

    # Calculate the reconstruction errors
    error_R = np.sum(np.linalg.norm(Q - Q_est, axis=0)**2)
    error_R_T = np.sum(np.linalg.norm(Q - Q_est_T, axis=0)**2)

    # Check determinant of R and R_T
    # Consider to add an abs value to allow for reflections
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6), "Determinant of R is not 1, check for reflection or skew."
    assert np.isclose(np.linalg.det(R_T), 1.0, atol=1e-6), "Determinant of R_T is not 1, check for reflection or skew."

    # Choose the rotation with the minimum error
    if error_R <= error_R_T:
        return R, t, s
    else:
        return R_T, t_T, s
