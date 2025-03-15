import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import scipy.interpolate

from scipy.ndimage import convolve
from scipy.linalg import circulant
from scipy.signal import find_peaks

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
    
def lowe_matches(des1, des2, lowe_threshold=0.6):
    """
    Apply Lowe's ratio test to filter good matches.

    Parameters:
        des1 (numpy.ndarray): Descriptors from the first image.
        des2 (numpy.ndarray): Descriptors from the second image.
        lowe_threshold (float): Threshold for Lowe's ratio test to filter good matches.

    Returns:
        good_matches (list): List of good matches after applying Lowe's ratio test.
    """
    
    # cv.BFMatcher: This is the brute-force matcher. It takes the descriptor of one feature in first set and is matched with all other features in second set using some distance calculation.
    # cv.NORM_L2: This specifies that we use Euclidean distance (L2 norm) to measure similarity.
    # crossCheck=False: Allows one-way matching, meaning we consider all potential matches without requiring mutual agreement.
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)

    # knnMatch() finds the two closest matches (best and second-best) for each descriptor in des1 from des2.
    matches = bf.knnMatch(des1, des2, k=2)        

    # Apply Lowe's ratio test to select good matches
    # m is the best match and n is the second best match to a given keypoint
    good_matches = []
    for m, n in matches:                            # m is the best match and n is the second best match
        if m.distance < lowe_threshold * n.distance:  # If the distance of the best match is less than the ratio times the distance of the second best match
            good_matches.append(m)   
    
    return good_matches






################################## Week 5 ###########################################

## 5.2 Task

def get_gray_image_255(image_path):
    """
    Converts an image path to a grayscale image in uint8 format.

    Parameters:
        image_path (str): The image path.

    Returns:
        numpy.ndarray: The grayscale image as a uint8 array (0-255).
    """
    image = skimage.io.imread(image_path)
    return skimage.img_as_ubyte(image)


def compute_V1(D, S, mu):
    """
    Computes the V1 energy term as the sum of squared differences between 
    the original grayscale image D and the intensity-realized version of S.

    Parameters:
        D (numpy.ndarray): The original grayscale image.
        S (numpy.ndarray): The segmentation (labels for each pixel).
        mu (dict): A dictionary mapping labels to mean intensities.

    Returns:
        float: The computed V1 energy.
    """
    # Create the intensity-realized image of S
    intensity_realized_S = np.vectorize(mu.get)(S) #Maps where the value of S is the key in mu to the value in mu which is the mean intensity of the label
    
    # Compute sum of squared differences
    V1 = np.sum((D - intensity_realized_S) ** 2)
    
    return V1

def compute_V2(S, beta):
    """
    Computes the V2 energy term as the sum of differences between neighboring pixels."
    """
    check_vertically   = beta*np.sum((S[1:, :] != S[:-1,:])) #Check if the pixel below is different from the pixel above
    check_horizontally = beta*np.sum((S[:, 1:] != S[:,:-1]))
    return check_vertically + check_horizontally

def compute_V1_and_V2(D, S, mu, beta, print_output=True):
    """
    Computes the V1 and V2 energy terms for a given segmentation.

    Parameters:
        D (numpy.ndarray): The original grayscale image.
        S (numpy.ndarray): The segmentation (labels for each pixel).
        mu (dict): A dictionary mapping labels to mean intensities.
        beta (float): The weight for the V2 term.

    Returns:
        tuple: A tuple containing the computed V1 and V2 energies.
    """
    V1 = compute_V1(D, S, mu)
    V2 = compute_V2(S, beta)
    if print_output:
        print(f'The likelihood energy term V1 is: {V1}')
        print(f'The prior term V2 is: {V2}')
        print(f'The posterior energy term V1+V2 is: {V1 + V2}')
    return V1, V2


def get_gray_image_float(image_path):
    """
    Converts an image path to a grayscale image in float format.

    Parameters:
        image_path (str): The image path.

    Returns:
        numpy.ndarray: The grayscale image as a float array.
    """
    # Read the image
    image = skimage.io.imread(image_path)

    # Check if the image is RGB and convert to grayscale if necessary
    if image.ndim == 3 and image.shape[2] == 3:
        image = skimage.color.rgb2gray(image)

    # Convert the image to float format
    return skimage.img_as_float(image)


def markov_segmentation(image, mu, beta):
    """
    Perform Markov Random Field segmentation on the given image.

    Parameters:
        image (numpy.ndarray): The input grayscale image to be segmented.
        mu (list): A list containing the mean intensities for the two classes.
        beta (float): The parameter controlling the smoothness of the segmentation.

    Returns:
        numpy.ndarray: The segmented image with labels for each pixel.
    """
    
    d = image.flatten()

    # Create a graph with integer capacities.
    g = maxflow.Graph[float]()

    # Add (non-terminal) nodes and retrieve an index for each node.
    nodes = g.add_nodes(len(d))

    # Create edges between nodes, that is equal to beta
    for i in range(len(d)-1):
        g.add_edge(nodes[i], nodes[i+1], beta, beta)

    # Set the capacities of the terminal edges.
    for i in range(len(d)):
        g.add_tedge(nodes[i], (d[i] - mu[1]) ** 2, (d[i] - mu[0]) ** 2)

    # Run the max flow algorithm.
    flow = g.maxflow()

    # Get the result as integer labels.
    labeling = [g.get_segment(n) for n in nodes]
    return np.array(labeling).reshape(image.shape)

def compute_and_plot_histogram(image, bins=256, num_max=1, tick_interval=0.05):
    """
    Compute the histogram of the input image, plot it, and return the histogram image and locations of local maxima.

    Parameters:
        image (numpy.ndarray): The input grayscale image.
        bins (int): The number of bins in the histogram.
        num_max (int): The number of local maxima to find.

    Returns:
        tuple: A tuple containing the indices of local maxima in the histogram.
    """
    # Compute the histogram
    hist, bin_edges = np.histogram(image.flatten(), bins=bins, range=(0, 1))
    
    # Find local maxima using scipy's find_peaks function
    local_maxima_indices, _ = find_peaks(hist)
    
    # Sort the local maxima by their heights (histogram values) in descending order
    local_maxima_indices = sorted(local_maxima_indices, key=lambda idx: hist[idx], reverse=True)
    
    # Select the top `num_max` local maxima
    local_maxima_indices = local_maxima_indices[:num_max]
    
    # Plot the histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor='black', align='edge')
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of the Image')
    x_ticks = np.arange(0.0, 1.05, tick_interval)  # Generate ticks from 0.0 to 1.0 with the given interval
    ax.set_xticks(x_ticks)  # Set the x-axis ticks
    
    # Mark the local maxima on the histogram
    ax.plot(bin_edges[local_maxima_indices], hist[local_maxima_indices], 'ro', label=f'Local Maxima')
    plt.legend()
    plt.show()

    return bin_edges[local_maxima_indices]

################################## Week 6 ########################################################

"""
Helping functions for snakes. 
Note that I have change the original functions, so the input is a N-by-2 array!
"""

def distribute_points(snake):
    """ Distributes snake points equidistantly. Expects snake to be N-by-2 array."""
    snake = snake.T
    N = snake.shape[1]
    
    # Compute length of line segments.
    d = np.sqrt(np.sum((np.roll(snake, -1, axis=1) - snake)**2, axis=0)) 
    f = scipy.interpolate.interp1d(np.hstack((0, np.cumsum(d))), 
                                   np.hstack((snake, snake[:,0:1])))
    return(f(sum(d) * np.arange(N) / N)).T

def is_crossing(p1, p2, p3, p4):
    """ Check if the line segments (p1, p2) and (p3, p4) cross."""
    crossing = False
    d21 = p2 - p1
    d43 = p4 - p3
    d31 = p3 - p1
    det = d21[0]*d43[1] - d21[1]*d43[0] # Determinant
    if det != 0.0 and d21[0] != 0.0 and d21[1] != 0.0:
        a = d43[0]/d21[0] - d43[1]/d21[1]
        b = d31[1]/d21[1] - d31[0]/d21[0]
        if a != 0.0:
            u = b/a
            if d21[0] > 0:
                t = (d43[0]*u + d31[0])/d21[0]
            else:
                t = (d43[1]*u + d31[1])/d21[1]
            crossing = 0 < u < 1 and 0 < t < 1         
    return crossing

def is_counterclockwise(snake):
    """ Check if points are ordered counterclockwise."""
    return np.dot(snake[0, 1:] - snake[0, :-1],
                  snake[1, 1:] + snake[1, :-1]) < 0

def remove_intersections(snake):
    """ Reorder snake points to remove self-intersections.
        Arguments: snake represented by a N-by-2 array.
        Returns: snake.
    """
    snake = snake.T
    pad_snake = np.append(snake, snake[:,0].reshape(2,1), axis=1)
    pad_n = pad_snake.shape[1]
    n = pad_n - 1 
    
    for i in range(pad_n - 3):
        for j in range(i + 2, pad_n - 1):
            pts = pad_snake[:, [i, i + 1, j, j + 1]]
            if is_crossing(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]):
                # Reverse vertices of smallest loop
                rb = i + 1 # Reverse begin
                re = j     # Reverse end
                if j - i > n // 2:
                    # Other loop is smallest
                    rb = j + 1
                    re = i + n                    
                while rb < re:
                    ia = rb % n
                    rb = rb + 1                    
                    ib = re % n
                    re = re - 1                    
                    pad_snake[:, [ia, ib]] = pad_snake[:, [ib, ia]]                    
                pad_snake[:,-1] = pad_snake[:,0]                
    snake = pad_snake[:, :-1]
    if is_counterclockwise(snake):
        return snake.T
    else:
        return np.flip(snake, axis=1).T
    
