"""Script for loading dRisk images from the LA drive data set"""

import numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
from scipy.optimize import minimize
from numpy.typing import NDArray
from typing import List
from skimage import color

# This sets OpenCV to import using the .EXR opener. This only works for .EXR files. If you wish to use a different
# file type, such as .png or .jpg, please comment out the following line.
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

# Global variables
"""Macbeth chart, dimly shadowed (partial): 20250204_071453_index0000_037000.png"""
regions_mcb_partial = [
    [860, 365, 885, 390], [898, 365, 923, 390], [937, 365, 962, 390], [976, 361, 1001, 386], [1014, 360, 1039, 385], [1052, 360, 1077, 385],
    [862, 405, 887, 430], [899, 402, 924, 427], [939, 402, 964, 427], [976, 401, 1001, 426], [1015, 399, 1040, 424], [1052, 398, 1077, 423],
    [862, 442, 887, 467], [902, 441, 927, 466], [941, 438, 966, 463], [977, 439, 1002, 464], [1017, 437, 1042, 462], [1053, 435, 1078, 460],
    [864, 479, 889, 504], [902, 477, 927, 502], [941, 475, 966, 500], [978, 474, 1003, 499], [1017, 473, 1042, 498], [1055, 473, 1080, 498],
]
"""Macbeth chart, reference simulation: macbeth_gm.png"""
regions_mcb_simulated = [
    [100, 150, 200, 250], [400, 150, 500, 250], [650, 150, 750, 250], [925, 150, 1025, 250], [1200, 150, 1300, 250], [1450, 150, 1550, 250],
    [100, 400, 200, 500], [400, 400, 500, 500], [650, 400, 750, 500], [925, 400, 1025, 500], [1200, 400, 1300, 500], [1450, 400, 1550, 500],
    [100, 675, 200, 775], [400, 675, 500, 775], [650, 675, 750, 775], [925, 675, 1025, 775], [1200, 675, 1300, 775], [1450, 675, 1550, 775],
    [100, 950, 200, 1050], [400, 950, 500, 1050], [650, 950, 750, 1050], [925, 950, 1025, 1050], [1200, 950, 1300, 1050], [1450, 950, 1550, 1050],
]
"""Regions for comparison within 20250204_071453_index0000_054155.pgm. Sim then meas."""
region_bonnet = [[1483, 906, 1583, 1006], [1250, 900, 1350, 1000]]
region_road = [[1158, 464, 1258, 564], [1200, 450, 1300, 550]]
region_shaded = [[1500, 625, 1600, 725], [1500, 625, 1600, 725]]
region_sky = [[1300, 100, 1400, 200], [1400, 100, 1500, 200]]
region_house = [[1030, 728, 1100, 798], [1050, 240, 1125, 315]]
region_tlight = [[856, 140, 864, 149], [859, 120, 868, 129]]
region_names = ['ego bonnet', 'illuminated road', 'shaded road', 'sky', 'house', 'traffic light green']


def ordering_constraint(m_flat):
    """Constraint that enforces the diagonals of M to be the largest."""
    m = m_flat.reshape(3, 3)
    return np.array([m[0, 0] - m[0, 1], m[0, 1] - m[0, 2],
                     m[1, 1] - m[1, 0], m[1, 1] - m[1, 2],
                     m[2, 1] - m[2, 0], m[2, 2] - m[2, 1]]
                    )


def magnitude_constraint(m_flat):
    """Constraint that enforces diagonals to be x larger than the off-diagonals"""
    m = m_flat.reshape(3, 3)
    x = 0.3
    return np.array([m[0, 0] - (m[0, 1] + x), m[0, 0] - (m[0, 2] + x),
                     m[1, 1] - (m[1, 0] + x), m[1, 1] - (m[1, 2] + x),
                     m[2, 2] - (m[2, 1] + x), m[2, 2] - (m[2, 1] + x)]
                    )


def process_macbeth_colours(chart: NDArray, rows: int, columns: int, regions: List, ao: str, avg: bool = False):
    """
    Load and analyze a reference Macbeth chart.

    Parameters:
        chart (NDArray): The loaded chart, in pixel coordinates.
        rows (int): Number of rows within the Macbeth chart image.
        columns (int): Number of columns within the Macbeth chart image.
        regions (List): The regions of the colours within the Macbeth chart image. Manually defined as [x1, y1, x2, y2].
        ao (str): Flag for 'array orientation' that enables a switch case between [x1, y1] and [y1, x1] formats.
        avg (bool): Bool whether to return averaged (True) or un-averaged (False) colour data.

    Returns:
        list: A list containing either the average luminance value (if .RAW. input array), or the average of each
        RGB channel (if 3-channel input array).
    """

    # Initialize a list to store the average color values
    colour_values = []

    # Loop through each color square to calculate average values
    j = 0
    for row in range(rows):
        for column in range(columns):
            # Crop the current square from the image using the calculated coordinates
            region = regions[j]
            if ao == 'xy':
                current_colour = chart[region[0]:region[2], region[1]:region[3]]
            elif ao == 'yx':
                current_colour = chart[region[1]:region[3], region[0]:region[2]]
            else:
                raise ValueError("ao param requires values of 'xy' or 'yx'.")

            if avg:
                if len(chart.shape) == 3 and chart.shape[2] == 3:
                    # If RGB, calculate average colour for each channel
                    avg_colour = np.mean(current_colour, axis=(0, 1))
                elif len(chart.shape) == 2:
                    # If single channel .raw, throw an exception:
                    raise NotImplementedError('You have passed flag avg=True for a 2d .RAW image. Averaging over the '
                                              'pixel values without demosaicing or subpixel processing would create '
                                              'an unphysical array of values. Script exiting...')
                else:
                    raise ValueError(f'arg[0] has incorrect dimensions: must be np.NDArray with shape [x, y, 3] or '
                                     f'[x, y]. Script exiting...')
                colour_values.append(avg_colour)
            else:
                colour_values.append(current_colour)
            j += 1

    return np.array(colour_values)


def display_colours(image, title: str, orientation: str = 'landscape'):
    """
    Visualize different Macbeth charts.

    Parameters:
        image (array-like): The input image, which can be a NumPy array or an image file.
        title (str): The title of the plot.
        orientation (str): The orientation of the plot, either 'landscape' or 'portrait'.
    Raises:
        ValueError: If the orientation is not 'landscape' or 'portrait'.
    """

    # Ensure the image is a NumPy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Set the layout based on the orientation
    if orientation == 'landscape':
        rows = 4
        columns = 6
        figsize = (10, 6)
    elif orientation == 'portrait':
        rows = 6
        columns = 4
        figsize = (6, 10)
    else:
        raise ValueError("Arg: 'orientation' requires value 'landscape' or 'portrait'.")

    # Create the plot
    fig, axes = plt.subplots(rows, columns, figsize=figsize)
    fig.suptitle(title, color='white')

    # Iterate over each subplot and display the corresponding color
    for j, ax in enumerate(axes.flat):
        colour = image[j]  # Normalize to [0, 1] for display
        ax.imshow(colour)
        ax.axis('off')  # Hide the axis

    # Set the background color of the figure
    fig.patch.set_facecolor('black')


def emulate_cfa(image: NDArray, rgb_weights, gain, offset) -> NDArray:
    """
    Apply colour filters to each channel of an image depending upon the supplied weightings.

    Parameters:
        image (numpy.ndarray): The input image
        rgb_weights (list): List of RGB weightings including global scaling factors.

    Returns:
        image (numpy.ndarray): the filtered image.
    """
    # Create a copy of the image to avoid modifying the original
    filtered_image = image.copy().astype(np.float64)
    min_orig = np.min(filtered_image)
    max_orig = np.max(filtered_image)

    # Apply the colour filters with global scaling factors
    filtered_image[:, :, 0] = (image[:, :, 0] * rgb_weights[0] +
                               image[:, :, 1] * rgb_weights[1] +
                               image[:, :, 2] * rgb_weights[2]
                               )
    filtered_image[:, :, 0] *= gain[0]
    filtered_image[:, :, 0] += offset[0]

    filtered_image[:, :, 1] = (image[:, :, 0] * rgb_weights[3] +
                               image[:, :, 1] * rgb_weights[4] +
                               image[:, :, 2] * rgb_weights[5]
                               )
    filtered_image[:, :, 1] *= gain[1]
    filtered_image[:, :, 1] += offset[0]

    filtered_image[:, :, 2] = (image[:, :, 0] * rgb_weights[6] +
                               image[:, :, 1] * rgb_weights[7] +
                               image[:, :, 2] * rgb_weights[8]
                               )
    filtered_image[:, :, 2] *= gain[2]
    filtered_image[:, :, 2] += offset[2]

    # Ensure values are within the valid range for uint8 format
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image


def decompress_12_to_24_bits(image, input_map, output_map):
    """Decompress values from the sensor to their original values,
    using the input_map and output_map"""

    image = image.astype(np.uint32)
    new_image = np.empty(image.shape, dtype=np.uint32)
    for i in range(1, len(input_map)):
        in_ = input_map[i - 1 : i + 1]
        out = output_map[i - 1 : i + 1]

        adjust = (lambda x: ((x - in_[0]) * (out[1] - out[0])) // (in_[1] - in_[0]) + out[0])
        mask = np.logical_and((in_[0] < image), (image <= in_[1]))
        new_image = np.where(mask, adjust(image), new_image)

    return new_image


def extract_bayer_channels(raw_image):
    """Extract Red, Green, and Blue channels from GBRG Bayer pattern"""
    # ignore the last row or column pixel if there is an odd length or height

    end_0 = raw_image.shape[0] & ~1
    end_1 = raw_image.shape[1] & ~1

    red_channel = ((raw_image[1:end_0:2, :end_1:2]) * 5) >> 2
    green_channel = (raw_image[:end_0:2, :end_1:2] + raw_image[1:end_0:2, 1:end_1:2]) // 2
    blue_channel = (raw_image[:end_0:2, 1:end_1:2])

    return red_channel, green_channel, blue_channel


def left_shift_saturate(arr, shift):
    """shift the array to the left by arr bits, but saturate if would overflow"""
    # Define the maximum value based on the dtype of the array
    max_val = np.iinfo(arr.dtype).max

    # Calculate the maximum value that can be safely shifted without overflow
    safe_max = max_val >> shift

    # Perform the left shift with saturation
    saturated = np.where(arr > safe_max, max_val, np.left_shift(arr, shift))

    return saturated


def grayworld_assumption(image):
    lab = color.rgb2lab(image)

    # we'll offset the average of each
    a_shift, b_shift = (np.average(lab[:, :, 1]), np.average(lab[:, :, 2]))

    # scale the chroma distance shifted according to amount of
    # luminance. The 1.1 overshoot is because we cannot be sure
    # to have gotten the data in the first place.
    a_delta = a_shift * (lab[:, :, 0]/100) * 1.1
    b_delta = b_shift * (lab[:, :, 0]/100) * 1.1
    lab[:, :, 1] = lab[:, :, 1]-a_delta
    lab[:, :, 2] = lab[:, :, 2]-b_delta

    return color.lab2rgb(lab)


def white_balance(image):
    """Apply white balancing using the grayscale assumption"""
    return grayworld_assumption(image)


def custom_white_balance(image):
    """Apply white balance using info about the average 'white' within the image"""

    avg_white = np.mean(image[339:349, 1166:1176], axis=(0, 1))
    scaling_factor = 255 / avg_white

    corr_image = image * scaling_factor
    corr_image = np.clip(corr_image, 0, 255).astype(np.uint8)

    return corr_image


def gamma_correction(raw_image, gamma):
    """Apply gamma correction and correct the bit size"""
    if gamma == 1:
        return (raw_image*(float(0xff)/float(0xffff))).astype(np.uint8)

    # Normalize values
    normalized_values = raw_image / float(0xffff)

    # Apply gamma correction
    corrected_values = np.power(normalized_values, 1.0 / gamma)

    return corrected_values


def analyse_pgm(filepath):
    """Analyses a pgm file according to passed arg."""

    image_data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    image_data.byteswap(inplace=True)
    image = decompress_12_to_24_bits(image_data,
                                     [0, 938, 1863, 2396, 3251, 4095],
                                     [0, 1566, 105740, 387380, 3818601, 16777215]
                                     )

    # # Extract Bayer channels
    red_channel, green_channel, blue_channel = extract_bayer_channels(image)
    image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    # Greyworld automatic white balancing
    image_proc = left_shift_saturate(image, 8)

    # Apply gain below saturation
    # image_proc = image * 16
    # image_proc = image_proc * (((2 ** 24) - 16) / np.max(image_proc))

    image_awb = white_balance(image_proc)
    image_awb = gamma_correction(image_awb, 2.0)

    # Perform image-informed white balancing
    image_iwb = custom_white_balance(image_proc)
    image_iwb = gamma_correction(image_iwb, 2.0)

    # Cvt to rgb for viewing
    image_awb_rgb = ((image_awb.astype(np.float64) / np.max(image_awb.astype(np.float64))) * 255.0).astype(np.uint8)
    image_iwb_rgb = ((image_iwb.astype(np.float64) / np.max(image_iwb.astype(np.float64))) * 255.0).astype(np.uint8)

    plt.figure()
    plt.imshow(image_awb_rgb)
    plt.title('Auto white balance (greyworld)')
    plt.grid(False)

    plt.figure()
    plt.imshow(image_iwb_rgb)
    plt.title('Informed white balance (white ref area)')
    plt.grid(False)
    plt.show()

    print('.pgm conversion complete.\n')
    return image, image_iwb_rgb


if __name__ == '__main__':
    # Fpath to .pgm file
    pgm_fpath = "C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Recorded_Images/Torrence1/20250204_071453_index0000_054155.pgm"

    # Analyse .pgm file
    pgm_image, pgm_image_rgb = analyse_pgm(pgm_fpath)
    
    # Load reference image and compare regions
    exr_fpath = 'C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Sim_Images/rt_xroads_lights_vehs.exr'
    exr_fpath_rgb = 'C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Sim_Images/rt_xroads_lights_vehs.bmp'
    exr_image = cv2.imread(exr_fpath, cv2.IMREAD_UNCHANGED)
    exr_image_rgb = cv2.imread(exr_fpath_rgb, cv2.IMREAD_UNCHANGED)

    # Make sure both are in float54
    pgm_image = pgm_image.astype(np.float64)
    exr_image = exr_image.astype(np.float64)

    # Loop over regions, crop to area, save as a result
    results_sim = []
    results_meas = []
    results_sim_rgb = []
    results_meas_rgb = []
    i = 0
    for region in [region_bonnet, region_road, region_shaded, region_sky, region_house, region_tlight]:
        results_sim.append(exr_image[region[0][1]:region[0][3], region[0][0]:region[0][2]])
        results_meas.append(pgm_image[region[1][1]:region[1][3], region[1][0]:region[1][2]])
        results_sim_rgb.append(exr_image_rgb[region[0][1]:region[0][3], region[0][0]:region[0][2]])
        results_meas_rgb.append(pgm_image_rgb[region[1][1]:region[1][3], region[1][0]:region[1][2]])

        # Print out region statistics
        print(f"Simulated region '{region_names[i]}': max: {np.max(results_sim[-1]):.2f}, min: {np.min(results_sim[-1]):.2f}, mean: {np.mean(results_sim[-1]):.2f}")
        print(f"Measured region '{region_names[i]}': max: {np.max(results_meas[-1]):.2f}, min: {np.min(results_meas[-1]):.2f}, mean: {np.mean(results_meas[-1]):.2f}")
        i += 1
    print('')

    # Calculate and display luminance and rgb histograms for entire image, and regions

    # Fpaths to calibration images
    mcb_partial = "C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Calibration_Images/20250204_071453_index0000_037000.png"
    mcb_shaded = "C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Calibration_Images/20250204_071453_index0000_023930.png"
    mcb_overex = "C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Calibration_Images/20250204_071453_index0000_065880.png"

    # Fpath to simulated 8bit macbeth chart
    mcb_ref = "C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/macbeth_gm.png"

    # Load the given sim/real data and get rgb colour regions
    mcb_ref_data = cv2.imread(mcb_ref)
    mcb_ref_data = cv2.cvtColor(mcb_ref_data, cv2.COLOR_BGR2RGB)
    mcb_ref_data_raw = process_macbeth_colours(mcb_ref_data, rows=4, columns=6, regions=regions_mcb_simulated, ao='yx', avg=False)
    mcb_ref_data_rgb = process_macbeth_colours(mcb_ref_data, rows=4, columns=6, regions=regions_mcb_simulated, ao='yx', avg=True)
    mcb_partial_data = cv2.imread(mcb_partial)
    mcb_partial_data = cv2.cvtColor(mcb_partial_data, cv2.COLOR_BGR2RGB)
    mcb_partial_data_raw = process_macbeth_colours(mcb_partial_data, rows=4, columns=6, regions=regions_mcb_partial, ao='yx', avg=False)
    mcb_partial_data_rgb = process_macbeth_colours(mcb_partial_data, rows=4, columns=6, regions=regions_mcb_partial, ao='yx', avg=True)

    print('Calculating optimal CFA values')
    # Bounds and constraints
    bounds = [(0, 1)] * 9
    constraints = [
        {'type': 'ineq', 'fun': ordering_constraint},
        {'type': 'ineq', 'fun': magnitude_constraint},
        {'type': 'ineq', 'fun': lambda m: np.min(m) - 0.1},
        {'type': 'ineq', 'fun': lambda m: 0.8 - np.max(m)}
    ]
    M_init = np.eye(3).flatten()

    def loss_fun(m_flat):
        """Optimisation function for scipy.minimize: find the optimal M (linalg matrix) config."""
        m = m_flat.reshape(3, 3)
        error = np.linalg.norm(mcb_ref_data_rgb @ m - mcb_partial_data_rgb)
        return error

    # Optimize
    res = minimize(loss_fun, M_init, bounds=bounds, method='SLSQP', options={'maxiter': 10000})
    M_opt = res.x.reshape(3, 3)

    # Calculate offset and gain
    print('Calculating offset/gain correction')
    black_idx = -1
    white_idx = -6
    black_corr = mcb_ref_data_rgb[black_idx] @ M_opt
    black_meas = mcb_partial_data_rgb[black_idx]
    offset = black_meas - black_corr
    white_corr = mcb_ref_data_rgb[white_idx] @ M_opt
    white_meas = mcb_partial_data_rgb[white_idx]
    gain = (white_meas - offset) / white_corr

    # Apply total transformation, optionally apply offset/gain
    mcb_ref_data_corr = (mcb_ref_data_rgb @ M_opt)
    mcb_ref_data_corr *= gain
    mcb_ref_data_corr += offset
    # Re-expand data
    mcb_ref_data_arr = np.zeros_like(mcb_ref_data_raw)
    for i in range(24):
        mcb_ref_data_arr[i] = np.tile(mcb_ref_data_corr[i], (100, 100, 1))

    # Print solution, gain and offset
    M_corr = np.zeros_like(M_opt)
    M_corr[0, 0] = M_opt[0, 0] * gain[0]
    M_corr[0, 1] = M_opt[0, 1] * gain[1]
    M_corr[0, 2] = M_opt[0, 2] * gain[2]
    M_corr[1, 0] = M_opt[1, 0] * gain[0]
    M_corr[1, 1] = M_opt[1, 1] * gain[1]
    M_corr[1, 2] = M_opt[1, 2] * gain[2]
    M_corr[2, 0] = M_opt[2, 0] * gain[0]
    M_corr[2, 1] = M_opt[2, 1] * gain[1]
    M_corr[2, 2] = M_opt[2, 2] * gain[2]
    with np.printoptions(precision=5, suppress=True):
        print(f'M: {M_opt}')
        print(f'Gain: {gain}')
        print(f'Offset: {offset}')
        print(f'M_corr: {M_corr}')

    display_colours(mcb_ref_data_raw, 'Simulated macbeth')
    display_colours(mcb_partial_data_raw, 'Measured macbeth')
    display_colours(mcb_ref_data_arr, 'Simulated macbeth (corrected')
    plt.show()

    # Apply to a sample image
    sample_image = cv2.imread('C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Sim_Images/raytrace_torrence1.png')
    sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
    sample_image_corr = emulate_cfa(sample_image, M_opt.flatten(), gain, offset)

    # Show images
    plt.imshow(sample_image_corr)
    plt.grid(False)
    plt.show()

    print('Script executed successfully.')
