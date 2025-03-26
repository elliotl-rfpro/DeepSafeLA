"""Main script for comparing the simulated and measured data."""
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageTk
import tkinter as tk
import seaborn as sns
from tqdm import tqdm

sns.set(palette="dark", font_scale=1.1, color_codes=True)
sns.set_style('darkgrid', {'axes.linewidth': 1, 'axes.edgecolor': 'black'})

matplotlib.use('TkAgg')


def extract_frames(video_path, output_folder, max_frames=None, fps=30):
    """Extract the frames from a supplied .mp4 video at a specified fps to a given output folder"""
    # Create folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check if frames have already been written
    existing_frames = [f for f in os.listdir(output_folder) if f.endswith('.png')]
    if max_frames is not None and len(existing_frames) >= max_frames:
        print('Frames already extracted, skipping extraction.')
        return

    # Open video file and process it
    video = cv2.VideoCapture(video_path)
    video_fps = int(video.get(cv2.CAP_PROP_FPS))
    if fps > video_fps:
        print(f'Requested FPS higher than video FPS, using video FPS of {video_fps} instead.')
        fps = video_fps

    frame_interval = max(1, int(video_fps / fps))
    frame_count = 0
    extracted_count = 0
    while True:
        # Read next frame
        ret, frame = video.read()

        if not ret or (max_frames and extracted_count >= max_frames):
            break

        # Save the frame if interval condition is satisfied
        if frame_count % frame_interval == 0:
            frame_name = os.path.join(output_folder, f'frame_{extracted_count:04d}.png')
            cv2.imwrite(frame_name, frame)
            extracted_count += 1
        frame_count += 1

    # Release video
    video.release()
    print(f'{extracted_count} frames extracted to {output_folder}')


def get_frame_count(video_path):
    """Count the number of frames in a video"""
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    return total_frames


def get_frames(folder):
    """Get the unwrapped frame sequence within a given folder"""
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')])


def compute_video_ssim(vid1_path, vid2_path, fps=30):
    """Compute the SSIM between two videos, frame by frame"""
    # Determine number of frames in each video
    vid1_frames = get_frame_count(vid1_path)
    vid2_frames = get_frame_count(vid2_path)

    # Limit frames to smallest between the two videos, in case videos are different lengths
    max_frames = min(vid1_frames, vid2_frames)
    print(f'Comparing the first {max_frames} frames...')

    # Extract frames
    extract_frames(vid1_path, 'vid1_frames', max_frames=max_frames, fps=28)
    extract_frames(vid2_path, 'vid2_frames', max_frames=max_frames, fps=28)
    frames_1 = sorted([os.path.join('vid1_frames', f) for f in os.listdir('vid1_frames') if f.endswith('.png')])
    frames_2 = sorted([os.path.join('vid2_frames', f) for f in os.listdir('vid2_frames') if f.endswith('.png')])

    # Compute SSIM for each frame pair
    ssim_values = []
    for f1, f2 in tqdm(zip(frames_1[:max_frames], frames_2[:max_frames]), desc='Computing SSIM'):
        img1 = cv2.imread(f1)
        img2 = cv2.imread(f2)
        img1_g = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_g = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(img1_g, img2_g, full=True)
        ssim_values.append(score)

    return ssim_values


def visualise_data(ssim_values):
    """Visualise the SSIM/colour histogram/image data"""
    frame_numbers = range(1, len(ssim_values) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(frame_numbers, ssim_values, '.')
    plt.xlabel('Frame number')
    plt.ylabel('SSIM value')
    plt.title('Temporal SSIM assessment')
    plt.show()


if __name__ == '__main__':
    # # Extract the .mp4 to file sequences
    sim_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Sim_Video'
    meas_path = 'C:/Users/ElliotLondon/Documents/PythonLocal/DeepSafeLA/data/Meas_Video'

    # Perform frame-by-frame SSIM assessment
    ssim_values = compute_video_ssim(os.path.join(sim_path, 'La_red14.mp4'),
                                     os.path.join(meas_path, 'GX012461p1_torrance_and_denker_to_western_480.mp4'),
                                     fps=28)

    print('All done!')
