import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def preprocess_and_extract_points(image_path, num_points=141, visualize=False):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Invert the image (assuming a white background and black waveform)
    inverted_image = cv2.bitwise_not(image)

    # Apply a Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours from the edges
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("No contours found in the image.")

    # Visualize edges and contours for debugging
    if visualize:
        cv2.imshow('Edges', edges)
        debug_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 1)
        cv2.imshow('Contours', debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Assuming the largest contour is the ECG waveform
    largest_contour = max(contours, key=cv2.contourArea)

    # Extract x, y coordinates of the contour
    points = [(point[0][0], point[0][1]) for point in largest_contour]

    # Sort points by x-coordinate
    points = sorted(points, key=lambda x: x[0])

    # Resample points to get exactly num_points
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
    x_new = np.linspace(x.min(), x.max(), num_points)
    y_new = f(x_new)
    points = list(zip(x_new.astype(int), y_new.astype(int)))

    return points[:num_points]

def map_coordinates_to_values(points, image_shape, amplitude_scale):
    height, _ = image_shape
    amplitude_points = []
    for _, y in points:
        # Map y to amplitude (assuming the y-axis represents amplitude)
        amplitude = (height - y) * amplitude_scale / height
        amplitude_points.append(amplitude)
    return amplitude_points

def extract_ecg_to_csv(image_path, output_csv_path, num_points=141, amplitude_scale=1.0, visualize=False):
    points = preprocess_and_extract_points(image_path, num_points, visualize)
    amplitude_points = map_coordinates_to_values(points, cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).shape, amplitude_scale)

    # Ensure the first value is "0"
    amplitude_points[0] = 0

    # Create a DataFrame with only the "Amplitude" column
    df = pd.DataFrame([amplitude_points], columns=[i for i in range(1, num_points + 1)])    
    # Save DataFrame to CSV without including the index or header
    df.to_csv(output_csv_path, header=False, index=False)
    print(f"ECG amplitude data successfully written to {output_csv_path}")

# Example usage
extract_ecg_to_csv(r'C:\\dinesh\\dinesh\\ed_final\\Abnormal3.jpg', r'C:\\dinesh\\dinesh\\ed_final\\data.csv', num_points=141, visualize=True)
