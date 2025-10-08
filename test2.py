import torch
from PIL import Image
import sys
sys.path.append(r'C:\Users\saiab\d2-net')
import numpy as np
from lib.model_test import D2Net
from lib.pyramid import process_multiscale
import cv2
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#%%
# Load the pretrained D2-Net model
model = D2Net(
    model_file=r'C:\Users\saiab\d2-net\models\d2_ots.pth',
    use_relu=True,
    use_cuda=torch.cuda.is_available()
)
model.eval()

def _extract_features_d2_net(image_path, scaling_factor):
    # Load and scale the image
    image = Image.open(image_path).convert('RGB')
    new_width = int(image.width * scaling_factor)
    new_height = int(image.height * scaling_factor)
    scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
    image_np = np.array(scaled_image).astype('float32') / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        keypoints, scores, descriptors = process_multiscale(image_tensor, model)
    cv2_keypoints = [
        cv2.KeyPoint(float(x), float(y), float(scale))
        for x, y, scale in keypoints
    ]
    return cv2_keypoints, scores, descriptors, image_np

def draw_inlier_outlier_matches_side_by_side(img1, kp1, img2, kp2, inlier_matches, outlier_matches):
    # Draw outliers in red
    img_out = cv2.drawMatches(
        img1, kp1, img2, kp2, outlier_matches, None,
        matchColor=(0,0,255),  # Red
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    # Draw inliers in green on top
    img_out = cv2.drawMatches(
        img1, kp1, img2, kp2, inlier_matches, img_out,
        matchColor=(0,255,0),  # Green
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return img_out

# Set your paths and scaling factor
image_1_path = r"C:\Users\saiab\Documents\MTP Datasets\mmm_hall\IMG20250421123951.jpg"
image_2_path = r"C:\Users\saiab\Documents\MTP Datasets\mmm_hall\IMG20250421124006.jpg"
scaling_factor = 0.25

# Extract features
kp_1, scores_1, des_1, img1_np = _extract_features_d2_net(image_1_path, scaling_factor)
kp_2, scores_2, des_2, img2_np = _extract_features_d2_net(image_2_path, scaling_factor)
K = np.load(r"C:\Users\saiab\Documents\sfm-master\calibration_data\sav_oppo_cam_calib.npy")
K *= scaling_factor
#%%
def strict_inlier_matches_plot(img1_path, img2_path, kp1, des1, kp2, des2, K, scaling_factor):
    # Load and scale images for visualization
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1_scaled = cv2.resize(img1, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
    img2_scaled = cv2.resize(img2, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)

    # Normalize descriptors (important for D2-Net)
    des1 = des1 / np.linalg.norm(des1, axis=1, keepdims=True)
    des2 = des2 / np.linalg.norm(des2, axis=1, keepdims=True)

    # Feature matching with Lowe's ratio test
    bf = cv2.BFMatcher()
    matches_knn = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    if len(good_matches) < 8:
        print("Not enough matches for essential matrix estimation.")
        return

    # Get matched keypoint coordinates
    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

    # Estimate essential matrix and get inlier mask
    E, inlier_mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    if inlier_mask is None:
        print("No inliers found.")
        return
    inlier_mask = inlier_mask.ravel().astype(bool)
    print(f"Number of strict geometric inliers: {np.sum(inlier_mask)}")

    # Only keep inlier matches
    inlier_matches = [m for m, inl in zip(good_matches, inlier_mask) if inl]

    # Draw only inlier matches (for strict validation)
    output_img = cv2.drawMatches(
        img1_scaled, kp1, img2_scaled, kp2, inlier_matches, None,
        matchColor=(0,255,0), singlePointColor=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.figure(figsize=(16,8))
    plt.imshow(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Strict Geometric Inliers (Green)")
    plt.show()

# Example usage:
# Assume you have already extracted kp1, des1, kp2, des2, and loaded K and scaling_factor
# kp1, des1, _, _ = _extract_features_d2_net(image_1_path, scaling_factor)
# kp2, des2, _, _ = _extract_features_d2_net(image_2_path, scaling_factor)
# K = np.load('your_intrinsics.npy') * scaling_factor

strict_inlier_matches_plot(
    image_1_path, image_2_path,
    kp_1, des_1, kp_2, des_2, K, scaling_factor
)
#%%
def extract_sift_features(image_path, scaling_factor):
    """
    Loads an image, scales it, and extracts SIFT keypoints and descriptors.
    
    Args:
        image_path (str): Path to the image file.
        scaling_factor (float): Factor to scale both width and height.
        
    Returns:
        keypoints (list of cv2.KeyPoint): Detected SIFT keypoints.
        descriptors (np.ndarray): SIFT descriptors (N x 128).
        scaled_image (np.ndarray): The scaled grayscale image.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Scale image
    if scaling_factor != 1.0:
        new_size = (int(gray.shape[1] * scaling_factor), int(gray.shape[0] * scaling_factor))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    # Initialize SIFT
    sift = cv2.SIFT_create(nfeatures=200000, contrastThreshold=0.001, edgeThreshold=100)
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors
#%%
kp_sift_1, des_sift_1 = extract_sift_features(image_1_path, scaling_factor)
kp_sift_2, des_sift_2 = extract_sift_features(image_2_path, scaling_factor)

strict_inlier_matches_plot(
    image_1_path, image_2_path,
    kp_sift_1, des_sift_1, kp_sift_2, des_sift_2, K, scaling_factor
)
