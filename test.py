# import cv2
# import torch
# import numpy as np

# from demo_superpoint import SuperPointFrontend

# sp = SuperPointFrontend(weights_path=r'C:\Users\saiab\SuperPointPretrainedNetwork\superpoint_v1.pth', nms_dist=4, conf_thresh=0.005, nn_thresh=0.1)

# #%%
# # Load grayscale image
# image = cv2.imread(r'C:\Users\saiab\Documents\sfm-master\data\inputs\IMG20250421123951.jpg', cv2.IMREAD_GRAYSCALE)
# image=image.astype(np.float32)
# output = sp.run(image)
# print(type(output), len(output))
# #%%
# for i, out in enumerate(output):
#     print(f"Output {i}: {type(out)}, shape: {getattr(out, 'shape', 'not array')}")
import torch
from PIL import Image
import sys
sys.path.append(r'C:\Users\saiab\d2-net')
import numpy as np
from lib.model_test import D2Net
from lib.pyramid import process_multiscale  # or process_multiscale
import cv2
# Display using matplotlib (since OpenCV uses BGR)
import matplotlib.pyplot as plt
#%%
# Load the pretrained model
model = D2Net(
    model_file=r'C:\Users\saiab\d2-net\models\d2_ots.pth',
    use_relu=True,
    use_cuda=torch.cuda.is_available()
)
model.eval()
#%%
def _extract_features_d2_net(image_path,scaling_factor):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    
    new_width = int(image.width * scaling_factor)
    new_height = int(image.height * scaling_factor)
    scaled_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # Convert scaled image to numpy array and normalize
    image_np = np.array(scaled_image).astype('float32') / 255.0
    #%
    # Convert numpy image to PyTorch tensor with shape [1, 3, H, W]
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    # Now image_tensor is ready for D2-Net
    print(image_tensor.shape)  # Should be [1, 3, H, W]
    print(type(image_tensor))
    
    # Extract features (multi-scale)
    with torch.no_grad():
        keypoints, scores, descriptors = process_multiscale(image_tensor, model)
    
    # These are already numpy arrays!
    print(type(keypoints), type(scores), type(descriptors))
    print('Keypoints:', keypoints.shape)
    print('Scores:', scores.shape)
    print('Descriptors:', descriptors.shape) 
    
    cv2_keypoints = [
        cv2.KeyPoint(float(x), float(y), float(scale))
        for x, y, scale in keypoints
    ]
    return cv2_keypoints, scores, descriptors, image_np
#%%
def _plot_keypoints(keypoints, image_path):
    # Convert D2-Net keypoints to OpenCV KeyPoint objects
    cv2_keypoints = [
        cv2.KeyPoint(float(x), float(y), float(scale))
        for x, y, scale in keypoints
    ]
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # Draw keypoints on the image
    output_image = cv2.drawKeypoints(
        image_bgr, cv2_keypoints, None,
        color=(0, 255, 0),  # Green keypoints
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title('D2-Net Keypoints')
    plt.axis('off')
    plt.show()
    
#%%
image_1_path = r"C:\Users\saiab\Documents\sfm-master\data\inputs\IMG20250421123951.jpg"
image_2_path = r"C:\Users\saiab\Documents\sfm-master\data\inputs\IMG20250421124006.jpg"

kp_1, scores_1, des_1, img1_np = _extract_features_d2_net(image_1_path,0.25) 
kp_2, scores_2, des_2, img2_np = _extract_features_d2_net(image_2_path,0.25) 
#%%
K = np.load(r"C:\Users\saiab\Documents\sfm-master\calibration_data\sav_oppo_cam_calib.npy")
print(K)
K*=0.25
print(K)
#%%
# Descriptor normalization
des_1 = des_1 / np.linalg.norm(des_1, axis=1, keepdims=True)
des_2 = des_2 / np.linalg.norm(des_2, axis=1, keepdims=True)

# BFMatcher with ratio test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des_1, des_2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

kp1_match = np.array([kp_1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
kp2_match = np.array([kp_2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

E, inlier = cv2.findEssentialMat(kp1_match, kp2_match, K, method=cv2.RANSAC, threshold=2, prob=0.999)
print('Number of inlier matches:', np.sum(inlier))
#%%
def extract_sift_features(image_path, scaling_factor=1.0):
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
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors, gray
#%%
# Extract features from both images
kp1, des1, img1 = extract_sift_features(r"C:\Users\saiab\Documents\sfm-master\data\inputs\IMG20250421123951.jpg", scaling_factor=0.25)
kp2, des2, img2 = extract_sift_features(r"C:\Users\saiab\Documents\sfm-master\data\inputs\IMG20250421124006.jpg", scaling_factor=0.25)

# Match descriptors using BFMatcher
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test for robust matching
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Get matched keypoint coordinates
pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches], dtype=np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches], dtype=np.float32)

# Estimate essential matrix and inliers
E, inlier_mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999)
num_inliers = np.sum(inlier_mask)
print(f"Number of inlier matches: {num_inliers}")

