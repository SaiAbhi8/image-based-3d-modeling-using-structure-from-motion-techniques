import numpy as np
from read_write_model import rotmat2qvec
data = np.load(r'C:\Users\saiab\Documents\MTP_outputs\moose_output_25s_25im\moose_output_25s_25im.npz')

print(data.files)

#%%
K = data['K']           # Intrinsic calibration matrix
Ps = data['P']         # 3D points
Ts = data['T']         # 4x4 camera transformations
pixels = data['pixels'] # Pixel coordinates
mm = data['mm']         # Measurements
unmatched = data['unmatched'] # Unmatched landmarks

data.close()
#%%
# Exporting "cameras.txt" file
import cv2

fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]
im = cv2.imread(r"C:\Users\saiab\Documents\MTP Datasets\moose\20221128_203954_43078365.jpg")
# Set your image dimensions
scale = 25
WIDTH = int(im.shape[1] * scale / 100)   # Replace with your width
HEIGHT = int(im.shape[0] * scale / 100)  # Replace with your height
with open('cameras.txt', 'w') as f:
    f.write("# Camera list with one line of data per camera:\n")
    f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f.write("# Number of cameras: 1\n")
    f.write(f"1 PINHOLE {WIDTH} {HEIGHT} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}\n")
    print("Exported Successfully")
# %%
# Exporting "images.txt" file.

import os

directory = r'C:\Users\saiab\Documents\sfm-master\data\inputs'
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
print(files)

with open('images.txt', 'w') as f:
    # Write header
    f.write("# Image list with two lines of data per image:\n")
    f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    f.write(f"# Number of images: {len(files)}, mean observations per image: to be calculated\n")
    
    
    for image_id in range(1, len(files)+1):
        T_image=Ts[image_id-1]
        R = T_image[:3, :3]
        t = T_image[:3, 3]
        qvec = rotmat2qvec(R)
        
        image_name = files[image_id-1]
        f.write(f"{image_id} {qvec[0]:.8f} {qvec[1]:.8f} {qvec[2]:.8f} {qvec[3]:.8f} "
            f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} 1 {image_name}\n")
        
        points2d = []
        
        for m in mm:
            if m[1]==image_id-1:
                x, y = m[2:4]
                point3d_id = m[0]
                points2d.append(f"{x:.8f} {y:.8f} {point3d_id}")
                # Write second line
        for un in unmatched:
            if(un[1]==image_id-1):
                x, y = un[2:4]
                point3d_id = -1
                points2d.append(f"{x:.8f} {y:.8f} {point3d_id}")
        f.write(" ".join(points2d) + "\n")
    print("Exported Successfully")
#%%
# Set a default reprojection error (update if you have actual values)
default_error = 1.0

# Open file for writing
with open('points3D.txt', 'w') as f:
    # Write header
    f.write("# 3D point list with one line of data per point:\n")
    f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    
    # Calculate mean track length for header
    total_tracks = 0
    valid_points = 0
    
    # First pass to calculate statistics
    for point_id in range(len(Ps)):
        # Find all measurements for this point
        point_measurements = mm[mm[:, 0] == point_id]
        if len(point_measurements) > 0:
            total_tracks += len(point_measurements)
            valid_points += 1
    
    # Complete header with statistics
    mean_track = total_tracks / valid_points if valid_points > 0 else 0
    f.write(f"# Number of points: {valid_points}, mean track length: {mean_track:.4f}\n")
    # Second pass to write points
    for point_id in range(len(Ps)):
        X, Y, Z = Ps[point_id]
        R, G, B = pixels[point_id]
        
        # Find all rows in mm where mm[:, 0] == point_id
        rows = np.where(mm[:, 0] == point_id)[0]  # row indices
        
        if len(rows) == 0:
            continue  # skip points with no observations
        
        line = f"{point_id} {X:.6f} {Y:.6f} {Z:.6f} {R} {G} {B} {default_error:.6f}"
        
        for row_idx in rows:
            image_id = mm[row_idx, 1]
            point2d_idx = row_idx  # POINT2D_IDX is the row number in mm
            line += f" {image_id} {point2d_idx}"
        
        f.write(line + "\n")


    
    
    



