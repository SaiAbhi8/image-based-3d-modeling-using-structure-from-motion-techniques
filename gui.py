import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import shutil
import traceback
import cv2
import matplotlib.pyplot as plt
from run import run
import threading

progress = None  # Global reference to the progress bar
# ==============================
# Calibration Code and Utilities
# ==============================

# Import your calibration function from camera_calibration.py.
# It is assumed that camera_calibration.py defines:
#   def calibrate_camera(images_directory: str) -> np.ndarray:
def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    return cv2.resize(img, (width, height))

def calibrate_camera(in_files: str) -> np.ndarray:

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    CHECKERBOARD = (8, 6)
    SCALE_PERCENT = 25  # Reduce image size to 50% of original (adjust as needed)


    valid_imgs = [".jpg", ".png"]
    
    images = [os.path.join(in_files, f) for f in sorted(os.listdir(in_files)) if os.path.splitext(f)[1].lower() in valid_imgs]
    
    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    objpoints = []
    imgpoints = []
    
    print(f"Using {len(images)} images to calibrate the camera ")
    idx=1
    for filename in images:
        image_path = os.path.join(in_files, filename)
        img_orig = cv2.imread(image_path)
        
        if img_orig is None:
            print(f"Could not read image: {image_path}")
            continue
        img = resize_image(img_orig, SCALE_PERCENT)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            print(f"\n Success on image {idx}")
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            # Visual verification
            cv2.drawChessboardCorners(img, CHECKERBOARD, corners_refined, ret)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))    
            plt.title(f'Corners Detected: Image {idx}')
            plt.axis('off')
            plt.show(block=False)
        else:
            print(f"Failed on image {idx}: {filename}")
    
        idx+=1

    if len(objpoints) > 10:
        ret, K_scaled, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        print("\nScaled Camera Matrix (K):")
        print(K_scaled)
        
        # To get original-scale parameters (if needed)
        scale_factor = SCALE_PERCENT / 100
        K_original = K_scaled.copy()
        K_original[0,0] /= scale_factor  # fx
        K_original[1,1] /= scale_factor  # fy
        K_original[0,2] /= scale_factor  # cx
        K_original[1,2] /= scale_factor  # cy
        
        print("\nEstimated Original-scale Camera Intrinsic Parameters:")
        print(K_original)
    else:
        print("Insufficient detections for calibration")
    
    
    return K_original

# Directory where calibration matrices are stored.
CALIBRATION_DIR = "calibration_data"
if not os.path.exists(CALIBRATION_DIR):
    os.makedirs(CALIBRATION_DIR)

def save_calibration_matrix(camera_name: str, matrix: np.ndarray):
    """Saves the calibration matrix for the given camera."""
    file_path = os.path.join(CALIBRATION_DIR, f"{camera_name}.npy")
    np.save(file_path, matrix)

def load_calibration_matrix(camera_name: str) -> np.ndarray:
    """Loads the calibration matrix for the given camera."""
    file_path = os.path.join(CALIBRATION_DIR, f"{camera_name}.npy")
    if os.path.exists(file_path):
        return np.load(file_path)
    else:
        raise FileNotFoundError(f"No calibration data found for camera '{camera_name}'.")

def list_camera_names():
    """Lists the available cameras (by file name) from the calibration directory."""
    files = os.listdir(CALIBRATION_DIR)
    camera_names = [os.path.splitext(f)[0] for f in files if f.endswith('.npy')]
    return camera_names

def delete_calibration_matrix(camera_name: str):
    """Deletes the calibration data file for the given camera."""
    file_path = os.path.join(CALIBRATION_DIR, f"{camera_name}.npy")
    if os.path.exists(file_path):
        os.remove(file_path)
    else:
        raise FileNotFoundError(f"No calibration data found for camera '{camera_name}'.")

# ============================
# Stage 1: Calibration GUI
# ============================

class CalibrationGUI(tk.Tk):
    """First-stage GUI for camera calibration and management."""
    def __init__(self):
        super().__init__()
        self.title("Camera Calibration Stage")
        self.geometry("900x600")
        
        # Frame for listing camera calibrations.
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        ttk.Label(list_frame, text="Available Cameras:").pack(anchor=tk.W)
        self.camera_listbox = tk.Listbox(list_frame, height=10)
        self.camera_listbox.pack(fill=tk.BOTH, expand=True, side=tk.LEFT)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.camera_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.camera_listbox.config(yscrollcommand=scrollbar.set)
        
        # Control buttons.
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=10, pady=(0,10))
        
        ttk.Button(control_frame, text="Add New Camera", command=self.add_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="View Parameters", command=self.view_camera_parameters).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Delete Camera", command=self.delete_camera).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh", command=self.refresh_camera_list).pack(side=tk.LEFT, padx=5)
        
        # Continue button to advance to stage 2.
        ttk.Button(self, text="Continue", command=self.continue_to_next_stage).pack(pady=10)
        
        self.refresh_camera_list()
        
    def refresh_camera_list(self):
        """Refresh the list of saved cameras."""
        self.camera_listbox.delete(0, tk.END)
        for camera in list_camera_names():
            self.camera_listbox.insert(tk.END, camera)
    
    def add_camera(self):
        """Prompt the user to add a new camera calibration."""
        camera_name = simpledialog.askstring("Camera Name", "Enter the new camera's name:", parent=self)
        if not camera_name:
            return
        
        image_dir = filedialog.askdirectory(title="Select Calibration Images Folder", parent=self)
        if not image_dir:
            return
        
        try:
            calibration_matrix = calibrate_camera(image_dir)
            save_calibration_matrix(camera_name, calibration_matrix)
            messagebox.showinfo("Success", f"Calibration for '{camera_name}' computed and saved.")
            self.refresh_camera_list()
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred while calibrating the camera:\n{e}")
    
    def view_camera_parameters(self):
        """Display the calibration matrix for the selected camera."""
        selection = self.camera_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a camera to view its parameters.")
            return
        
        index = selection[0]
        camera_name = self.camera_listbox.get(index)
        try:
            calibration_matrix = load_calibration_matrix(camera_name)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load calibration for '{camera_name}':\n{e}")
            return
        
        # Open a new window displaying the matrix.
        matrix_window = tk.Toplevel(self)
        matrix_window.title(f"Calibration Matrix for '{camera_name}'")
        matrix_window.geometry("300x200")
        
        ttk.Label(matrix_window, text=f"Calibration Matrix (K) for {camera_name}:").pack(pady=5)
        matrix_str = np.array2string(calibration_matrix, precision=4, separator=', ')
        matrix_text = tk.Text(matrix_window, height=8, width=40, wrap=tk.NONE)
        matrix_text.insert(tk.END, matrix_str)
        matrix_text.config(state=tk.DISABLED)
        matrix_text.pack(padx=10, pady=10)
    
    def delete_camera(self):
        """Delete the calibration data for the selected camera."""
        selection = self.camera_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a camera to delete.")
            return
        
        index = selection[0]
        camera_name = self.camera_listbox.get(index)
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{camera_name}'?")
        if confirm:
            try:
                delete_calibration_matrix(camera_name)
                messagebox.showinfo("Deleted", f"Camera '{camera_name}' has been deleted.")
                self.refresh_camera_list()
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while deleting the camera:\n{e}")
    
    def continue_to_next_stage(self):
        """Pass the selected camera's calibration matrix to the second stage GUI and close this window."""
        selection = self.camera_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a camera to continue.")
            return
        
        index = selection[0]
        camera_name = self.camera_listbox.get(index)
        try:
            calibration_matrix = load_calibration_matrix(camera_name)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load calibration for '{camera_name}':\n{e}")
            return
        
        self.destroy()  # Close the calibration GUI.
        SecondStageGUI(camera_name, calibration_matrix)

# ==================================
# Stage 2: Second Stage GUI
# ==================================
# This code is based on your provided GUI code for further processing.
# It now accepts camera_name and calibration_matrix from stage 1 and
# sends them as input to the run function.

# Import the "run" function from your module.
# It is assumed that run is defined in a file named run.py.
try:
    from run import run
except ImportError:
    # Dummy run function for demonstration.
    def run(**args):
        print("Running with arguments:")
        for key, value in args.items():
            print(f"{key}: {value}")

DEFAULT_DIR = "data/inputs"

class SecondStageGUI(tk.Tk):
    def __init__(self, camera_name, calibration_matrix):
        super().__init__()
        self.camera_name = camera_name
        self.calibration_matrix = calibration_matrix
        self.title("Structure from Motion - GUI")
        self.geometry("900x600")
        
        # Instance variable to track the number of selected images.
        self.selected_image_count = 0

        # Main frame.
        main_frame = ttk.Frame(self, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Display the calibration details from stage 1.
        calibration_info = (
            f"Camera: {self.camera_name}\n"
            f"Calibration Matrix:\n{np.array2string(self.calibration_matrix, precision=4)}"
        )
        ttk.Label(main_frame, text=calibration_info, foreground="blue").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,10))
        
        # --- Image Selection ---
        ttk.Label(main_frame, text="Select Images:").grid(row=1, column=0, sticky="w")
        self.in_files_var = tk.StringVar(value=DEFAULT_DIR)
        ttk.Entry(main_frame, textvariable=self.in_files_var, width=40, state='readonly').grid(row=1, column=1)
        ttk.Button(main_frame, text="Choose Images", command=self.select_and_copy_images).grid(row=1, column=2)
        
        # --- Feature Type ---
        ttk.Label(main_frame, text="Feature Type:").grid(row=2, column=0, sticky="w")
        self.feat_var = tk.StringVar(value="sift")
        ttk.Combobox(main_frame, textvariable=self.feat_var, values=["sift", "orb"], width=10).grid(row=2, column=1, sticky="w")
        
        # --- Scale ---
        ttk.Label(main_frame, text="Image Scale (%):").grid(row=3, column=0, sticky="w")
        self.scale_var = tk.StringVar(value="100")
        ttk.Spinbox(main_frame, from_=1, to=100, increment=1, textvariable=self.scale_var, width=10).grid(row=3, column=1, sticky="w")
        
        # --- Plot Option ---
        self.plot_var = tk.BooleanVar()
        ttk.Checkbutton(main_frame, text="Plot", variable=self.plot_var).grid(row=4, column=1, sticky="w")
        
        # --- Output File ---
        ttk.Label(main_frame, text="Output File:").grid(row=5, column=0, sticky="w")
        self.outfile_var = tk.StringVar(value="results/out.npz")
        ttk.Entry(main_frame, textvariable=self.outfile_var, width=40).grid(row=5, column=1)
        ttk.Button(main_frame, text="Browse", command=self.browse_outfile).grid(row=5, column=2)
        
        # --- Run Button ---
        ttk.Button(main_frame, text="Run", command=self.run_from_gui).grid(row=6, column=1, pady=10)
        
        global progress
        progress = ttk.Progressbar(main_frame, mode='indeterminate')
        progress.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(5, 0))
        
        self.mainloop()
    
    def select_and_copy_images(self):
        image_files = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        if not image_files:
            return
        
        # Clear existing files in the default directory.
        if os.path.exists(DEFAULT_DIR):
            for file in os.listdir(DEFAULT_DIR):
                file_path = os.path.join(DEFAULT_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        else:
            os.makedirs(DEFAULT_DIR)
        
        # Copy selected files to the default directory.
        for file_path in image_files:
            filename = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(DEFAULT_DIR, filename))
        
        self.in_files_var.set(DEFAULT_DIR)
        self.selected_image_count = len(image_files)
        messagebox.showinfo("Images Loaded", f"{self.selected_image_count} images copied to {DEFAULT_DIR}")
    
    def browse_outfile(self):
        file_selected = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NPZ files", "*.npz")])
        if file_selected:
            self.outfile_var.set(file_selected)
    
    def show_error_popup(self, error_text):
        error_window = tk.Toplevel(self)
        error_window.title("Error Details")
        error_window.geometry("600x400")
        ttk.Label(error_window, text="An error occurred:", foreground="red").pack(pady=5)
        text_area = tk.Text(error_window, wrap="word")
        text_area.insert("1.0", error_text)
        text_area.config(state="disabled")
        text_area.pack(expand=True, fill="both", padx=10, pady=5)
        ttk.Button(error_window, text="Close", command=error_window.destroy).pack(pady=5)
    
    def run_from_gui(self):
        if progress:
            progress.start(10)
        args = {
            "in_files": self.in_files_var.get(),
            "feat": self.feat_var.get(),
            "scale": int(self.scale_var.get()),
            "num_images": self.selected_image_count if self.selected_image_count > 0 else None,
            "plot": self.plot_var.get(),
            "outfile": self.outfile_var.get(),
            "calibration_matrix": self.calibration_matrix  # Pass the calibration matrix here.
        }
        try:
            run(**args)
            messagebox.showinfo("Success", "Execution finished successfully!")
        except Exception as e:
            error_msg = traceback.format_exc()
            self.show_error_popup(error_msg)
        finally:
            if progress:
                progress.stop()

# ====================
# Main Integration
# ====================

def main():
    # Start with the calibration stage GUI.
    app = CalibrationGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
