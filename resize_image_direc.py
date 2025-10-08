
import os
import cv2

def read_image(filename, scale):
    # load images
    im = cv2.imread(filename)
    
    # Check if image was loaded properly
    if im is None:
        print(f"Error: Could not read image {filename}")
        return None
        
    # Calculate new dimensions
    width = int(im.shape[1] * scale / 100)
    height = int(im.shape[0] * scale / 100)
    dim = (width, height)

    # resize image
    im = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    return im

def resize_images_in_directory(directory, scale_percent):
    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    
    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist")
        return
    
    # Counter for processed images
    processed = 0
    
    for filename in os.listdir(directory):
        if filename.lower().endswith(image_extensions):
            filepath = os.path.join(directory, filename)
            
            # Resize the image
            resized_image = read_image(filepath, scale_percent)
            
            # If image was successfully resized, save it
            if resized_image is not None:
                cv2.imwrite(filepath, resized_image)
                print(f"Resized and overwritten: {filename}")
                processed += 1
    
    print(f"Completed: {processed} images processed")

# Example usage
if __name__ == '__main__':
    # Get directory path from user
    directory = r'C:\Users\saiab\Documents\sfm-master\colmap_workspace\images'
    
    # Get scaling percentage from user
    try:
        scale_percent = 25
        resize_images_in_directory(directory, scale_percent)
    except ValueError:
        print("Error: Please enter a valid number for scaling percentage")