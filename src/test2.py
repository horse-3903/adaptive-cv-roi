import os
import cv2

def draw_bounding_boxes_from_labels(image_dir, label_dir):
    """
    Maps all images from the raw directory with their corresponding labels, 
    draws the bounding boxes on each image, and displays them.
    
    Args:
    - image_dir (str): Path to the directory containing images.
    - label_dir (str): Path to the directory containing label files (YOLO format).
    """
    # Get the list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read image {image_file}")
            continue

        # Get corresponding label file (replace .jpg/.png with .txt)
        label_file = os.path.join(label_dir, image_file.rsplit('.', 1)[0] + ".txt")

        if not os.path.exists(label_file):
            print(f"Label file not found for {image_file}")
            continue
        
        # Read the label file (YOLO format: class_id, x_center, y_center, width, height)
        with open(label_file, 'r') as file:
            for line in file.readlines():
                # Split the line into class and bounding box coordinates
                class_id, x_center, y_center, width, height = map(float, line.strip().split())

                # Convert normalized YOLO coordinates to pixel values
                img_h, img_w = image.shape[:2]
                x_center = int(x_center * img_w)
                y_center = int(y_center * img_h)
                box_width = int(width * img_w)
                box_height = int(height * img_h)

                # Calculate top-left and bottom-right coordinates of the bounding box
                x1 = int(x_center - box_width / 2)
                y1 = int(y_center - box_height / 2)
                x2 = int(x_center + box_width / 2)
                y2 = int(y_center + box_height / 2)

                # Draw the bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Optionally, draw the class ID on the box
                cv2.putText(image, f"Class: {int(class_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Set window size to match image size
        window_name = f"Image with Boxes: {image_file}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, img_w, img_h)

        # Show the image with drawn bounding boxes
        cv2.imshow(window_name, image)

        # Press any key to close the current image and move to the next
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
image_dir = "./train-test/raw/images"  # Directory containing images
label_dir = "./train-test/raw/labels"  # Directory containing corresponding label files in YOLO format

draw_bounding_boxes_from_labels(image_dir, label_dir)