import os
import shutil
from PIL import Image
import random
from pathlib import Path

def convert_to_yolo_format():
    # Source directories
    your_dog_dir = 'dog_data/your_dog'
    not_your_dog_dir = 'dog_data/not_your_dog'
    
    # Create YOLO directories if they don't exist
    os.makedirs('dog_data/images/train', exist_ok=True)
    os.makedirs('dog_data/images/val', exist_ok=True)
    os.makedirs('dog_data/labels/train', exist_ok=True)
    os.makedirs('dog_data/labels/val', exist_ok=True)
    
    # Process your_dog images
    for img_file in os.listdir(your_dog_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Open image to get dimensions
            img_path = os.path.join(your_dog_dir, img_file)
            with Image.open(img_path) as img:
                width, height = img.size
            
            # Create YOLO format label (full image bounding box)
            # Format: class x_center y_center width height
            # All values normalized to [0,1]
            label = f"0 0.5 0.5 1.0 1.0\n"  # class 0 is dog
            
            # Split into train/val (80/20 split)
            if random.random() < 0.8:
                dest_img_dir = 'dog_data/images/train'
                dest_label_dir = 'dog_data/labels/train'
            else:
                dest_img_dir = 'dog_data/images/val'
                dest_label_dir = 'dog_data/labels/val'
            
            # Copy image
            shutil.copy2(img_path, os.path.join(dest_img_dir, img_file))
            
            # Create label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            with open(os.path.join(dest_label_dir, label_file), 'w') as f:
                f.write(label)
    
    print("Conversion complete!")
    print("Images and labels have been organized into train/val splits")
    print("Each image has a corresponding .txt file with YOLO format annotations")

if __name__ == "__main__":
    convert_to_yolo_format() 