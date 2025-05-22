import os
import shutil
from pathlib import Path
import yaml

def create_yolo_dataset():
    # Create necessary directories
    os.makedirs('dog_data/images/train', exist_ok=True)
    os.makedirs('dog_data/images/val', exist_ok=True)
    os.makedirs('dog_data/labels/train', exist_ok=True)
    os.makedirs('dog_data/labels/val', exist_ok=True)
    
    # Create data.yaml
    data_yaml = {
        'path': 'dog_data',
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'dog'
        }
    }
    
    with open('dog_data/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    print("Dataset structure created. Please add your images and labels to the appropriate directories.")
    print("For each image, create a corresponding .txt file with YOLO format annotations.")
    print("Example label format: <class> <x_center> <y_center> <width> <height>")
    print("All coordinates should be normalized to [0,1]")

if __name__ == "__main__":
    create_yolo_dataset() 