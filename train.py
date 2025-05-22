from ultralytics import YOLO

def train_model():
    # Load a model
    model = YOLO('yolov5s.pt')  # load a pretrained model

    # Train the model
    results = model.train(
        data='dog_data/data.yaml',  # path to data config file
        epochs=50,                  # number of epochs
        imgsz=640,                 # image size
        batch=8,                   # reduced batch size for CPU
        name='dog_detector',       # experiment name
        patience=10,               # early stopping patience
        save=True,                 # save checkpoints
        device='cpu'               # use CPU
    )

    # Save the final model
    model.save('dog_detector_final.pt')
    print("Training complete! Model saved as 'dog_detector_final.pt'")

if __name__ == "__main__":
    train_model()
