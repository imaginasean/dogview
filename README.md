# Dog Detector

A real-time dog detection system that can identify specific dogs using YOLOv5 and custom training. This project uses computer vision to detect dogs in a video stream and can distinguish between a specific dog (Annie) and other dogs.

## Features

- Real-time dog detection using YOLOv5
- Custom model training for specific dog identification
- Text-to-speech announcements for detections
- Confidence score reporting
- Support for both general dog detection and specific dog identification

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- YOLOv5
- pyttsx3 (for text-to-speech)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/dog-detector.git
cd dog-detector
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Run the dog detection script:

```bash
python dog.py
```

2. The script will:
   - Open your webcam
   - Detect dogs in real-time
   - Announce when Annie is detected
   - Display confidence scores
   - Press 'q' to quit

## Training

To train your own custom model:

1. Prepare your dataset in the `dog_data` directory
2. Run the conversion script:

```bash
python convert_to_yolo.py
```

3. Train the model:

```bash
python train.py
```

## License

MIT License

## Contributing

Feel free to open issues or submit pull requests for any improvements.
