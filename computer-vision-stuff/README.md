# Garbage Classifier

A Computer Vision program that uses R-CNN and YOLOv5 to detect and classify an object

## Dependencies

The project requires the following Python libraries:

- OpenCV (`cv2`)
- PyTorch (`torch`)
- Torchvision
- Pillow (`PIL`)
- NumPy
- YOLOv5

## Installation (for powershell)

1. Clone this repository:
```
git clone https://github.com/ableflyer/robotalgos.git
cd robotalgos/computer-vision-stuff
```
2. Create and activate a venv file
```
python -m venv venv # or py -m venv venv
./venv/Scripts/Activate.ps1
```

3. Install the required packages:
```
pip install torch torchvision opencv-python pillow numpy
```

4. Install YOLOv5:
```
pip install yolov5
```

5. Run the program:
```
python cvtest.py
```
