# UAV Video Panorama Stitching
A Python-based implementation for creating panoramic images from UAV video footage using OpenCV.

## Overview
This repository contains two alternative implementations for creating panoramic images from video sequences. Both implementations use feature detection and matching techniques to align and stitch consecutive video frames together.

## Features
- Video frame extraction and processing
- Image calibration and distortion correction
- Multiple feature detection methods (SURF, SIFT, MSER)
- Adaptive frame selection
- Black border removal
- Image stretching and perspective correction
- Artifact removal using median blur

## Dependencies
- OpenCV (with contrib modules for SURF/SIFT)
- NumPy
- SciPy
- Matplotlib

## Installation
```bash
pip install opencv-contrib-python numpy scipy matplotlib
```

## Usage

### Main Implementation (panoramamod.py)
```python
# Load calibration data
mtx = np.load("path/to/calibration/mtx.npz")['arr_0']
dist = np.load("path/to/calibration/dist.npz")['arr_0']
newcameramtx = np.load("path/to/calibration/newcameramtx.npz")['arr_0']
roi = np.load("path/to/calibration/roi.npz")['arr_0']

# Create parameter dictionary
parameter_dict = dict(mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi)

# Process video
videodir = 'path/to/video.avi'
cap = cv2.VideoCapture(videodir)
```

### Alternative Implementation (alternativePanorama.py)
```python
# Import and run
from alternativePanorama import generatePanaroma, videoread

# Read video frames
frames = videoread('path/to/video.mp4')

# Generate panorama
panorama = generatePanaroma(frames)
```

## Key Components

### Image Calibration
Both implementations support camera calibration to correct lens distortion:

```python
def calibrate_image(img, mtx, dist, newcameramtx, roi):
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    return dst
```

### Feature Detection and Matching
The system supports multiple feature detection methods:

1. SURF (panoramamod.py):
```python
surf = cv2.xfeatures2d.SURF_create(surfparameter)
kp1, des1 = surf.detectAndCompute(gray, None)
```

2. SIFT (alternativePanorama.py):
```python
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(added, None)
```

3. MSER (alternativePanorama.py):
```python
mser = cv2.MSER_create()
kp1 = mser.detect(added, None)
```

### Image Stitching Process
1. Feature detection and matching
2. Homography calculation
3. Perspective warping
4. Image blending
5. Black border removal
6. Final cleanup using median blur

## Configuration Parameters

### panoramamod.py
- `stepnum`: Frame selection interval
- `min_keypoint`: Minimum number of matching keypoints (default: 70)
- `offset`: Image padding size
- `framesizefactor`: Frame resize factor
- `surfparameter`: SURF detector parameter

### alternativePanorama.py
- `step`: Frame selection interval
- `offset`: Image padding size
- `MIN_MATCH_COUNT`: Minimum number of matches for adaptive stitching

## Output
The scripts generate intermediate results and final panorama:
- Base frame
- Component images
- Iteration results
- Final stitched panorama

Output files are saved in the specified output directory:
```
output_image/
├── baseframe.png
├── componentimage{n}.png
├── resultiteration{n}.png
└── finalresult.png
```

## Limitations
- Requires pre-calibrated camera parameters
- Memory intensive for long video sequences
- Performance depends on video quality and movement patterns
- May produce artifacts in scenes with rapid movement

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
[MIT License](LICENSE)
