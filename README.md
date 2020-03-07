# Harris-Corner-Detection using Sobel and Roberts Cross operator
# Installation:
# Requirements:
- OpenCV 4.0.0 [http://opencv.org/]
- g++ 7.4.0
- OS Linux
# How to build:
- cd harrisCornerDetect
- cd Debug
- make clean
- make all
- It will build the binary execution file in the Debug directory. To run the binary file please follow the
following instructions.
# Execution:
In the Debug directory,
- ./harrisCornerDetect image-file Gaussian-Std NMS-threshold NMS-window-size NMS-window-separation
Grad-sum-window-size gradient-type 
- Image-file (string): Complete path of the jpg image file (color or grayscale).
Following input parameters are optional.
- Gaussian Std (float): Standard deviation of the Gaussian filter. The value must be greater than zero.
Default: 1.0
- NMS threshold (float): Threshold applied to the maximum Harris response value. This value must be
between 0 and 1. Default: 0.02
- NMS window size (int): Size of the square window in the response matrix where the local maxima are
computed. The value must be greater than zero and must be smaller than the smaller dimension of
the input image. Default: 3
- NMS window separation (int): The gap between two subsequent NMS windows. The value must be
greater than zero and less than non-maximum suppression window size. Default: 1
- Grad Sum window size (int): Size of the square window of the gradient matrix to compute the
summation of products. The value must be greater than 1 and odd number and must be smaller than
the smaller dimension of the input image. Default: 3
- gradient type (int): 1 is for Sobel and 2 is for Roberts gradient. Default: Sobel.Output:
- The updated image (i.e., out.jpg) with corner locations will be saved in the debug directory.

# Results:
# Original image: 
![Original Image](Results/original.jpg?raw=true "Title")
# Using Sobel operator: 
![Original Image](Results/outSobel.jpg?raw=true "Title")
# Using Roberts operator: 
![Original Image](Results/outRoberts.jpg?raw=true "Title")

# Performance Analysis:
![performance analysis](Results/performance.png?raw=true "Title")
