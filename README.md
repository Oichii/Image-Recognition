
# Image Recognition

## About
This project implements Superpharm logo recognition using classical image processing metods. OpenCV image processing functions was not used (OpenCV was used only for reading and displaying image).

## Implementation

* reading image from file
* converting RGB color space to HSV
* segmentation based on color range
* morphological operations - open and close to enhance binary mask
* separating objects using connected components algorithm
* calculating M1 for objects bigger than specific value of their area
* calculating distance between recognised components of the logo
* decision werther logo is found on the image

## Technology
* C ++

