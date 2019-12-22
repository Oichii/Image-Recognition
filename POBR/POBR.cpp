#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;
int main()
{
    Mat img = imread("img.jpg");
    namedWindow("image_to detect", WINDOW_NORMAL);
    imshow("image_to_detect", img);
    waitKey(0);

    return 0;
}