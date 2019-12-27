#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>

#define PI 3.141592

struct Box {
	int ymin, ymax, xmin, xmax;
};

int checkRange(int value) {
	if (value > 255)
		return 255;
	else if (value < 0)
		return 0;
	else return value;
}

cv::Mat& Lightness(cv::Mat& I, int value){
  CV_Assert(I.depth() != sizeof(uchar));
  switch(I.channels())  {
  case 1:
    for( int i = 0; i < I.rows; ++i)
        for( int j = 0; j < I.cols; ++j )
            I.at<uchar>(i,j) = (I.at<uchar>(i,j)/32)*32;
    break;
  case 3:
    cv::Mat_<cv::Vec3b> _I = I;
    for( int i = 0; i < I.rows; ++i)
        for( int j = 0; j < I.cols; ++j ){
            _I(i,j)[0] = checkRange(_I(i,j)[0]+value);
            _I(i,j)[1] = checkRange(_I(i,j)[1]+value);
            _I(i,j)[2] = checkRange(_I(i,j)[2]+value);
        }
    I = _I;
    break;
  }
  return I;
}

cv::Mat& Contrast(cv::Mat& I, int value) {
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels()) {
	case 1:
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = (I.at<uchar>(i, j) / 32) * 32;
		break;
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				_I(i, j)[0] = checkRange(_I(i, j)[0]* (1 + value / 100.0));
				_I(i, j)[1] = checkRange(_I(i, j)[1]* (1 + value / 100.0));
				_I(i, j)[2] = checkRange(_I(i, j)[2]* (1 + value / 100.0));
			}
		I = _I;
		break;
	}
	return I;
}

float Area (cv::Mat& I) {
	CV_Assert(I.depth() != sizeof(uchar));
	float S = 0;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				int grey = (_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3;
				if (grey < 10)
					S++;
			}
	}
	return S;
}

float Len(cv::Mat& I) {
		CV_Assert(I.depth() != sizeof(uchar));
		float len = 0;
		switch (I.channels()) {
		case 3:
			cv::Mat_<cv::Vec3b> _I = I;
			for (int i = 1; i < I.rows - 1; ++i)
				for (int j = 1; j < I.cols - 1; ++j) {
					int grey = (_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3;
					bool isEdge = false;
					if (grey < 10) {
						for (int k = i - 1; k <= i + 1; ++k) {
							for (int l = j - 1; l <= j + 1; ++l) {
								if (((_I(k, l)[0] + _I(k, l)[1] + _I(k, l)[2]) / 3) > 250) {
									len++;
									isEdge = true;
									break;
								}
							}
							if (isEdge)
								break;
						}
					}
				}
		return len;
	}
}

float Malinowska(float S, float L) {
	return ((L / (2 * sqrt(PI *S))) - 1);
}

float moment(cv::Mat& I, int p, int q) {
	CV_Assert(I.depth() != sizeof(uchar));
	float m = 0;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if((_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3 < 10) {
					m += pow(i, p) * pow(j, q);
				}
			}
	}
	return m;
}

float momentCentr(cv::Mat& I, int p, int q) {
	CV_Assert(I.depth() != sizeof(uchar));
	float MC = 0;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		float m00, m10, m01, i_, j_;
		m00 = moment(I, 0, 0);
		m10 = moment(I, 1, 0);
		m01 = moment(I, 0, 1);
		i_ = m10 / m00;
		j_ = m01 / m00;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if ((_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3 < 10) {
					MC += pow(i-i_, p) * pow(j-j_, q);
				}
			}
	}
	return MC;
}

cv::Mat& Grey(cv::Mat& I) {
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				int grey = (_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3;
				_I(i, j)[0] = checkRange(_I(i, j)[2] - grey);
				_I(i, j)[1] = checkRange(_I(i, j)[2] - grey);
				_I(i, j)[2] = checkRange(_I(i, j)[2] - grey);
			}
		I = _I;
		break;
	}
	return I;
}

cv::Mat RankFilter(cv::Mat& I, int rank, int size) {
	if (rank >= 0 && rank < size^2 && size%2==1 && size>0)
	{
		CV_Assert(I.depth() != sizeof(uchar));
		cv::Mat  res(I.rows, I.cols, CV_8UC3);
		switch (I.channels()) {
		case 3:
			cv::Mat_<cv::Vec3b> _I = I;
			cv::Mat_<cv::Vec3b> _R = res;
			for (int i = (size-1)/2; i < I.rows - (size - 1) / 2; ++i)
				for (int j = (size - 1) / 2; j < I.cols - (size - 1) / 2; ++j) {
					std::vector<int> red;
					std::vector<int> green;
					std::vector<int> blue;
					for (int k = i - (size - 1) / 2; k <= i + (size - 1) / 2; ++k) {
						for (int l = j - (size - 1) / 2; l <= j + (size - 1) / 2; ++l) {
							red.push_back(_I(k, l)[2]);
							green.push_back(_I(k, l)[1]);
							blue.push_back(_I(k, l)[0]);
						}
					}
					std::sort(red.begin(), red.end());
					std::sort(green.begin(), green.end());
					std::sort(blue.begin(), blue.end());
					_R(i, j)[0] = blue[rank];
					_R(i, j)[1] = green[rank];
					_R(i, j)[2] = red[rank];
				}
			res = _R;
			break;
		}
		return res;
	}
	else {
		std::cout << "Error: Niewlasciwy size lub rank" << std::endl;
	}
}

cv::Mat& Hist(cv::Mat& I) {
	long table[8];
	for (int i = 0; i < 8; i++) {
		table[i] = 0;
	}
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				int grey = (_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3;
				int temp = grey / 32;
//				std::cout << temp << std::endl;
				table[temp]++;
			}
		I = _I;
		break;
	}
	int sum=0;
	for (int i = 0; i < 8; i++) {
		sum += table[i];
		std::cout << 32 * i << " - " << 31 + 32 * i << " : " << table[i] << std::endl;
	}
	std::cout << "sum: " << sum << std::endl;
	return I;
}

cv::Mat selectMax(cv::Mat& I){
    CV_Assert(I.depth() != sizeof(uchar));
    cv::Mat  res(I.rows,I.cols, CV_8UC3); // jaki obrazek ma byæ przygotowany - trzy bajty reprezentuj¹ piksel wynikowy
    switch(I.channels())  {
    case 3:
        cv::Mat_<cv::Vec3b> _I = I;
        cv::Mat_<cv::Vec3b> _R = res;
        for( int i = 0; i < I.rows; ++i)
            for( int j = 0; j < I.cols; ++j ){
                int sel = (_I(i,j)[0] < _I(i,j)[1])?1:0;
                sel = _I(i,j)[sel] < _I(i,j)[2]?2:sel;
                _R(i,j)[0] = sel==0?255:0;
                _R(i,j)[1] = sel==1?255:0;
                _R(i,j)[2] = sel==2?255:0;
            }
        res = _R;
        break;
    }
    return res;
}

cv::Mat filter(cv::Mat& I, int blue, int red, int green) {
	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat  res(I.rows, I.cols, CV_8UC3);
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		cv::Mat_<cv::Vec3b> _R = res;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if (_I(i, j)[2] == red && _I(i, j)[1] == green && _I(i, j)[0] == blue) {
					_R(i, j)[2] = 0;
					_R(i, j)[1] = 0;
					_R(i, j)[0] = 0;
				}
				else {
					_R(i, j)[2] = 255;
					_R(i, j)[1] = 255;
					_R(i, j)[0] = 255;
				}
			}
		res = _R;
		break;
	}
	return res;
}

int max(int fst, int snd) {
	if (fst > snd)
		return fst;
	else
		return snd;
}

int min(int fst, int snd) {
	if (fst < snd)
		return fst;
	else
		return snd;
}

std::pair<int, int> centerGeom(cv::Mat& I) {
	CV_Assert(I.depth() != sizeof(uchar));
	std::pair<int, int> center;
	int ymin, ymax, xmin, xmax;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		ymax = 0;
		xmax = 0;
		ymin = I.cols;
		xmin = I.rows;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if ((_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3 < 10) {
					ymax = max(ymax, j);
					xmax = max(xmax, i);
					ymin = min(ymin, j);
					xmin = min(xmin, i);
				}
			}
		center.first = xmin + (xmax - xmin) / 2;
		center.second = ymin + (ymax - ymin) / 2;
		for (int i = center.first - 1; i <= center.first + 1; ++i)
			for (int j = center.second - 1; j <= center.second + 1; ++j) {
				_I(i, j)[0] = 255;
				_I(i, j)[1] = 255;
				_I(i, j)[2] = 0;
			}
		break;
	}
	return center;
}

std::pair<int, int> imageCenter(cv::Mat& I) {
	CV_Assert(I.depth() != sizeof(uchar));
	std::pair<int, int> center;
	int ymin, ymax, xmin, xmax;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		float m00, m10, m01, i_, j_;
		m00 = moment(I, 0, 0);
		m10 = moment(I, 1, 0);
		m01 = moment(I, 0, 1);
		i_ = m10 / m00;
		j_ = m01 / m00;
		center.first = i_;
		center.second = j_;
		for (int i = center.first - 1; i <= center.first + 1; ++i)
			for (int j = center.second - 1; j <= center.second + 1; ++j) {
				_I(i, j)[0] = 255;
				_I(i, j)[1] = 0;
				_I(i, j)[2] = 255;
			}
		break;
	}
	return center;
}

std::string intToString(int x) {
	std::stringstream ss;
	ss << x;
	return ss.str();
}

int main(int, char *[]) {
	std::cout << "Start ..." << std::endl;

	float area[5];
	float len[5];
	float w3[5];
	float M1[5];
	float M7[5];

/*	std::string tablestr[5];
	tablestr[0] = "elipsa.dib";
	tablestr[1] = "elipsa1.dib";
	tablestr[2] = "kolo.dib";
	tablestr[3] = "prost.dib";
	tablestr[4] = "troj.dib";
	cv::Mat image[5];
	for (int i = 0; i < 5; i++) {
		image[i] = cv::imread(tablestr[i]);
		area[i] = Area(image[i]);
		len[i] = Len(image[i]);
		w3[i] = Malinowska(area[i], len[i]);
		M1[i] = (momentCentr(image[i], 2, 0) + momentCentr(image[i], 0, 2)) / pow(moment(image[i], 0, 0), 2);
		M7[i] = (momentCentr(image[i], 2, 0)*momentCentr(image[i], 0, 2) - pow(momentCentr(image[i], 1, 1), 2)) / pow(moment(image[i], 0, 0), 4);
		std::cout << i << ". Plik: " << tablestr[i] << " S: " << area[i] << " L: " << len[i] << " W3: " << w3[i] << " M1: " << M1[i] << " M7: " << M7[i] <<  std::endl;
		cv::imshow(tablestr[1], image[i]);
		cv::waitKey(-1);
	} */
	/*
	cv::Mat image = cv::imread("strzalki_1.dib");
	cv::imshow("strzalki", image);
	cv::Mat iStr[5];
	iStr[0] = filter(image, 0, 180, 0);
	iStr[1] = filter(image, 0, 135, 45);
	iStr[2] = filter(image, 0, 90, 90);
	iStr[3] = filter(image, 0, 45, 135);
	iStr[4] = filter(image, 0, 0, 180);
	std::pair<int, int> geom_cent[5];
	std::pair<int, int> img_cent[5];
	float angle[5];
	for (int i = 0; i < 5; i++) {
		area[i] = Area(iStr[i]);
		len[i] = Len(iStr[i]);
		w3[i] = Malinowska(area[i], len[i]);
		M1[i] = (momentCentr(iStr[i], 2, 0) + momentCentr(iStr[i], 0, 2)) / pow(moment(iStr[i], 0, 0), 2);
		M7[i] = (momentCentr(iStr[i], 2, 0)*momentCentr(iStr[i], 0, 2) - pow(momentCentr(iStr[i], 1, 1), 2)) / pow(moment(iStr[i], 0, 0), 4);
		geom_cent[i] = centerGeom(iStr[i]);
		img_cent[i] = imageCenter(iStr[i]);
		angle[i] = -atan2(img_cent[i].first - geom_cent[i].first, img_cent[i].second - geom_cent[i].second) * 180 / PI;
		std::cout << "Strzalka: R" << i*45 << " nachylenie: " << angle[i] << " S: " << area[i] << " L: " << len[i] << " W3: " << w3[i] << " M1: " << M1[i] << " M7: " << M7[i] << std::endl;
		cv::imshow(intToString(i), iStr[i]);
		cv::waitKey(-1);
	} */
	

/*  cv::Mat image1 = image(cv::Rect(0 ,0, image.cols/2, image.rows/2));
	cv::Mat image2 = image(cv::Rect(image.cols / 2, 0, image.cols / 2, image.rows / 2));
	cv::Mat image3 = image(cv::Rect(0, image.rows / 2, image.cols / 2, image.rows / 2));
	cv::Mat image4 = image(cv::Rect(image.cols / 2, image.rows / 2, image.cols / 2, image.rows / 2));
	int lightness, contrast;
	std::cout << "Podaj wzrost jasnosci: " << std::endl;
	std::cin >> lightness;
	std::cout << "Podaj kontrast: " << std::endl;
	std::cin >> contrast;
	image1 = Lightness(image1, lightness);
	image2 = Contrast(image2, contrast);
	image3 = Grey(image3);
	Hist(image); */
	cv::Mat image = cv::imread("Lena.png");
    cv::imshow("Lena_przed",image);
	cv::Mat img1 = RankFilter(image, 1, 5);
	cv::imshow("Lena_po", img1);
//    cv::imshow("Max",max);
//    std::cout << image2.isContinuous() << max.isContinuous() << std::endl; //is continious image 2 nie jest spójnyu bo odwo³uje siê do fragmentu innego obrazu
//    cv::imwrite("out.png", img1);
    cv::waitKey(-1); //¿eby siê wyœwietli³o (samo imshow nie wyœwietla obrazka)0
    return 0;
}
