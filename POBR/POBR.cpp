#include<opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

int checkRange(int value) {
	if (value > 255)
		return 255;
	else if (value < 0)
		return 0;
	else return value;
}

Mat RankFilter(Mat& I, int rank, int size) {
	if (rank >= 0 && rank < (size*size) && size % 2 == 1 && size>0)
	{
		CV_Assert(I.depth() != sizeof(uchar));
		Mat  res(I.rows, I.cols, CV_8UC3);
		switch (I.channels()) {
		case 3:
			Mat_<Vec3b> _I = I;
			Mat_<Vec3b> _R = res;
			for (int i = (size - 1) / 2; i < I.rows - (size - 1) / 2; ++i)
				for (int j = (size - 1) / 2; j < I.cols - (size - 1) / 2; ++j) {
					vector<int> red;
					vector<int> green;
					vector<int> blue;
					for (int k = i - (size - 1) / 2; k <= i + (size - 1) / 2; ++k) {
						for (int l = j - (size - 1) / 2; l <= j + (size - 1) / 2; ++l) {
							red.push_back(_I(k, l)[2]);
							green.push_back(_I(k, l)[1]);
							blue.push_back(_I(k, l)[0]);
						}
					}
					sort(red.begin(), red.end());
					sort(green.begin(), green.end());
					sort(blue.begin(), blue.end());
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
		cout << "Error: Niewlasciwy rozmiar filtra lub wartość" << endl;
		return I;
	}
}
Mat& Segment(Mat& I, int value) {
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels()) {
	case 1:
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = (I.at<uchar>(i, j) / 32) * 32;
		break;
	case 3:
		Mat_<Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				//int gray = (_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3; //BGR
				float r_ = _I(i, j)[2] / 255.0;
				float g_ = _I(i, j)[1] / 255.0;
				float b_ = _I(i, j)[0] / 255.0;
				float c[] = {r_, g_,b_};
				float Cmax = max(max(b_, g_), r_); // max(r_, b_, g_);
				//cout <<"Cmax" <<Cmax << endl;
				float Cmin = min(min(b_, g_), r_);  // min(r_, b_, g_);
				// cout << "Cmin" <<Cmin << endl;
				float delta = Cmax - Cmin;
				float H = 0;
				float S = 0;
				//H = (g_ - b_) / delta % 6 * 60.0;

				if (delta == 0) {
					H = 0;
				}
				else if(Cmax == r_) {
					H = fmod((60 * ((g_ - b_) / delta) + 360), 360);
				}
				else if (Cmax == g_) {
					H = fmod((60 * ((g_ - b_) / delta) + 120), 360);
				}
				else if (Cmax == g_) {
					H = fmod((60 * ((g_ - b_) / delta) + 240), 360);
				}

				if (Cmax == 0) {
					S = 0;
				}
				else {
					S = (delta / Cmax )*100;
				}
				//cout << S << endl;
				if ((H > 160 && H < 174) && (S > 100 )) {
					_I(i, j)[0] = 255;
					_I(i, j)[1] = 255;
					_I(i, j)[2] = 255;
				}
				else if ((H >200) && (S>55)) {
					_I(i, j)[0] = 255;
					_I(i, j)[1] = 255;
					_I(i, j)[2] = 255;
				}
				else if (H<9 && S>80) {
					_I(i, j)[0] = 255;
					_I(i, j)[1] = 255;
					_I(i, j)[2] = 255;
				}
				else{
					_I(i, j)[0] = 0;
					_I(i, j)[1] = 0;
					_I(i, j)[2] = 0;
				}
			}
		I = _I;
		break;
	}
	return I;
}
Mat& Lightness(Mat& I, int value) {
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels()) {
	case 1:
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = (I.at<uchar>(i, j) / 32) * 32;
		break;
	case 3:
		Mat_<Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				_I(i, j)[0] = checkRange(_I(i, j)[0] + value);
				_I(i, j)[1] = checkRange(_I(i, j)[1] + value);
				_I(i, j)[2] = checkRange(_I(i, j)[2] + value);
			}
		I = _I;
		break;
	}
	return I;
}

Mat& Contrast(Mat& I, int value) {
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels()) {
	case 1:
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = (I.at<uchar>(i, j) / 32) * 32;
		break;
	case 3:
		cv::Mat_<Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				_I(i, j)[0] = checkRange(_I(i, j)[0] * value );
				_I(i, j)[1] = checkRange(_I(i, j)[1] * value );
				_I(i, j)[2] = checkRange(_I(i, j)[2] * value );
			}
		I = _I;
		break;
	}
	return I;
}

int moment(cv::Mat& I, int p, int q) { //area of white object 
	int S = 0;
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels()) {
	case 1:
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = (I.at<uchar>(i, j) / 32) * 32;
		break;
	case 3:
		cv::Mat_<Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				int gray = (_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3;
				if (gray > 10) { //< 10 obiect is black 
					S += pow(i, p) * pow(j, q);

				}
			}
	}
	return S;
}

int Len(cv::Mat& I) { //length of white obiect
	int S = 0;
	CV_Assert(I.depth() != sizeof(uchar));
	switch (I.channels()) {
	case 1:
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j)
				I.at<uchar>(i, j) = (I.at<uchar>(i, j) / 32) * 32;
		break;
	case 3:
		cv::Mat_<Vec3b> _I = I;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				int gray = (_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3;
				if (gray > 10) { //<10 obiect is black 
					bool edge = false;
					for (int k = i - 1; k <= i + 1; k++) {
						for (int l = j - 1; l <= j + 1; l++) {
							int neigh = (_I(k, l)[0] + _I(k, l)[1] + _I(k,l)[2]) / 3;
							if (neigh < 10) { //>10 neibhbour is white 
								S++;
								edge = true;
								break;
							}
						}
						if (edge == true) {
							break;
						}
					}
				}
			}
	}
	return S;
}

float momentCentr(Mat& I, int p, int q) {
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
				if ((_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3 > 10) {
					MC += pow(i - i_, p) * pow(j - j_, q);
				}
			}
	}
	return MC;
}

float M1(Mat& I) {
	CV_Assert(I.depth() != sizeof(uchar));
	float MC = 0;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		float m00, M02, M20;
		M20 = momentCentr(I, 2, 0);
		M02 = momentCentr(I, 0, 2);
		m00 = moment(I, 0, 0);
		
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if ((_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3 > 10) {
					MC += (M20 + M02) / pow(m00, 2);
				}
			}
	}
	return MC;
}

float M7(Mat& I) {
	CV_Assert(I.depth() != sizeof(uchar));
	float MC = 0;
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		float m00, M02, M20, M11;
		M20 = momentCentr(I, 2, 0);
		M02 = momentCentr(I, 0, 2);
		M11 = momentCentr(I, 1, 1);
		m00 = moment(I, 0, 0);

		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if ((_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3 < 10) {
					MC += (M20 + M02 - pow(M11, 2)) / pow(m00, 4);
				}
			}
	}
	return MC;
}

Mat open(Mat& I, int size) {
	CV_Assert(I.depth() != sizeof(uchar));
	Mat  res(I.rows, I.cols, CV_8UC3);
	switch (I.channels()) {
	case 3:
		res = RankFilter(I, 1, size);
		return RankFilter(res, (size*size)-1, size);
	}
}

Mat close(Mat& I, int size) {
	CV_Assert(I.depth() != sizeof(uchar));
	Mat  res(I.rows, I.cols, CV_8UC3);
	switch (I.channels()) {
	case 3:
		res = RankFilter(res, (size*size) - 1, size); 
		return RankFilter(I, 1, size);
	}
}

void DFS(Mat I, int x, int y, int label) {
	int row_count = I.rows;
	int col_count = I.cols;
	if (x < 0 || x == row_count) return; // out of bounds
	if (y < 0 || y == col_count) return; // out of bounds
	const int dx[] = { +1, 0, -1, 0 };
	const int dy[] = { 0, +1, 0, -1 };
	switch (I.channels()) {
	case 3:
		cv::Mat_<cv::Vec3b> _I = I;
		if (_I(x, y)[0] != 255 && _I(x, y)[1] != 255 && _I(x, y)[2] != 255) return; // already labeled or not marked with 1 in m
		_I(x, y)[0] = label;
		_I(x, y)[1] = label;
		_I(x, y)[2] = label;
		I = _I;
		for (int direction = 0; direction < 4; ++direction) {
			//cout << "hello" << endl;
			DFS(I, x + dx[direction], y + dy[direction], label);
		}		
	}
}

Mat connectedComp(Mat& I) {
	Mat labeled(I.rows, I.cols, CV_8UC3);
	CV_Assert(I.depth() != sizeof(uchar));
	Mat  res(I.rows, I.cols, CV_8UC3);
	switch (I.channels()) {
	case 3:
		cv::Mat_<Vec3b> _I = I;
		cv::Mat_<Vec3b> _L = labeled;
		int label = 0;
		for (int i = 0; i < I.rows; ++i) {
			for (int j = 0; j < I.cols; ++j) {
				int gray = (_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3;
				if (_I(i, j)[0] != 0 && _I(i, j)[1] !=0 && _I(i, j)[2] != 0) {
					DFS(I, i, j, ++label);
					//cout << label << endl;
				}
			}
		}
		I = _I;
		return I;
	}
}

Mat devideImage(Mat& I, int blue, int red, int green) {
	CV_Assert(I.depth() != sizeof(uchar));
	cv::Mat  res(I.rows, I.cols, CV_8UC3);
	switch (I.channels()) {
	case 3:
		Mat_<Vec3b> _I = I;
		Mat_<Vec3b> _R = res;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if (_I(i, j)[2] == red && _I(i, j)[1] == green && _I(i, j)[0] == blue) {
					_R(i, j)[2] = 255;
					_R(i, j)[1] = 255;
					_R(i, j)[0] = 255;
				}
				else {
					_R(i, j)[2] = 0;
					_R(i, j)[1] = 0;
					_R(i, j)[0] = 0;
				}
			}
		res = _R;
		break;
	}
	return res;
}

//Mat bounding_box(Mat I) {
//}

Mat centerGeom(Mat& I, Mat& Base) {
	CV_Assert(I.depth() != sizeof(uchar));
	pair<int, int> center;
	int ymin, ymax, xmin, xmax;
	switch (I.channels()) {
	case 3:
		Mat_<cv::Vec3b> _I = I;
		ymax = 0;
		xmax = 0;
		ymin = I.cols;
		xmin = I.rows;
		for (int i = 0; i < I.rows; ++i)
			for (int j = 0; j < I.cols; ++j) {
				if ((_I(i, j)[0] + _I(i, j)[1] + _I(i, j)[2]) / 3 > 10) {
					ymax = max(ymax, j);
					xmax = max(xmax, i);
					ymin = min(ymin, j);
					xmin = min(xmin, i);
				}
			}
		center.first = xmin + (xmax - xmin) / 2;
		center.second = ymin + (ymax - ymin) / 2;
		int width = xmax - xmin;
		int height = ymax - ymin;
		//int list[]
		for (int i = center.first -width/2; i <= center.first + width/2; ++i)
			for (int j = center.second - height/2; j <= center.second + height/2; ++j) {
				_I(i, j)[0] = 100;
				_I(i, j)[1] = 100;
				_I(i, j)[2] = 100;
			}
		I = _I;
		Base += I;
		break;
	}
	return Base;
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

float distance(pair<int, int> c1, pair<int, int> c2) {
	return pow(pow(abs(c1.first - c2.first), 2) + pow(abs(c1.second - c2.second), 2), 0.5);
}


int main()
{
	// rect jest 400 x 200 pix 
	Mat rect = imread("rect.png");
	cout << moment(rect, 0,0) << endl;
    //
	Mat img_clean = imread("S1.jpg");
	Mat img = imread("S1.jpg");
	//Mat img = imread("S0_wzor.png");
	cout << "wczytanie" << endl;
    //cv::namedWindow("image_to detect", WINDOW_NORMAL);
    //cv::imshow("image_to_detect", img);
	cout << "segmentacja" << endl;

	Mat img2 = Segment(img, 100);
	//img2 = open(img2, 3);
	cout << "zamkniecie" << endl;
	img2 = close(img2, 3);
	namedWindow("segmented", WINDOW_NORMAL);
	cv::imshow("segmented", img2);
	waitKey(0);

	Mat con = connectedComp(img2);
	cv::imshow("segmented", con);
	waitKey(0);
	vector<Mat> letterM;
	vector<Mat> letterA;
	vector<Mat> logo;
	for (int i = 1; i <= 255; i++) {
		Mat selected = devideImage(con, i, i, i);
		if (moment(selected, 0, 0)>100) {
			cout << moment(selected, 0, 0) << endl;
			float curr_M1 = M1(selected);
			float curr_M7 = M7(selected);
			cout << "M1 " << curr_M1 << endl;
			cout << "M7 " << curr_M7 << endl;
			//cv::namedWindow("image", WINDOW_NORMAL);
			// sprawdzic zakres momentow dla 3 elementow, obliczyc odleglosc miedzy tymi ktorych momenty sie zgadzaja jesli odpowiednia to rozpoznane - tylko l/2s zamiast odleglosci moze
			// obliczac jak daleko jestem od srodka zakresu danego parametru 
			//cv::namedWindow("image", WINDOW_NORMAL);
			//cv::imshow("image", selected);
			//cv::waitKey(0);
		
			if (curr_M1 > 300.0 && curr_M1 < 800.0) {
				logo.push_back(selected);
				cout << "dodano logo" << endl;
				//cv::namedWindow("image", WINDOW_NORMAL);
				//cv::imshow("image", selected);
				//cv::waitKey(0);
			}
			else if (curr_M1 > 90.0 && curr_M1 < 110.0) { //70 - 110
				letterM.push_back(selected);
				cout << "dodano m" << endl;
				//cv::namedWindow("image", WINDOW_NORMAL);
				//cv::imshow("image", selected);
				//cv::waitKey(0);
			}
			else if (curr_M1 > 60.0 && curr_M1 < 90.0) {
				cout << "dodano A" << endl;
				letterA.push_back(selected);
				//cv::namedWindow("image", WINDOW_NORMAL);
				//cv::imshow("image", selected);
				//cv::waitKey(0);

			}
			
		}
	}
	if (letterM.size() < 1 || logo.size() < 1 || letterA.size() < 1) {
		cout << "nie znaleziono" << endl;
	}
	else {
		vector<Mat> bboxes;
		vector<pair<int,int>> centersA;
		vector<pair<int, int>> centersM;
		vector<pair<int, int>> centersLogo;
		for (int p = 0; p < letterA.size(); p++) {
			 //obliczyc bounding box
			centersA.push_back(imageCenter(letterA[p]));
			//cv::imshow("image", logoElements[p]);
			//cv::waitKey(0);
		}
		for (int p = 0; p < letterM.size(); p++) {
			//obliczyc bounding box
			centersM.push_back(imageCenter(letterM[p]));
			//cv::imshow("image", logoElements[p]);
			//cv::waitKey(0);
		}
		for (int p = 0; p < logo.size(); p++) {
			//obliczyc bounding box
			centersLogo.push_back(imageCenter(logo[p]));
			//cv::imshow("image", logoElements[p]);
			//cv::waitKey(0);
		}
		//imshow("image", logoElements[0] + logoElements[1]+ logoElements[2]);
		//obliczyc odleglosci miedzy bboxami 
		//vector<float> dist;
		//float distances[centers.size][centers.size]; //46, 50 
		for (int k = 0; k < centersLogo.size(); k++) {
			for (int l = 0; l < centersM.size(); l++) {
				float LL = pow(distance(centersLogo[k], centersM[l]), 2) / moment(logo[k], 0, 0);
				cout << LL << endl;
				if (LL > 0 && LL < 2) {
					for (int m = 0; m < centersA.size(); m++) {
						float LL2 = pow(distance(centersLogo[k], centersA[m]), 2) / moment(logo[k], 0, 0);
						cout << LL2 << endl;
						
						if (LL2 > 0 && LL2 < 2) {
							cout << "wykryto" << endl;			
							Mat I = logo[k] + letterA[m] + letterM[l];
							Mat b = centerGeom(I, img_clean);
							namedWindow("image", WINDOW_NORMAL);
							imshow("image", img_clean);
							waitKey(0);
							


						}
					}
				}
			}
		}
	}
	//cv::imshow("segmented", img2);

	img2 = Lightness(img2, 200);
	//cv::namedWindow("light", WINDOW_NORMAL);
	//cv::imshow("light", img2);
	//cv::waitKey(0);

    return 0;
}