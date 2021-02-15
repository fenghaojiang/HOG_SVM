#include <iostream> 
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <string>
#include <fstream>
#include <ctype.h>
#include <string>
#include "my_svm.h"

using namespace std;
using namespace cv;

//测试Annotation数据标注是否准确，输入图片显示出标注数据所画的方框

#define CalculateTrainIOUList "TrainAnnotation.txt"  //
#define CalculateTestIOUList "TestAnnotation.txt" //

const string CalculateTrainIOUDir = "E://HOG_SVM//Project1//Project1//INRIAPerson//Train//pos//";
const string CalculateTestIOUDir = "E://HOG_SVM//Project1//Project1//INRIAPerson//Test//pos//";


int main()
{
	string name;

	cin >> name;
	cv::Mat img = cv::imread(CalculateTestIOUDir + name + ".png");
	
	Point tl, br;
	int x, y;
	cin >> x >> y;
	tl = Point(x, y);
	cin >> x >> y;
	br = Point(x, y);

	rectangle(img, tl, br, cv::Scalar(0, 255, 0), 3);
	

	imshow("result", img);
	cv::waitKey(0);
	return 0;
}
