#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include "my_svm.h"

using namespace std;
using namespace cv;

const int AverageWidth = 631;
const int AverageHeight = 647;


//定义搜索的HSV 颜色范围

/*
//person_273
int LowH = 0;
int HighH = 180;
int LowS = 0;
int HighS = 255;
int LowV = 0;
int HighV = 255;
int R = 166;
int G = 252;
int B = 255;
*/

//crop001836
//int LowH = 0;
//int HighH = 180;
//int LowS = 0;
//int HighS = 255;
//int LowV = 0;
//int HighV = 255;
int R = 105;
int G = 123;
int B = 163;


//crop001638
//with RGBtoHSV();
//int R = 209;
//int G = 57;
//int B = 69;


//crop001504
//int R = 66;
//int G = 88;
//int B = 101;


/*//crop001501 without RGBtoHSV();
int LowH = 156;
int HighH = 180;
int LowS = 43;
int HighS = 255;
int LowV = 46;
int HighV = 255;
int R = 185;
int G = 35;
int B = 46;
//*/


//crop001222
//int LowH = 100;
//int HighH = 124;
//int LowS = 43;
//int HighS = 255;
//int LowV = 46;
//int HighV = 255;

//crop001668
//int LowH = 90;
//int HighH = 180;
//int LowS = 0;
//int HighS = 43;
//int LowV = 46;
//int HighV = 220;

//person_and_bike_147
int LowH = 90;
int HighH = 93;
int LowS = 43;
int HighS = 255;
int LowV = 46;
int HighV = 255;




void RGBtoHSV() {
	int max;
	if (R > G && R > B) {
		max = R;
	}
	else if (G > R && G > B) {
		max = G;
	}
	else{
		max = B;
	}
	
	int min;
	if (R < G && R < B) {
		min = R;
	}
	else if (G < R && G < B) {
		min = G;
	}
	else {
		min = B;
	}

	//HSV计算
	int V = max;

	LowV = max - 20;
	HighV = max + 20;
	if(LowV < 0) LowV = 0;
	if (HighV > 255) HighV = 255;

	double S = ((double)(max - min) / max) * 255.0;
	LowS =  int(S - 20);
	HighS = int(S + 20);
	if(LowS < 0) LowS = 0;
	if (HighS > 255) HighS = 255;

	int H;
	if (max == min) {
		H = 0;
	} else if(R == max && G >= B) {
		H = ((double)(G - B) / (max - min)) * 60;
	} else if(R == max && G < B) {
		H = ((double)(G - B) / (max - min)) * 60 + 360;
	} else if (G == max) {
		H = ((double)(B - R) / (max - min)) * 60 + 120;
	} else if (max == B) {
		H = ((double)(R - G) / (max - min)) * 60 + 240;
	}
	H = H / 2;

	LowH = H - 20;
	HighH = H + 20;
	if (LowH < 0) LowH = 0;
	if (HighS > 180) HighS = 180;

	cout << "H : " << H << endl;
	cout << "S : " << S << endl;
	cout << "V : " << V << endl;

}

int main()
{
	//RGBtoHSV();
	Mat img;
	string fileName;
	cin >> fileName;

	const string CalculateTrainIOUDir = "E://HOG_SVM//Project1//Project1//INRIAPerson//Train//pos//";
	const string CalculateTestIOUDir = "E://HOG_SVM//Project1//Project1//INRIAPerson//Test//pos//";

	img = imread(CalculateTrainIOUDir + fileName + ".png");

	int scale = max(img.rows / AverageHeight, img.cols / AverageWidth);//图片缩放比例
	if (scale == 0 || scale == 1) { scale = 1; }
	resize(img, img, Size(img.cols / scale, img.rows / scale), 0, 0, INTER_LINEAR);



	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	  //HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器
	svm.load("SVM_HOG.xml");

	/*************************************************************************************************
	  线性SVM训练完成后得到的XML文件里面，有一个数组，叫做support vector，还有一个数组，叫做alpha,有一个浮点数，叫做rho;
	  将alpha矩阵同support vector相乘，注意，alpha*supportVector,将得到一个列向量。之后，再该列向量的最后添加一个元素rho。
	  如此，变得到了一个分类器，利用该分类器，直接替换opencv中行人检测默认的那个分类器（cv::HOGDescriptor::setSVMDetector()），
	  就可以利用你的训练样本训练出来的分类器进行行人检测了。
	  ***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

	//将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i < supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j < DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i < supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

//	//保存检测子参数到文件
//	ofstream fout("HOGDetectorForOpenCV.txt");
//	for(int i=0; i<myDetector.size(); i++)
//	{
//		fout<<myDetector[i]<<endl;
//	}

	

	Mat origin = img;
	vector<Rect> found, found_filtered;
	
	double t = (double)getTickCount();
	// run the detector with default parameters. to get a higher hit-rate
	// (and more false alarms, respectively), decrease the hitThreshold and
	// groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
	myHOG.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	
	t = (double)getTickCount() - t;
	printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
	size_t i, j;

	Mat imgHSV;
	vector<Mat> hsvSplit;
	Mat imgThresholded;
	Mat element;
	Mat cannyImage;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	//在这里加入HSV代码
	cvtColor(img, imgHSV, COLOR_BGR2HSV);
	
	//imshow("before EqualizeHist",imgHSV);

	//split(imgHSV, hsvSplit);
	//equalizeHist(hsvSplit[2], hsvSplit[2]);
	//merge(hsvSplit, imgHSV);
	//imshow("after EqualizeHist", imgHSV);

	//LowH = 100;
	//HighH = 124;
	//LowS = 0;
	//HighS = 255;
	//LowV = 0;
	//HighV = 255;
	inRange(imgHSV, Scalar(LowH, LowS, LowV), Scalar(HighH, HighS, HighV), imgThresholded);
	element = getStructuringElement(MORPH_RECT, Size(5, 5));

	imshow("afterHSV", imgThresholded);

	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element); //闭操作 (连接一些连通域)
	
	//imshow("afterMorphologyEx", imgThresholded);

	//imshow("beforeMorphologyEx", imgThresholded);

	//morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

	imshow("afterMorphologyEx", imgThresholded);
	//对灰度图进行滤波
	GaussianBlur(imgThresholded, imgThresholded, Size(3, 3), 0, 0);

	imshow("afterGaussian", imgThresholded);

	Canny(imgThresholded, cannyImage, 128, 255, 3);
	findContours(cannyImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int j = 0; j < (int)contours.size(); j++) {
			drawContours(cannyImage, contours, j, Scalar(255), 1, 8);
	}
	cv::imshow("处理后的图像", cannyImage);




	if (!imgThresholded.empty())
		cv::imshow("滤波后图像", imgThresholded);


	for (i = 0; i < found.size(); i++)
	{
		Rect r = found[i];
		for (j = 0; j < found.size(); j++)
			if (j != i && (r & found[j]) == r)
				break;
		if (j == found.size())
			found_filtered.push_back(r);
	}

	for (i = 0; i < found_filtered.size(); i++)
	{
		Rect r = found_filtered[i];
		// the HOG detector returns slightly larger rectangles than the real objects.
		// so we slightly shrink the rectangles to get a nicer output.
		r.x += cvRound(r.width*0.1);
		r.width = cvRound(r.width*0.8);
		r.y += cvRound(r.height*0.07);
		r.height = cvRound(r.height*0.8);
		
		
		for (int j = 0; j < contours.size(); j++) {
			vector<Point> p = contours[j];
			for (int k = 0; k < p.size(); k++) {
				if (r.contains(p[k])) {
					rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);
					break;
				}
			}
		}
		//HSV
	}

	cv::imshow("special people detector", img);
	cv::waitKey(0);

	return 0;
}
