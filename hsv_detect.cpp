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


//����������HSV ��ɫ��Χ

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

	//HSV����
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

	int scale = max(img.rows / AverageHeight, img.cols / AverageWidth);//ͼƬ���ű���
	if (scale == 0 || scale == 1) { scale = 1; }
	resize(img, img, Size(img.cols / scale, img.rows / scale), 0, 0, INTER_LINEAR);



	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	  //HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������
	svm.load("SVM_HOG.xml");

	/*************************************************************************************************
	  ����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho;
	  ��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho��
	  ��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()����
	  �Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�
	  ***************************************************************************************************/
	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

	//��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i < supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j < DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i < supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i < DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

//	//�������Ӳ������ļ�
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

	//���������HSV����
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

	morphologyEx(imgThresholded, imgThresholded, MORPH_OPEN, element); //�ղ��� (����һЩ��ͨ��)
	
	//imshow("afterMorphologyEx", imgThresholded);

	//imshow("beforeMorphologyEx", imgThresholded);

	//morphologyEx(imgThresholded, imgThresholded, MORPH_CLOSE, element);

	imshow("afterMorphologyEx", imgThresholded);
	//�ԻҶ�ͼ�����˲�
	GaussianBlur(imgThresholded, imgThresholded, Size(3, 3), 0, 0);

	imshow("afterGaussian", imgThresholded);

	Canny(imgThresholded, cannyImage, 128, 255, 3);
	findContours(cannyImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	for (int j = 0; j < (int)contours.size(); j++) {
			drawContours(cannyImage, contours, j, Scalar(255), 1, 8);
	}
	cv::imshow("������ͼ��", cannyImage);




	if (!imgThresholded.empty())
		cv::imshow("�˲���ͼ��", imgThresholded);


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
