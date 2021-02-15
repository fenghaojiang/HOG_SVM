#include <iostream> 
#include <opencv2/opencv.hpp>  
#include <string>
#include <fstream>
#include <ctype.h>
#include <string>
#include "my_svm.h"

using namespace std;
using namespace cv;
const int AverageWidth = 631;
const int AverageHeight = 647;

/*
TP：将行人样本分类为行人的样本数
FN：将行人样本分类为非行人的样本数
FP：将非行人样本分类为行人的样本数
TN：将非行人样本分类为非行人的样本数
*/

#define CalculateTrainIOUList "TrainAnnotation.txt"  //
#define CalculateTestIOUList "TestAnnotation.txt" //

const string CalculateTrainIOUDir = "E://HOG_SVM//Project1//Project1//INRIAPerson//Train//pos//";
const string CalculateTestIOUDir = "E://HOG_SVM//Project1//Project1//INRIAPerson//Test//pos//";

Point getCenterPoint(Rect r) {
	Point cpt;
	cpt.x = r.x + cvRound(r.width / 2.0);
	cpt.y = r.y + cvRound(r.height / 2.0);
	return cpt;
}

Point getCenterPointFromString(String s) {
	Point center;
	string temp = "";
	int i = 0;
	for(; i < s.length(); i++) {
		if (s[i] != ',' && s[i] != '(') {
			temp += s[i];
		}
		else if (s[i] == ',') break;
	}
	center.x = stoi(temp);
	temp = "";
	for (; i < s.length(); i++) {
		if (s[i] != ',' && s[i] != ')') {
			temp += s[i];
		}
	}
	center.y = stoi(temp);
	cout << "Center:(" << center.x << "," << center.y << ")" << endl;
	return center;
}

Rect getRectFromString(String s) {
	Point leftTop, rightBottom;
	int i = 0;
	string temp = "";
	for (; i < s.length(); i++) {
		if (s[i] != ',' && s[i] != '(') {
			temp += s[i];
		}
		else if (s[i] == ',') break;
	}
	leftTop.x = stoi(temp);
	temp = "";
	for (; i < s.length(); i++) {
		if (s[i] != ',' && s[i] != ')') {
			temp += s[i];
		}
		else if (s[i] == ')') break;
	}
	leftTop.y = stoi(temp);
	cout << "LeftTop:(" << leftTop.x << "," << leftTop.y << ")" << endl;

	temp = "";
	for (; i < s.length(); i++) {
		if (s[i] != ',' && s[i] != '(' && s[i] != ')') {
			temp += s[i];
		}
		else if (s[i] == ',') break;
	}

	rightBottom.x = stoi(temp);
	temp = "";
	for (; i < s.length(); i++) {
		if (s[i] != ',' && s[i] != ')') {
			temp += s[i];
		}
	}
	rightBottom.y = stoi(temp);
	cout << "RightBottom:(" << rightBottom.x << "," << rightBottom.y << ")" << endl;

	int width = rightBottom.x - leftTop.x;
	int height = rightBottom.y - leftTop.y;

	return Rect(leftTop.x, leftTop.y, width, height);
}

int main()
{
	double totalTimeTrain = 0;
	double totalTimeTest = 0;

	double totalTrainIOU = 0.0;
	double totalTestIOU = 0.0;
	double AverageIOUofTest = 0.0;   //Test数据集中的平均IOU
	double AverageIOUofTrain = 0.0;  //Train数据集中的平均IOU

	double AverageDistanceofTrain = 0.0;    //Train平均欧式距离
	double AverageDistanceofTest = 0.0;     //Test平均欧式距离

	//HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器
	svm.load("SVM_HOG.xml");
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

	ifstream finTrain(CalculateTrainIOUList);
	ifstream finTest(CalculateTestIOUList);

	string name;
	int n;
	std::vector<cv::Rect> regions;//模型检测

	std::vector<Point> center;//标注的中心点
	std::vector<cv::Rect> annotations;//数据集标注

	int trainPersonNum = 0;
	int testPersonNum = 0;
	int trainCnt = 0; //有效框框数
	int testCnt = 0;
	int trainNum = 0; //样本数目
	int testNum = 0;
	double TestStandard = 0.5;

	while(getline(finTrain, name)) {
		cv::Mat image = cv::imread(CalculateTrainIOUDir + name);
		if (image.empty()) {
			if (isdigit(name[0])) {
				n = stoi(name);
				trainPersonNum += n;
			}
			else {
				if (name.length() <= 12) {
					center.push_back(getCenterPointFromString(name));
				}
				else {
					annotations.push_back(getRectFromString(name));
				}
			}
		} else {
			cout << "Processing Image Name :" << name << endl;
			
			regions.clear();
			center.clear();
			annotations.clear();

			n = 0;
			trainNum++;

			//计时
			double t = (double)getTickCount();
			myHOG.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
			//TODO 修正框框大小
			t = (double)getTickCount() - t;
			printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
		}

		// IOU高于70%才算是检测到了行人
		if(n != 0 && center.size() == n && annotations.size() == n) {
			for(int i = 0; i < regions.size(); i++) {
				for (int j = 0; j < annotations.size(); j++) {
					Rect r = regions[i];
					// the HOG detector returns slightly larger rectangles than the real objects.
					// so we slightly shrink the rectangles to get a nicer output.
					r.x += cvRound(r.width*0.1);
					r.width = cvRound(r.width*0.8);
					r.y += cvRound(r.height*0.07);
					r.height = cvRound(r.height*0.8);

					Rect r1 = r & annotations[j];
					Rect r2 = r | annotations[j];

					double iou = r1.area() * 1.0 / r2.area();
					if (iou >= TestStandard) {
						trainCnt++;
						totalTrainIOU += iou;
						cout << "trainIOU: " << iou << endl;
					}
				}
			}
		}
	}



	while (getline(finTest, name)) {
		cv::Mat image = cv::imread(CalculateTestIOUDir + name);
		if (image.empty()) {
			if (isdigit(name[0])) {
				n = stoi(name);
				testPersonNum += n;
			}
			else {
				if (name.length() <= 12) {
					center.push_back(getCenterPointFromString(name));
				}
				else {
					annotations.push_back(getRectFromString(name));
				}
			}
		}
		else {
			cout << "Processing Image Name :" << name << endl;

			regions.clear();
			center.clear();
			annotations.clear();

			n = 0;
			testNum++;

			//计时
			double t = (double)getTickCount();
			myHOG.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
			//TODO 修正框框大小
			t = (double)getTickCount() - t;
			printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
		}

		// IOU高于70%才算是检测到了行人
		if (n != 0 && center.size() == n && annotations.size() == n) {
			for (int i = 0; i < regions.size(); i++) {
				for (int j = 0; j < annotations.size(); j++) {
					Rect r = regions[i];
					// the HOG detector returns slightly larger rectangles than the real objects.
					// so we slightly shrink the rectangles to get a nicer output.
					r.x += cvRound(r.width*0.1);
					r.width = cvRound(r.width*0.8);
					r.y += cvRound(r.height*0.07);
					r.height = cvRound(r.height*0.8);

					Rect r1 = r & annotations[j];
					Rect r2 = r | annotations[j];

					double iou = r1.area() * 1.0 / r2.area();
					if (iou >= TestStandard) {
						testCnt++;
						totalTestIOU += iou;
						cout << "testIOU: " << iou << endl;
					}
				}
			}
		}
	}

	AverageIOUofTrain = totalTrainIOU / trainCnt;
	AverageIOUofTest = totalTestIOU / testCnt;

	cout << "Train 数量是:" << trainNum << "张图片" << endl;
	cout << "totalTrainIOU :" << totalTrainIOU << endl;
	cout << "共有行人" << trainPersonNum <<"人" << endl;
	cout << "检测出有效（大于" << TestStandard*100 <<"%）的框框个数: " << trainCnt << endl;
	cout << "Train平均IOU ：" << AverageIOUofTrain << endl;

	cout << endl;

	cout << "Test 数量是:" << testNum << "张图片" << endl;
	cout << "totalTestIOU :" << totalTestIOU << endl;
	cout << "共有行人" << testPersonNum << "人" << endl;
	cout << "检测出有效（大于" << TestStandard * 100 << "%）的框框个数: " << testCnt << endl;
	cout << "Test平均IOU ：" << AverageIOUofTest << endl;

	finTrain.close();
	finTest.close();
	
	cv::waitKey(0);
	return 0;
}
