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
TP����������������Ϊ���˵�������
FN����������������Ϊ�����˵�������
FP������������������Ϊ���˵�������
TN������������������Ϊ�����˵�������
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
	double AverageIOUofTest = 0.0;   //Test���ݼ��е�ƽ��IOU
	double AverageIOUofTrain = 0.0;  //Train���ݼ��е�ƽ��IOU

	double AverageDistanceofTrain = 0.0;    //Trainƽ��ŷʽ����
	double AverageDistanceofTest = 0.0;     //Testƽ��ŷʽ����

	//HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������
	svm.load("SVM_HOG.xml");
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

	ifstream finTrain(CalculateTrainIOUList);
	ifstream finTest(CalculateTestIOUList);

	string name;
	int n;
	std::vector<cv::Rect> regions;//ģ�ͼ��

	std::vector<Point> center;//��ע�����ĵ�
	std::vector<cv::Rect> annotations;//���ݼ���ע

	int trainPersonNum = 0;
	int testPersonNum = 0;
	int trainCnt = 0; //��Ч�����
	int testCnt = 0;
	int trainNum = 0; //������Ŀ
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

			//��ʱ
			double t = (double)getTickCount();
			myHOG.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
			//TODO ��������С
			t = (double)getTickCount() - t;
			printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
		}

		// IOU����70%�����Ǽ�⵽������
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

			//��ʱ
			double t = (double)getTickCount();
			myHOG.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);
			//TODO ��������С
			t = (double)getTickCount() - t;
			printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
		}

		// IOU����70%�����Ǽ�⵽������
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

	cout << "Train ������:" << trainNum << "��ͼƬ" << endl;
	cout << "totalTrainIOU :" << totalTrainIOU << endl;
	cout << "��������" << trainPersonNum <<"��" << endl;
	cout << "������Ч������" << TestStandard*100 <<"%���Ŀ�����: " << trainCnt << endl;
	cout << "Trainƽ��IOU ��" << AverageIOUofTrain << endl;

	cout << endl;

	cout << "Test ������:" << testNum << "��ͼƬ" << endl;
	cout << "totalTestIOU :" << totalTestIOU << endl;
	cout << "��������" << testPersonNum << "��" << endl;
	cout << "������Ч������" << TestStandard * 100 << "%���Ŀ�����: " << testCnt << endl;
	cout << "Testƽ��IOU ��" << AverageIOUofTest << endl;

	finTrain.close();
	finTest.close();
	
	cv::waitKey(0);
	return 0;
}
