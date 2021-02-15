#include <iostream> 
#include <opencv2/opencv.hpp>  
#include <string>
#include <fstream>
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

#define CalculateTestPosList "CalculateFinalTestPosList.txt"  //
#define CalculateTestNegList "CalculateFinalTestNegList.txt"  //
#define CalculateTrainPosList "CalculateFinalTrainPosList.txt"  //
#define CalculateTrainNegList "CalculateFinalTrainNegList.txt"  //


//E://HOG_SVM//TestAcc//Train//neg//

const string CalculateTestPosDir = "E://HOG_SVM//TestAcc//Test//pos//";
const string CalculateTestNegDir = "E://HOG_SVM//TestAcc//Test//neg//";
const string CalculateTrainPosDir = "E://HOG_SVM//TestAcc//Train//pos//";
const string CalculateTrainNegDir = "E://HOG_SVM//TestAcc//Train//neg//";
int tp = 0, fn = 0, fp = 0, tn = 0;
string Title = "Final_HOG_SVM: Result :";

int main(int argc, char** argv)
{
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

	string name;
	ifstream TestPos(CalculateTestPosList);
	ifstream TestNeg(CalculateTestNegList);
	ifstream TrainPos(CalculateTrainPosList);
	ifstream TrainNeg(CalculateTrainNegList);

	int count = 0;
	int PosCnt = 0;
	int NegCnt = 0;
	double totalTime = 0;

	string tag = "Test";

	
	while (getline(TrainPos, name)) {
		cout << "Processing Image Name :" << name << endl;
		cout << CalculateTestPosDir + name << endl;
		cv::Mat image = cv::imread(CalculateTrainPosDir + name);
		if (image.empty()) {
			std::cout << "read image failed" << std::endl;
		}

		std::vector<cv::Rect> regions;

		//��ʱ
		double t = (double)getTickCount();

		myHOG.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

		t = (double)getTickCount() - t;
		printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
		totalTime += t * 1000. / cv::getTickFrequency();
		count++;
		PosCnt++;
		if (regions.size() != 0) {
			tp++;
		}
		else if (regions.size() == 0) {
			fn++;
		}
	}

	while (getline(TrainNeg, name)) {
		cout << "Processing Image Name :" << name << endl;
		cv::Mat image = cv::imread(CalculateTrainNegDir + name);
		if (image.empty()) {
			std::cout << "read image failed" << std::endl;
		}
		std::vector<cv::Rect> regions;
		//��ʱ
		double t = (double)getTickCount();

		myHOG.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

		t = (double)getTickCount() - t;
		printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
		totalTime += t * 1000. / cv::getTickFrequency();

		//��һ��cv��winStride �ڶ�����Padding
		// ��ʾ
		count++;
		NegCnt++;
		if (regions.size() != 0) {
			fp++;
		}
		else if (regions.size() == 0) {
			tn++;
		}
	}

	/*
	TP����������������Ϊ���˵�������
	FN����������������Ϊ�����˵�������
	FP������������������Ϊ���˵�������
	TN������������������Ϊ�����˵�������
	*/

	cout << Title << endl;
	cout << tag << "��������������Ϊ:" << count << endl;

	cout << tag << "����������Ϊ:" << PosCnt << endl;
	cout << tag << "������Ƭʶ��������˵�TP:" << tp << endl;
	cout << tag << "������Ƭʶ�𲻳������˵�FN:" << fn << endl;

	cout << tag << "����������Ϊ:" << NegCnt << endl;
	cout << tag << "��������Ƭʶ��������˵�FP:" << fp << endl;
	cout << tag << "��������Ƭʶ�𲻳������˵�TN:" << tn << endl;

	cout << tag << "ƽ������ÿ֡ʱ��Ϊ:" << (double)totalTime / count << "ms" << endl;

	TestPos.close();
	TestNeg.close();
	TrainPos.close();
	TrainNeg.close();
	//cout << "Scale :" << scale << endl;
	cv::waitKey(0);
	return 0;
}
