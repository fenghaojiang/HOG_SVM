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
TP：将行人样本分类为行人的样本数
FN：将行人样本分类为非行人的样本数
FP：将非行人样本分类为行人的样本数
TN：将非行人样本分类为非行人的样本数
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

		//计时
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
		//计时
		double t = (double)getTickCount();

		myHOG.detectMultiScale(image, regions, 0, cv::Size(8, 8), cv::Size(32, 32), 1.05, 2);

		t = (double)getTickCount() - t;
		printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
		totalTime += t * 1000. / cv::getTickFrequency();

		//第一个cv是winStride 第二个是Padding
		// 显示
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
	TP：将行人样本分类为行人的样本数
	FN：将行人样本分类为非行人的样本数
	FP：将非行人样本分类为行人的样本数
	TN：将非行人样本分类为非行人的样本数
	*/

	cout << Title << endl;
	cout << tag << "正负样本集总数为:" << count << endl;

	cout << tag << "正样本数量为:" << PosCnt << endl;
	cout << tag << "行人照片识别出的行人的TP:" << tp << endl;
	cout << tag << "行人照片识别不出的行人的FN:" << fn << endl;

	cout << tag << "负样本数量为:" << NegCnt << endl;
	cout << tag << "非行人照片识别出的行人的FP:" << fp << endl;
	cout << tag << "非行人照片识别不出的行人的TN:" << tn << endl;

	cout << tag << "平均处理每帧时间为:" << (double)totalTime / count << "ms" << endl;

	TestPos.close();
	TestNeg.close();
	TrainPos.close();
	TrainNeg.close();
	//cout << "Scale :" << scale << endl;
	cv::waitKey(0);
	return 0;
}
