#include "stdafx.h"
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/ml/ml.hpp>

#include<iostream>
#include<sstream>
#include<string>


#include "PossibleChar.h"
#include "Recognition.h"

const int MIN_CONTOUR_AREA = 750; //100

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;


class ContourWithData {
public:
	std::vector<cv::Point> ptContour;
	std::vector<std::vector<cv::Point> > ptContours;
	cv::Rect boundingRect;
	float fltArea;
	std::string path;

	ContourWithData::ContourWithData() {
		ContourWithData::path = "temp.png";
	}

	bool checkIfContourIsValid(PossibleChar &possibleChar) {
		if (fltArea < MIN_CONTOUR_AREA) return false;
		else {
			if (possibleChar.boundingRect.height*1.5 < possibleChar.boundingRect.width)
			{
				return false;
			}
			/*if (checkIfPossibleChar(possibleChar)) { //jei konturas gali but simbolis, pridet prie vektoriaus galimu simb.
			return true;
			}*/
		}
	}

	bool checkIfPossibleChar(PossibleChar &possibleChar) {
		if (possibleChar.boundingRect.area() > MIN_PIXEL_AREA &&
			possibleChar.boundingRect.width > MIN_PIXEL_WIDTH && possibleChar.boundingRect.height > MIN_PIXEL_HEIGHT &&
			MIN_ASPECT_RATIO < possibleChar.dblAspectRatio && possibleChar.dblAspectRatio < MAX_ASPECT_RATIO) {
			return true;
		}
		else {
			return false;
		}
	}

	//Rikiuoja is kaires i desine
	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
	}

	std::string init() {
		std::vector<ContourWithData> allContoursWithData;
		std::vector<ContourWithData> validContoursWithData;

		cv::Mat matClassificationInts;

		cv::FileStorage fsClassifications("classifications2.xml", cv::FileStorage::READ);

		if (fsClassifications.isOpened() == false) {
			std::cout << "Klaida, nepavyko atidaryti klasifikaciju failo, isjungiama\n\n";
			return "";
		}
		fsClassifications["classifications"] >> matClassificationInts;
		fsClassifications.release();


		cv::Mat matTrainingImagesAsFlattenedFloats;

		cv::FileStorage fsTrainingImages("images2.xml", cv::FileStorage::READ);

		if (fsTrainingImages.isOpened() == false) {
			std::cout << "Klaida, nepavyko atidaryti trainingo failo, isjungiama\n\n";
			return "";
		}

		fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;//surasom i training vektoriu          
		fsTrainingImages.release();

		cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());


		kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);

		//Pradedam ieskoti skaiciu

		cv::Mat matNumbers = cv::imread(path);
		matNumbers.convertTo(matNumbers, -1, 1.3, -50);
		cv::resize(matNumbers, matNumbers, cv::Size(), 0.5, 0.5);

		if (matNumbers.empty()) {
			std::cout << "Klaida, nepavyksta nuskaityti nuotraukos\n\n";
			return "";
		}

		cv::Mat matGrayscale;
		cv::Mat matBlurred;
		cv::Mat matThresh;
		cv::Mat matThreshCopy;
		cv::Mat matThreshCopy1; //open funkcijai
		cv::Mat matThreshCopy2; //close funkcijai
		cv::Mat matThreshCopy3;

		cv::cvtColor(matNumbers, matGrayscale, CV_BGR2GRAY); //i grayscale


		cv::GaussianBlur(matGrayscale, //blurras
			matBlurred,
			cv::Size(5, 5),
			0);


		cv::adaptiveThreshold(matBlurred, //thresholdas                     
			matThresh,
			255,
			cv::ADAPTIVE_THRESH_GAUSSIAN_C,
			cv::THRESH_BINARY_INV,
			11,
			2);


		matThreshCopy = matThresh.clone();
		matThreshCopy1 = matThresh.clone(); //open
		matThreshCopy2 = matThresh.clone(); //close

		cv::morphologyEx(matThreshCopy, //Sumazinam noise
			matThreshCopy1,
			cv::MORPH_CLOSE,
			cv::Mat(),
			cv::Point(-1, -1),
			2);

		//cv::imshow("bla", imgThreshCopy2);

		cv::morphologyEx(matThreshCopy1, //Sujungiam per tarpus
			matThreshCopy2,
			cv::MORPH_OPEN,
			cv::Mat(),
			cv::Point(-1, -1),
			1);

		cv::morphologyEx(matThreshCopy2, //Sujungiam per tarpus
			matThreshCopy3,
			cv::MORPH_CLOSE,
			cv::Mat(),
			cv::Point(-1, -1),
			4);


		std::vector<std::vector<cv::Point> > ptContours;  //Vektorius konturams
		std::vector<cv::Vec4i> v4iHierarchy;

		cv::findContours(matThreshCopy3, //Ieskom konturu           
			ptContours,
			v4iHierarchy,
			cv::RETR_EXTERNAL,
			cv::CHAIN_APPROX_SIMPLE);

		for (int i = 0; i < ptContours.size(); i++) {
			ContourWithData contourWithData;
			contourWithData.ptContour = ptContours[i];   //Jei konturas tinka pridedam prie contour with data
			contourWithData.boundingRect = cv::boundingRect(contourWithData.ptContour);
			contourWithData.fltArea = cv::contourArea(contourWithData.ptContour);
			allContoursWithData.push_back(contourWithData);
		}

		for (int i = 0; i < allContoursWithData.size(); i++) {
			PossibleChar possibleChar(ptContours[i]);
			if (allContoursWithData[i].checkIfContourIsValid(possibleChar)) {  //Patikrinam ar gali buti simbolis
				validContoursWithData.push_back(allContoursWithData[i]);       //Jei taip pridedam prie sekos
			}
		}
		//Rikiuojam
		std::sort(validContoursWithData.begin(), validContoursWithData.end(), ContourWithData::sortByBoundingRectXPosition);

		std::string strFinalString; //outputas

		for (int i = 0; i < validContoursWithData.size(); i++) {


			cv::rectangle(matNumbers,
				validContoursWithData[i].boundingRect,
				cv::Scalar(0, 255, 0),
				1);

			cv::Mat matROI = matThresh(validContoursWithData[i].boundingRect);

			cv::Mat matROIResized;
			cv::resize(matROI, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

			cv::Mat matROIFloat;
			matROIResized.convertTo(matROIFloat, CV_32FC1);

			cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

			cv::Mat matCurrentChar(0, 0, CV_32F);

			kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);

			float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

			strFinalString = strFinalString + char(int(fltCurrentChar));  //pridedam prie outputo
		}

		std::cout << "\n\n" << "Rasta skaiciu seka = " << strFinalString << "\n\n";

		cv::imshow("Skaiciai", matNumbers);

		cv::waitKey(0);

		return strFinalString;

	}

};

