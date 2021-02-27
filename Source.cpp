#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmimage/diregist.h"
#include <filesystem>
#pragma comment(lib, "ws2_32.lib")
#define _CRT_SECURE_NO_WARNINGS
#define OPENCV_TRAITS_ENABLE_DEPRECATED

using namespace std;
namespace stdfs = std::filesystem;


float aneurism_size(vector<cv::Point> cnt)
{
	cv::Point2f center;
	float radius;
	cv::minEnclosingCircle(cnt, center, radius);
	return 2 * radius;
}


cv::Mat prepare_image(string file)
{
	cv::Mat dst;

	DicomImage* image = new DicomImage(file.c_str());
	int nWidth = image->getWidth();
	int nHeight = image->getHeight();
	int depth = image->getDepth();
	/*cout << nWidth << ", " << nHeight << endl;
	cout << depth << endl;*/

	image->setWindow(100, 400);
	if (image != NULL)
	{
		if (image->getStatus() == EIS_Normal)
		{
			Uint8* pixelData = (Uint8*)(image->getOutputData(8));
			if (pixelData != NULL)
			{
				dst = cv::Mat(nHeight, nWidth, CV_8UC3, pixelData);
				/*imshow("image2", dst);
				cv::waitKey(0);
				system("pause");*/
			}
		}
		else
			cerr << "Error: cannot load DICOM image (" << DicomImage::getString(image->getStatus()) << ")" << endl;
	}
	return dst;
}


tuple<vector<vector<vector<cv::Point>>>, vector<vector<vector<cv::Point>>>> compute_correct_contours(stdfs::path path)
{
	vector<cv::RotatedRect> minRect;
	vector<string> files;
	const stdfs::directory_iterator end{};
	cv::Mat dst;
	cv::Mat img;
	cv::Mat mask1;
	vector<vector<cv::Point>> suspected_contours;
	vector<vector<cv::Point>>::iterator contour;
	vector<vector<cv::Point>> contours;
	cv::Mat mask2;
	cv::Mat mask3;
	cv::Mat mask;
	cv::Mat kernel;
	vector<vector<vector<cv::Point>>> all_contours;
	vector<vector<vector<cv::Point>>> correct_contours;
	float h, w;

	for (stdfs::directory_iterator iter{ path }; iter != end; ++iter)
		files.push_back(string(iter->path().string()));
	correct_contours.resize(files.size());
	all_contours.resize(files.size());
	for (int i = 0; i < files.size(); i++)
	{
		dst = prepare_image(files[i]);
		int nHeight = dst.rows;
		int nWidth = dst.cols;
		img = cv::Mat(nHeight, nWidth, CV_8UC3);
		img = dst;

		//create brightness mask
		mask1 = cv::Mat(nHeight, nWidth, CV_8UC3);
		cv::inRange(img, cv::Scalar(110, 110, 110), cv::Scalar(215, 215, 215), mask1);
		cv::findContours(mask1, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		//exclude outer bones and big area objects
		for (size_t i = 0; i < contours.size(); i++)
			if (/*cv::contourArea(contours[i]) > 80 and*/ cv::contourArea(contours[i]) < 1500 and cv::arcLength(contours[i], true) <= 1500)
				suspected_contours.push_back(contours[i]);
		mask2 = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
		cv::drawContours(mask2, suspected_contours, -1, (255), -1);
		suspected_contours.clear();

		kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
		cv::erode(mask2, mask2, kernel);
		cv::cvtColor(mask2, mask2, cv::COLOR_BGR2GRAY);

		//exclude dark areas
		mask = cv::Mat(nHeight, nWidth, CV_8UC3);
		cv::bitwise_and(mask1, mask2, mask);

		//exclude small inner bones
		mask3 = cv::Mat(nHeight, nWidth, CV_8UC3);
		cv::inRange(img, cv::Scalar(215, 215, 215), cv::Scalar(255, 255, 255), mask3);
		kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
		cv::dilate(mask3, mask3, kernel);
		mask = mask - mask3;

		cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		//exclude elongate objects
		minRect.resize(contours.size());
		for (size_t j = 0; j < contours.size(); j++)
		{
			all_contours[i].push_back(contours[j]);
			if (contours[j].size() >= 3 and aneurism_size(contours[j]) >= 7. * 1.5 /*and cv::contourArea(contours[i]) > 80*/ and cv::contourArea(contours[j]) < 1500)
			{
				minRect[j] = minAreaRect(contours[j]);
				if (minRect[j].size.width > minRect[j].size.height)
				{
					h = minRect[j].size.width;
					w = minRect[j].size.height;
				}
				else
				{
					w = minRect[j].size.width;
					h = minRect[j].size.height;
				};
				if (h / w <= 5)
					correct_contours[i].push_back(contours[j]);
			}
		}
		cout << float(i)/files.size()*100. << "%\n";
	}
	return { all_contours, correct_contours };
}


vector<vector<vector<cv::Point>>> volume(
	vector<vector<vector<cv::Point>>> aneurism_biggest_sections,
	vector<vector<vector<cv::Point>>> correct_contours,
	int i, int max_i, int w, int h)
{
	const int cnt_number = i;
	for (int j=0; j < correct_contours[i].size(); j++)
	{
		bool first = true;
		int j_ = j;
		int k = 0;
		do
		{
			if (k == correct_contours[i + 1].size())
				break;
			if (i < max_i)
			{
				vector<vector<cv::Point>> intersection_cnt;
				cv::Mat cnt_mask1 = cv::Mat::zeros(h, w, CV_8UC1);
				cv::Mat cnt_mask2 = cv::Mat::zeros(h, w, CV_8UC1);
				cv::Mat intersection = cv::Mat::zeros(h, w, CV_8UC1);
				cv::drawContours(cnt_mask1, vector<vector<cv::Point>>(1, correct_contours[i + 1][k]), -1, (255, 255, 255), -1);
				cv::imshow("mask1", cnt_mask1);
				cv::drawContours(cnt_mask2, vector<vector<cv::Point>>(1, correct_contours[i][j_]), -1, (255, 255, 255), -1);
				cv::imshow("mask2", cnt_mask2);
				cv::bitwise_and(cnt_mask1, cnt_mask2, intersection);
				cv::imshow("mask", intersection);
				cv::findContours(intersection, intersection_cnt, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

				if (!intersection_cnt.empty() and aneurism_size(correct_contours[i + 1][k]) >= aneurism_size(correct_contours[i][j_]) and i-1 > 0)
				{
					if (!first)
						aneurism_biggest_sections[i].pop_back();
					aneurism_biggest_sections[i+1].push_back(correct_contours[i+1][k]);
					i++;
					j_ = k;
					k = 0;
					first = false;
				}
				else
					k++;
			}
			else break;
		} while (true);
		i = cnt_number;
	}
	return aneurism_biggest_sections;
}


int main(int argc, char** argv)
{
	cv::Mat dst;
	vector<string> files;
	vector<vector<vector<cv::Point>>> all_contours;
	vector<vector<vector<cv::Point>>> correct_contours;
	vector<vector<vector<cv::Point>>> aneurism_biggest_sections;
	stdfs::path path = "ANONIM";
	int i = 0;

	tuple cnts = compute_correct_contours(path);
	all_contours = get<0>(cnts);
	correct_contours = get<1>(cnts);
	
	const stdfs::directory_iterator end{};
	for (stdfs::directory_iterator iter{ path }; iter != end; ++iter)
		files.push_back(string(iter->path().string()));
	
	aneurism_biggest_sections.resize(files.size());
	while (i <= files.size())
	{
		dst = prepare_image(files[i]);
		int nHeight = dst.rows;
		int nWidth = dst.cols;

		aneurism_biggest_sections = volume(aneurism_biggest_sections, correct_contours, i, files.size()-1, nWidth, nHeight);

		cv::drawContours(dst, correct_contours[i], -1, cv::Scalar(0, 255, 0), 1);
		for (int j = 0; j < aneurism_biggest_sections[i].size(); j++)
			if (aneurism_size(aneurism_biggest_sections[i][j]) >= 7. * 1.5)
				cv::drawContours(dst, vector<vector<cv::Point>>(1, aneurism_biggest_sections[i][j]), -1, cv::Scalar(0, 0, 255), 1);
		cv::imshow("suspected", dst);

		int k = cv::waitKey(0);
		if (k == 255)  //z
			i--;
		if (k == 247)  //x
			i++;
		if (k == 27)  //Esc
			break;
	}
	return 0;
}
