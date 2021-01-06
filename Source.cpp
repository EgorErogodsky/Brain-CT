#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
#include "dcmtk/dcmimgle/dcmimage.h"
#include "dcmtk/dcmimage/diregist.h"
#pragma comment(lib, "ws2_32.lib")
#define _CRT_SECURE_NO_WARNINGS
#define OPENCV_TRAITS_ENABLE_DEPRECATED

using namespace std;

int main(int argc, char** argv)
{

    DicomImage* image = new DicomImage("marked/Anonymous.CT._.5384.128.2020.11.23.18.45.54.618.35824502.dcm");
    int nWidth = image->getWidth();
    int nHeight = image->getHeight();
    int depth = image->getDepth();
    cout << nWidth << ", " << nHeight << endl;
    cout << depth << endl;

    cv::Mat dst;
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

    cv::Mat img = cv::Mat(nHeight, nWidth, CV_8UC3);
    img = dst;

    //create brightness mask
    cv::Mat mask1 = cv::Mat(nHeight, nWidth, CV_8UC3);
    cv::inRange(img, cv::Scalar(110, 110, 110), cv::Scalar(215, 215, 215), mask1);
    vector<vector<cv::Point>> contours;
    cv::findContours(mask1, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    //exclude outer bones and big area objects
    vector<vector<cv::Point>> suspected_contours;
    vector<vector<cv::Point>>::iterator contour;
    for (contour = contours.begin(); contour != contours.end(); contour++)
        if (cv::contourArea(*contour) >80 and cv::contourArea(*contour) < 600 and cv::arcLength(*contour, true) <= 300)
            suspected_contours.push_back(*contour);
    cv::Mat mask2 = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);
    cv::drawContours(mask2, suspected_contours, -1, (255), -1);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(mask2, mask2, kernel);
    cv::cvtColor(mask2, mask2, cv::COLOR_BGR2GRAY);

    //exclude dark areas
    cv::Mat mask = cv::Mat(nHeight, nWidth, CV_8UC3);
    cv::bitwise_and(mask1, mask2, mask);

    //exclude small inner bones
    cv::Mat mask3 = cv::Mat(nHeight, nWidth, CV_8UC3);
    cv::inRange(img, cv::Scalar(215, 215, 215), cv::Scalar(255, 255, 255), mask3);
    kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::dilate(mask3, mask3, kernel);
    mask = mask - mask3;

    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //exclude elongate objects
    vector<vector<cv::Point>> correct_contours;
    for (contour = contours.begin(); contour != contours.end(); contour++)
    {
        if ((*contour).size() >= 3 and cv::contourArea(*contour) > 80 and cv::contourArea(*contour) < 600)
        {
            cv::Moments M = cv::moments(*contour);
            double cx = M.m10 / M.m00;
            double cy = M.m01 / M.m00;
            double w = 10000;
            double h = 0;
            for (vector<cv::Point>::iterator point = (*contour).begin(); point != (*contour).end(); point++)
            {
                double distance = sqrt(pow(((*point).x - cx), 2.0) + pow(((*point).y - cy), 2.0));
                if (distance < w)
                    w = distance;
                if (distance > h)
                    h = distance;
            }
            if (h / w <= 20)
                correct_contours.push_back(*contour);
        }
    }

    cv::drawContours(dst, correct_contours, -1, cv::Scalar(0, 255, 0), 1);
    cv::imshow("suspected", dst);
    cv::waitKey(0);

    delete image;
    return 0;
}
