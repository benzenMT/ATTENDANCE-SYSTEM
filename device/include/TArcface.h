#ifndef TARCFACE_H
#define TARCFACE_H

#include <cmath>
#include <vector>
#include <string>
#include "net.h"
#include <opencv2/highgui.hpp>
//----------------------------------------------------------------------------------------
//
// Created by Xinghao Chen 2020/7/27
//
// Modified by Q-engineering 2020/12/28
//
//----------------------------------------------------------------------------------------
using namespace std;

class TArcFace {
private:
    ncnn::Net net;
    cv::Mat Zscore(const cv::Mat &fc);
    int feature_dim;
public:

    TArcFace(const char* bin_path, const char* param_path, int feature_dim);
    ~TArcFace(void);
    cv::Mat GetFeature(cv::Mat img);
};

#endif
