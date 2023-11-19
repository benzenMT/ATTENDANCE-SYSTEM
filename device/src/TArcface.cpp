#include "TArcface.h"
#include <iostream>

TArcFace::TArcFace(const char* bin_path, const char* param_path, int feature_dim)
{
    net.load_param(param_path);
    net.load_model(bin_path);
    this->feature_dim = feature_dim;
    //net.load_param("~/Face-Recognition-Raspberry-Pi-64-bits/models/mobilefacenet/mobilefacenet-opt.param");
    //net.load_model("~/Face-Recognition-Raspberry-Pi-64-bits/models/mobilefacenet/mobilefacenet-opt.bin");
}
//----------------------------------------------------------------------------------------
TArcFace::~TArcFace()
{
    this->net.clear();
}
//----------------------------------------------------------------------------------------
//    This is a normalize function before calculating the cosine distance. Experiment has proven it can destory the
//    original distribution in order to make two feature more distinguishable.
//    mean value is set to 0 and std is set to 1
cv::Mat TArcFace::Zscore(const cv::Mat &fc)
{
    cv::Mat mean, std;
    meanStdDev(fc, mean, std);
    return((fc - mean) / std);
}
//----------------------------------------------------------------------------------------
cv::Mat TArcFace::GetFeature(cv::Mat img)
{
    vector<float> feature;
    //cv to NCNN
    ncnn::Mat in = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    ncnn::Extractor ex = net.create_extractor();
    ex.input("data", in);
    ncnn::Mat out;
    ex.extract("fc1", out);
    feature.resize(this->feature_dim);
    for (int i = 0; i < this->feature_dim; i++) feature[i] = out[i];
    //normalize(feature);
    cv::Mat feature__=cv::Mat(feature,true);
    return Zscore(feature__);
}
//----------------------------------------------------------------------------------------
