#ifndef TRETINA_H
#define TRETINA_H
#include <opencv2/highgui.hpp>
#include <stdio.h>
#include <vector>
#include "net.h"

struct FaceObject
{
    cv::Rect_<float> rect;
    cv::Point2f landmark[5];
    int NameIndex {-1};
    float FaceProb {0.0};
    double NameProb {0.0};
    double LiveProb {0.0};
    double Angle {0.0};
    int Color {0};      //background color of label on screen
};
//----------------------------------------------------------------------------------------
class TRetina
{
private:
    ncnn::Net retinaface;
    int img_w;
    int img_h;
protected:
public:
    TRetina(int Width, int Height, const char* bin_path, 
                const char* param_path, bool UseVulkan);
    virtual ~TRetina();

    int detect_retinaface(const cv::Mat& bgr,std::vector<FaceObject> &Faces);
    void draw_faceobjects(const cv::Mat& bgr);
};

#endif // TRETINA_H


