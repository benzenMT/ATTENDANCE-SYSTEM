#include <iostream>
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include "TArcface.h"
#include "TRetina.h"
#include "TWarp.h"
#include <unistd.h>
#include <stdint.h> // for uint32_t
#include <sys/ioctl.h> // for ioctl
#include <fcntl.h> // for O_RDWR
#include <fstream> // for std::ofstream
#include <thread>
#include <string>
#include "mqtt/async_client.h"
#include </usr/include/arm-linux-gnueabihf/curl/curl.h>
// #include <wiringPi.h>
#include <time.h>
#include <pthread.h>

//----------------------------------------------------------------------------------------
// Adjustable Parameters
//----------------------------------------------------------------------------------------
const int   MinHeightFace    = 50;
const float MinFaceThreshold = 0.55;
const double MaxAngle        = 60.0;

char* person_name            = NULL;
bool is_unlocked             = false;

int detectCount = 0;

const char* cam_index       = "/dev/video1";
//----------------------------------------------------------------------------------------
// Some globals
//----------------------------------------------------------------------------------------
const int   RetinaWidth      = 320;
const int   RetinaHeight     = 240;

const char* detect_param_path = "./models/retina/mnet.25-opt.param";
const char* detect_bin_path = "./models/retina/mnet.25-opt.bin";

const char* extract_param_path = "./models/mobilefacenet/mobilefacenet-opt.param";
const char* extract_bin_path = "./models/mobilefacenet/mobilefacenet-opt.bin";

const char* face_cut_model = "./models/haarcascade_frontalface_alt.xml";

const int   feature_dim      = 128;

string pattern_jpg = "./img/*.jpg";

float ScaleX, ScaleY;
vector<cv::String> NameFaces;
//----------------------------------------------------------------------------------------
using namespace std;
using namespace cv;
using namespace std::chrono;
//----------------------------------------------------------------------------------------
//  Computing the cosine distance between input feature and ground truth feature
//----------------------------------------------------------------------------------------
cv::VideoCapture cap_truedeep(cam_index);

//////////////////////////////////
bool checkPerson=true;
//////////////////////////////////

inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2)
{
    // std::cout << v1 << "\n ===========  " << v2 << endl;
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}


void rescale_original(std::vector<FaceObject>& Faces){
    for (auto &face:Faces){
        face.rect.x *= ScaleX;
        face.rect.y *= ScaleY;
        face.rect.width *= ScaleX;
        face.rect.height *= ScaleY;
        for (auto &lmk: face.landmark){
            lmk.x *= ScaleX;
            lmk.y *= ScaleY;
        }
    }
}
void face_cut(string name_img)
{
    string path = "./img_raw/" + name_img + ".jpg";
    cv::Mat image = cv::imread(path.c_str()); // Đường dẫn đến ảnh của bạn

    if (image.empty()) {
        std::cout << "not load image raw" << std::endl;
        return -1;
    }

    cv::CascadeClassifier face_cascade;
    face_cascade.load(face_cut_model); // Đường dẫn đến tệp XML của bộ phân loại khuôn mặt

    if (face_cascade.empty()) {
        std::cout << "not load model" << std::endl;
        return -1;
    }

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.1, 3, 0, cv::Size(30, 30));

    for (const cv::Rect& face : faces) {
        cv::Mat face_roi = image(face); // Cắt ảnh khuôn mặt

        // Thay đổi kích thước ảnh khuôn mặt về 112x112
        cv::resize(face_roi, face_roi, cv::Size(112, 112));
        // Lưu ảnh khuôn mặt vào file
        string path_save = "./img/" + name_img + ".jpg";
        cv::imwrite(path_save.c_str(), face_roi);
    }
}
void do_infer(){
    int   n, count=1;
    size_t i;
    cv::Mat frame;
    cv::Mat result_cnn;
    cv::Mat faces;
    std::vector<FaceObject> Faces;
    vector<cv::Mat> fc1;

    size_t FaceCnt;
    // init the networks
    TWarp Warp;
    TArcFace ArcFace(extract_bin_path, extract_param_path, feature_dim);
    TRetina Rtn(RetinaWidth, RetinaHeight, detect_bin_path, detect_param_path, false);     //no Vulkan support on a RPi

    //loading the faces
	cv::glob(pattern_jpg, NameFaces);
    FaceCnt=NameFaces.size();
	if(FaceCnt==0) {
		std::cout << "No image files[jpg] in database" << endl;
	}
	else{
        std::cout << "Found "<< FaceCnt << " pictures in database." << endl;
        for(i=0; i<FaceCnt; i++){
            //convert to landmark vector and store into fc
            faces = cv::imread(NameFaces[i]);
            fc1.push_back(ArcFace.GetFeature(faces));
            //get a proper name
            string Str = NameFaces[i];
            n   = Str.rfind('/');
            Str = Str.erase(0,n+1);
            n   = Str.find('#');
            if(n>0) Str = Str.erase(n,Str.length()-1);                //remove # some numbers.jpg
            else    Str = Str.erase(Str.length()-4, Str.length()-1);  //remove .jpg
	        NameFaces[i]=Str;
            if(FaceCnt>1) printf("\rloading: %.2lf%% ",(i*100.0)/(FaceCnt-1));
        }
        std::cout << "" << endl;
        std::cout << "Loaded "<<FaceCnt<<" faces in total"<<endl;
    }

    // Start inference

    if (!cap_truedeep.isOpened()) {
        cerr << "ERROR: Unable to open the camera" << endl;
        return;
    }

    while(1){
        cap_truedeep >> frame;
        cv::imshow("RPi 64 OS - 1,95 GHz - 2 Mb RAM", frame);
        if (frame.empty()) {
            cerr << "End of movie" << endl;
            break;
        }
        
        ScaleX = ((float) frame.cols) / RetinaWidth;
        ScaleY = ((float) frame.rows) / RetinaHeight;

        // copy/resize image to result_cnn as input tensor
        cv::resize(frame, result_cnn, Size(RetinaWidth,RetinaHeight),INTER_LINEAR);

        auto start = high_resolution_clock::now();

        // detect face
        Rtn.detect_retinaface(result_cnn,Faces);
        rescale_original(Faces);

        for(i=0;i<Faces.size();i++){
            Faces[i].NameIndex = -2;    //-2 -> too tiny (may be negative to signal the drawing)
            Faces[i].Color     =  2;
            Faces[i].NameProb  = 0.0;
            Faces[i].LiveProb  = 0.0;
        }
        //run through the faces only when you got one face.
        //more faces (if large enough) are not a problem
        //in this app with an input image of 324x240, they become too tiny
        if(Faces.size() > 0){
            for(i=0;i<Faces.size();i++){
                if(Faces[i].FaceProb>MinFaceThreshold){
                    //get centre aligned image and angle
                    cv::Mat aligned = Warp.Process(frame, Faces[i]);
                    // cv::imwrite("align.jpg", aligned);
                    Faces[i].Angle  = Warp.Angle;

                    // reject face that too skew
                    if (Warp.Angle > MaxAngle){
                        Faces[i].NameIndex = -1;    //a stranger
                        Faces[i].Color     =  1;
                        is_unlocked = false;
                        free(person_name);
                        person_name = NULL;
                        continue;
                    }

                    //features of camera image
                    cv::Mat fc2 = ArcFace.GetFeature(aligned);

                    //reset indicators
                    Faces[i].NameIndex = -1;    //a stranger
                    //the similarity score
                    if(FaceCnt > 0){
                        vector<double> score_;
                        for(size_t c=0;c<FaceCnt;c++) score_.push_back(CosineDistance(fc1[c], fc2));
                        int Pmax = max_element(score_.begin(),score_.end()) - score_.begin();
                        Faces[i].NameIndex = Pmax;
                        Faces[i].NameProb  = score_[Pmax];
                        score_.clear();
                        if(Faces[i].NameProb >= MinFaceThreshold){
                            person_name = strdup(NameFaces[Faces[i].NameIndex].c_str());
                            std::cout << "reg name: " <<person_name<< endl;
                        }
                        else{
                            detectCount += 1;
                            // std::cout << detectCount << endl;
                            Faces[i].NameIndex = -1;    //a stranger
                            free(person_name);
                            person_name = NULL;
                        }
                        
                    }
                }
            }
        }
        else{
            free(person_name);
            person_name = NULL;
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "FPS: " << 1000.0/(duration.count()/1000) << endl;
    }
}
// Callback function to write data to a file
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), totalSize);
    return totalSize;
}

void download_image(string url, string output_file_name)
{
    CURL* curl = curl_easy_init();

    if (curl) {
        string output_file_path = "./img_raw/" + output_file_name + ".jpg";
        // Open a file for writing the downloaded image
        std::ofstream outputFile(output_file_path.c_str(), std::ios::binary);

        // Set the URL to download
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());

        // Set the write callback to save the downloaded content to the file
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &outputFile);

        // Perform the download
        CURLcode res = curl_easy_perform(curl);

        // Clean up
        outputFile.close();
        curl_easy_cleanup(curl);

        if (res == CURLE_OK) {
            std::cout << "Download successful. Image saved as " << output_file_name << std::endl;
        } else {
            std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
        }
    }
}
void mqtt_prog(){
    const std::string server_address = "tcp://broker.example.com:1883"; // Thay b?ng d?a ch? broker MQTT c?a b?n
    const std::string client_id = "paho_cpp_publish_example";
    const std::string topic = "test/topic"; // Thay b?ng ch? d? (topic) MQTT b?n mu?n xu?t b?n d?n

    mqtt::async_client client(server_address, client_id);

    mqtt::connect_options connOpts;
    connOpts.set_keep_alive_interval(20);
    connOpts.set_clean_session(true);

    try {
        client.connect(connOpts)->wait(); // K?t n?i d?n broker MQTT

        std::string payload = "Hello, MQTT!"; // N?i dung c?a th�ng di?p c?n xu?t b?n

        mqtt::message_ptr pubmsg = mqtt::make_message(topic, payload);
        client.publish(pubmsg)->wait(); // Xu?t b?n th�ng di?p d?n ch? d? MQTT

        client.disconnect()->wait(); // Ng?t k?t n?i sau khi ho�n th�nh xu?t b?n

        std::cout << "Message published successfully." << std::endl;
    } catch (const mqtt::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
}
//----------------------------------------------------------------------------------------
// main
//----------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    std::thread threadMQTT(mqtt_prog);
    std::thread threadFaceReg(do_infer);
    threadMQTT.join();
    threadFaceReg.join();
    return 0;
}
