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
#include <cstring>
#include <cstdlib>
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
bool is_newImg             = false;
bool is_timekeep             = false;

int detectCount = 0;

const char* cam_index       = "/dev/video1";
//----------------------------------------------------------------------------------------
// Some globals
//----------------------------------------------------------------------------------------
const int   RetinaWidth      = 320;
const int   RetinaHeight     = 240;

const char* detect_param_path = "/home/pi/work_space/Benzen_Project_230_AI/models/retina/mnet.25-opt.param";
const char* detect_bin_path = "/home/pi/work_space/Benzen_Project_230_AI/models/retina/mnet.25-opt.bin";

const char* extract_param_path = "/home/pi/work_space/Benzen_Project_230_AI/models/mobilefacenet/mobilefacenet-opt.param";
const char* extract_bin_path = "/home/pi/work_space/Benzen_Project_230_AI/models/mobilefacenet/mobilefacenet-opt.bin";

const char* face_cut_model = "/home/pi/work_space/Benzen_Project_230_AI/models/haarcascade_frontalface_alt.xml";

string pattern_jpg = "/home/pi/work_space/Benzen_Project_230_AI/img/*.jpg";
string path_img_raw = "/home/pi/work_space/Benzen_Project_230_AI/img_raw/";
string path_img_clean = "/home/pi/work_space/Benzen_Project_230_AI/img/";

const int   feature_dim      = 128;

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

inline float CosineDistance(const cv::Mat &v1, const cv::Mat &v2)
{
    // std::cout << v1 << "\n ===========  " << v2 << endl;
    double dot = v1.dot(v2);
    double denom_v1 = norm(v1);
    double denom_v2 = norm(v2);
    return dot / (denom_v1 * denom_v2);
}


void rescale_original(std::vector<FaceObject>& Faces)
{
    for (auto &face:Faces)
    {
        face.rect.x *= ScaleX;
        face.rect.y *= ScaleY;
        face.rect.width *= ScaleX;
        face.rect.height *= ScaleY;
        for (auto &lmk: face.landmark)
        {
            lmk.x *= ScaleX;
            lmk.y *= ScaleY;
        }
    }
}

void do_infer()
{
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

    reload:
    //loading the faces
	cv::glob(pattern_jpg, NameFaces);
    FaceCnt=NameFaces.size();
	if(FaceCnt==0)
    {
		std::cout << "No image files[jpg] in database" << endl;
	}
	else
    {
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

    if (!cap_truedeep.isOpened())
    {
        cerr << "ERROR: Unable to open the camera" << endl;
        return;
    }

    while(1){
        if(is_newImg == true)
        {
            is_newImg = false;
            goto reload;
        }
        cap_truedeep >> frame;
        if (frame.empty())
        {
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

        for(i=0;i<Faces.size();i++)
        {
            Faces[i].NameIndex = -2;    //-2 -> too tiny (may be negative to signal the drawing)
            Faces[i].Color     =  2;
            Faces[i].NameProb  = 0.0;
            Faces[i].LiveProb  = 0.0;
        }
        //run through the faces only when you got one face.
        //more faces (if large enough) are not a problem
        //in this app with an input image of 324x240, they become too tiny
        if(Faces.size() > 0)
        {
            for(i=0;i<Faces.size();i++)
            {
                if(Faces[i].FaceProb>MinFaceThreshold)
                {
                    //get centre aligned image and angle
                    cv::Mat aligned = Warp.Process(frame, Faces[i]);
                    // cv::imwrite("align.jpg", aligned);
                    Faces[i].Angle  = Warp.Angle;

                    // reject face that too skew
                    if (Warp.Angle > MaxAngle){
                        Faces[i].NameIndex = -1;    //a stranger
                        Faces[i].Color     =  1;
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
                            is_timekeep = true;
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
        else
        {
            free(person_name);
            person_name = NULL;
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "FPS: " << 1000.0/(duration.count()/1000) << endl;
    }
}

std::string file_name_extract(std::string input_string)
{
    std::string startPattern = "%2F";
    std::string endPattern = "?";
    std::size_t startPos = input_string.find(startPattern);
    std::size_t endPos = input_string.find(endPattern, startPos + startPattern.length());
    std::string output_file_name =  input_string.substr(startPos + startPattern.length(), 
                                                        endPos - (startPos + startPattern.length()));
    return output_file_name;
}

void face_cut(string name_img)
{
    cv::Mat image = cv::imread(string(path_img_raw + name_img).c_str()); // Đường dẫn đến ảnh của bạn

    if (image.empty()) {
        std::cout << "not load image raw" << std::endl;
        return -1;
    }

    cv::CascadeClassifier face_cascade;
    face_cascade.load(face_cut_model); // Đường dẫn đến tệp XML của bộ phân loại khuôn mặt

    if (face_cascade.empty())
    {
        std::cout << "not load model" << std::endl;
        return -1;
    }

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.1, 3, 0, cv::Size(30, 30));

    for (const cv::Rect& face : faces)
    {
        cv::Mat face_roi = image(face); // Cắt ảnh khuôn mặt

        // Thay đổi kích thước ảnh khuôn mặt về 112x112
        cv::resize(face_roi, face_roi, cv::Size(112, 112));
        // Lưu ảnh khuôn mặt vào file
        cv::imwrite(string(path_img_clean + name_img).c_str(), face_roi);
    }
}

// Callback function to write data to a file
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    size_t totalSize = size * nmemb;
    std::ofstream* file = static_cast<std::ofstream*>(userp);
    file->write(static_cast<char*>(contents), totalSize);
    return totalSize;
}

bool download_image(string url)
{
    CURL* curl = curl_easy_init();
    std::string output_file_name =  file_name_extract(url);

    if (curl)
    {
        string output_file_path = path_img_raw + output_file_name;
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

        if (res == CURLE_OK)
        {
            std::cout << "Download successful. Image saved as " << output_file_name << std::endl;
            return true;
        } 
        else 
        {
            std::cerr << "Download failed: " << curl_easy_strerror(res) << std::endl;
            return false;
        }
    }
}

class SubCallback : public virtual mqtt::callback {
    public:
        std::string message;
        std::string topics;
        bool status = false;

        void message_arrived(mqtt::const_message_ptr msg) override {
            std::cout << "Message Arrived! Topic: " << msg->get_topic() << ", Message: " << msg->to_string() << std::endl;
            message = msg->to_string();
            topics = msg->get_topic();
            status = true;
        }
        bool get_msg(std::string& topic, std::string& payload)
        {
            if(status)
            {
                topic = topics;
                payload = message;
                status = false;
                return true;
            }
            else
            {
                return false;
            }

        }
};

long long get_milis()
{
    auto currentTimePoint = std::chrono::system_clock::now();

    // Convert the time point to milliseconds
    auto currentTimeMs = std::chrono::time_point_cast<std::chrono::milliseconds>(currentTimePoint);

    // Extract the time since epoch in milliseconds
    auto timeSinceEpochMs = currentTimeMs.time_since_epoch();

    // Convert the time duration to milliseconds
    long long currentTimeMsValue = timeSinceEpochMs.count();
    return currentTimeMsValue;
}
void mqtt_prog()
{
    const std::string server_address = "tcp://test.mosquitto.org:1883"; // broker address MQTT
    const std::string client_id = "device_time_keeping";
    const std::vector<std::string> pub_topics = {"timekeep/new_employee_res", 
                                                "timekeep/time_keeping_res", 
                                                "timekeep/time_keeping"}; // Thay b?ng ch? d? (topic) MQTT b?n mu?n xu?t b?n d?n
    const std::vector<std::string> sub_topics = {"timekeep/new_employee", 
                                                "timekeep/time_keeping_res_host"}; // Thay b?ng ch? d? (topic) MQTT b?n mu?n xu?t b?n d?n
    while(true)
    {
        int status = system("ping -c 1 8.8.8.8");
        if(WEXITSTATUS(status) == 0)
        {
            cout<<"internet connected"<<endl;
            mqtt::async_client client(server_address, client_id);

            mqtt::connect_options connOpts;
            connOpts.set_keep_alive_interval(20);
            connOpts.set_clean_session(true);

            SubCallback callback;
            client.set_callback(callback);

            long long timer_sleep = 0;

            try
            {   
                
                client.connect(connOpts)->wait(); // K?t n?i d?n broker MQTT
                for(const std::string& sub_topic : sub_topics)
                {
                    client.subscribe(sub_topic, 1);
                }
                while(true)
                {
                    std::string topic;
                    std::string msg;
                    if(callback.get_msg(topic, msg))
                    {
                        if(strstr(sub_topics[0].c_str(), topic.c_str()) != NULL)
                        {
                            if(download_image(msg))
                            {
                                std::string output_file_name =  file_name_extract(msg); //make contens msg
                                face_cut(output_file_name);
                                is_newImg = true;
                                mqtt::message_ptr pubmsg = mqtt::make_message(pub_topics[0], output_file_name); //commit msg to buffer
                                client.publish(pubmsg)->wait(); // push the msg to MQTT
                            }
                        }
                        if(strstr(sub_topics[1].c_str(), topic.c_str()) != NULL)
                        {
                            is_timekeep = false;
                            std::string payload = "time_keeping_res: " + msg; //make contens msg
                            mqtt::message_ptr pubmsg = mqtt::make_message(pub_topics[1], payload); //commit msg to buffer
                            client.publish(pubmsg)->wait(); // push the msg to MQTT
                        }
                    }
                    if(is_timekeep == true && person_name != NULL)
                    {
                        if(get_milis() - timer_sleep > 3000)
                        {
                            std::string payload = "timekeep: " + string(person_name); //make contens msg
                            mqtt::message_ptr pubmsg = mqtt::make_message(pub_topics[2], payload); //commit msg to buffer
                            client.publish(pubmsg)->wait(); // push the msg to MQTT
                            timer_sleep = get_milis(); 
                        }
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }

                client.disconnect()->wait(); // disconnect after exited function
            }
            catch (const mqtt::exception& exc)
            {
                std::cerr << "Error: " << exc.what() << std::endl;
                return 1;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
//----------------------------------------------------------------------------------------
// main
//----------------------------------------------------------------------------------------

int main(int argc, char **argv)
{
    std::thread threadFaceReg(do_infer);
    std::thread threadMQTT(mqtt_prog);
    threadMQTT.join();
    threadFaceReg.join();
    return 0;
}
