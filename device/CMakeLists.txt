cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(FaceRecognition)

add_compile_options(-fpermissive)

find_package(OpenCV REQUIRED)

set(INCLUDE_PATH "/home/pi/work_space/ncnn")
set(INCLUDE_PATH_MQTT "/usr/local/lib")
set(INCLUDE_PATH_CURL "/lib/arm-linux-gnueabihf")

set(SOURCES src/main_comp.cpp src/TRetina.cpp src/TWarp.cpp src/TArcface.cpp)
set(SOURCES_CAP src/face_capture.cpp src/TRetina.cpp src/TWarp.cpp src/TLandmark.cpp)
set(SOURCES_EX src/face_extract.cpp src/TRetina.cpp src/TWarp.cpp src/TArcface.cpp)

set(EXTRA_LIBS ${OpenCV_LIBS} ${INCLUDE_PATH}/lib/libncnn.a ${INCLUDE_PATH_MQTT}/libpaho-mqttpp3.so ${INCLUDE_PATH_MQTT}/libpaho-mqtt3a.so ${INCLUDE_PATH_MQTT}/libpaho-mqtt3as.so ${INCLUDE_PATH_CURL}/libcurl.so.4 ${INCLUDE_PATH_CURL}/libcurl-gnutls.so.4)

set(EXECUTABLE_OUTPUT_PATH "../")

include_directories(include ${OpenCV_INCLUDE_DIRS} ${INCLUDE_PATH}/include/ncnn)

add_link_options(-fopenmp -s -ldl -lpthread)

add_executable(FaceRecognition ${SOURCES})

target_link_libraries(FaceRecognition ${EXTRA_LIBS} )