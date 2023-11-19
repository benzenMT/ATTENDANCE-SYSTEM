# Benzen-Project-230-AI-Device



## Getting started
### Install ncnn
> - $ sudo apt install build-essential git cmake libprotobuf-dev protobuf-compiler libvulkan-dev vulkan-utils libopencv-dev libzbar-dev libzbar0
> - $ git clone https://github.com/Tencent/ncnn.git
> - $ cd ncnn
> - $ git submodule update --init
> - $ mkdir build && cd build
> - $ cmake -DCMAKE_BUILD_TYPE=Release \
            -DNCNN_VULKAN=ON \
            -DNCNN_BUILD_EXAMPLES=ON \
            -DNCNN_SYSTEM_GLSLANG=ON \
            -DCMAKE_TOOLCHAIN_FILE=../toolchains/pi3.toolchain.cmake ..
> - make -j2 (make -j1 nếu bị treo)
> - make install
- Sau khi build xong sẽ có lib trong build/instal/lib/libncnn.a

####### Install lib
##########[install c++13 ARMv7]##################################
> - $sudo apt update
> - $sudo apt install software-properties-common
> - $sudo add-apt-repository ppa:ubuntu-toolchain-r/test
> - $sudo apt update
> - $sudo apt install gcc-13 g++-13 -y
> - $rm /usr/bin/gcc
> - $rm /usr/bin/g++
> - $ln -s /usr/bin/gcc-13 /usr/bin/gcc
> - $ln -s /usr/bin/g++-13 /usr/bin/g++
> - $######[check simlink gcc/g++]
> - $ls -la /usr/bin/ | grep gcc
> - $ls -la /usr/bin/ | grep g++
######check version gcc/g++
> - $gcc -v
> - $g++ -v

###################[install SSL & MQTT ARMv7]##################################
> - $<!-- sudo apt-get install build-essential gcc make cmake cmake-gui cmake-curses-gui git doxygen graphviz libssl-dev
> - $git clone https://github.com/eclipse/paho.mqtt.c.git
> - $cd paho.mqtt.c
> - $git checkout 1.4
> - $cmake -Bbuild -H. -DPAHO_WITH_SSL=ON
> - $sudo cmake --build build/ -j4 --target install
> - $sudo ldconfig
> - $cd ..
> - $git clone https://github.com/eclipse/paho.mqtt.cpp
> - $cd paho.mqtt.cpp
> - $cmake -Bbuild -H. -DPAHO_BUILD_DOCUMENTATION=TRUE -DPAHO_BUILD_SAMPLES=TRUE
> - $sudo cmake --build build/ -j4 --target install
> - $
> - $######[build code mqtt]
> - $g++ -o mqtt_example testMQTT.cpp -lpaho-mqttpp3 -lpaho-mqtt3a -lpaho-mqtt3as -->
### Clone repo
> - cd ~/
> - git clone https://github.com/Qengineering/Face-Recognition-Raspberry-Pi-64-bits
> - vi CMakeLists.txt (sửa /usr/local/lib/ncnn/libncnn.a thành đường dẫn tới libncnn.a ở phía trên, thêm /usr/lib/x86_64-linux-gnu/libvulkan.so vào set(EXTRA_LIBS...))
> - mkdir build && cd build
> - cmake ..
> - make -j4
- Sau khi build xong
> - cd ..
> - ./FaceRecognition
> - ./QrReader

Công dụng từng file
* [main_comp.cpp](src/main_comp.cpp) <br>
Luồng chính của hệ thống
* [TRetina.cpp](src/TRetina.cpp) <br>
Thực hiện việc detect face và 5 facial landmarks
* [TWarp.cpp](src/TWarp.cpp) <br>
Thực hiện việc align face dựa trên 5 facial landmarks
* [TArcface.cpp](src/TArcface.cpp) <br>
Trích xuất features của khuôn mặt



