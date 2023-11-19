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



