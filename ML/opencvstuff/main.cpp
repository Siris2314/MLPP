#include <iostream>
#include <memory>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <torch/script.h>
#include <torch/torch.h>

#define HEIGHT 720
#define WIDTH 1280
#define IMG_SIZE 512


cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model);
torch::jit::Module load_model(std::string model_name);

int main(){
    torch::jit::script::Module module;
    cv::VideoCapture cap;
    cv::Mat frame;
    cap.open("/Users/arihanttripathi/Documents/C++ML/ML/opencvextras/cars3.mp4");

    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
    }

    std::cout << "Press space to stop the video" << std::endl;

    try{
        module = load_model("/Users/arihanttripathi/Documents/C++ML/ML/opencvextras/quantized_lanesNet.pt");
    }catch(const c10::Error &e){
        std::cerr << "Error loading the model\n";
    }

    for(;;){
        cap.read(frame);
        if(frame.empty()){
            std::cerr << "Error loading the frame\n";
        }

        frame = frame_prediction(frame, module);
        cv::imshow("video", frame);

        if(cv::waitKey(1) >= 27){
            break;
        }
    }
}


torch::jit::Module load_model(std::string model_name){
    torch::jit::Module module;
    try{
        module = torch::jit::load(model_name);
        module.eval();
        std::cout << "Model loaded successfully\n";
    }catch(const c10::Error &e){
        std::cerr << e.what();
    }
    return module;
}

cv::Mat frame_prediction(cv::Mat frame, torch::jit::Module model){
    double lr = 0.4;
    double beta = (1-lr);

    cv::Mat frame_copy, dist; 
    std::vector<torch::jit::IValue> inputs;
    std::vector<double> mean = {0.406, 0.456, 0.485};
    std::vector<double> std = {0.225, 0.224, 0.229};
    cv::resize(frame, frame, cv::Size(IMG_SIZE, IMG_SIZE));
    frame_copy = frame;
    frame.convertTo(frame, CV_32FC3, 1.0f/255.0f);
    torch::Tensor tensor_image = torch::from_blob(frame.data, {1, IMG_SIZE, IMG_SIZE, 3});
    tensor_image = tensor_image.permute({0, 3, 1, 2});
    tensor_image = torch::data::transforms::Normalize<>(mean,std)(tensor_image);
    inputs.push_back(tensor_image);

    auto prediction = model.forward(inputs).toTensor().detach().to(torch::kCPU);
    prediction = prediction.mul(100).clamp(0.255).to(torch::kU8);

    cv::Mat output(cv::Size(IMG_SIZE, IMG_SIZE), CV_8UC1, prediction.data_ptr());
    cv:cvtColor(output, output, cv::COLOR_GRAY2RGB);
    cv::applyColorMap(output, output, cv::COLORMAP_TWILIGHT_SHIFTED);
    cv::addWeighted(frame_copy, lr, output, beta, 0.0, dist);
    cv::resize(dist, dist, cv::Size(WIDTH, HEIGHT));
    return dist;


}