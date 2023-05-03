#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <string>
#include <unordered_map>
#include "data/tensor.hpp"
#include "../src/layer/details/softmax.hpp"
#include "runtime/runtime_ir.hpp"
#include "tick.hpp"

using namespace TinyInfer;

// 借助openCV预处理图像和生成输入tensor
TinyInfer::sftensor PreprocessImg(const cv::Mat& image) {
  assert(!image.empty());
  // 调整输入图片大小
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(224, 224));

  // 转换BGR通道到RGB
  cv::Mat rgb_image;
  cv::cvtColor(resized_image, rgb_image, cv::COLOR_BGR2RGB);
  
  // 改变像素点数值类型为float32
  rgb_image.convertTo(rgb_image, CV_32FC3);
  
  uint32_t input_c = 3;
  uint32_t input_h = 224;
  uint32_t input_w = 224;
  sftensor input = std::make_shared<ftensor>(input_c, input_h, input_w);

  // 拆开保存RGB通道
  std::vector<cv::Mat> split_channels;
  cv::split(rgb_image, split_channels);
  
  for (uint32_t c = 0; c < split_channels.size(); ++c) {
    const auto& split_channel = split_channels.at(c);
    assert(split_channel.total() == input_h * input_h);
    const cv::Mat& split_channel_t = split_channel.t();
    memcpy(input->slice(c).memptr(), split_channel_t.data, 
        sizeof(float) * split_channel.total());
  }

  assert(input->channels() == 3);

  // 归一化
  float mean_r = 0.485f, var_r = 0.229f;
  float mean_g = 0.456f, var_g = 0.224f;
  float mean_b = 0.406f, var_b = 0.225f;

  input->data() = input->data() / 255.f;
  input->slice(0) = (input->slice(0) - mean_r) / var_r;
  input->slice(1) = (input->slice(1) - mean_g) / var_g;
  input->slice(2) = (input->slice(2) - mean_b) / var_b;
  
  return input;
}

// python ref https://pytorch.org/hub/pytorch_vision_resnet/
int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("usage: ./resnet_demo [image path]\n");
    exit(-1);
  }

  const std::string& path = argv[1];

  // 读取imagenet类别索引和含义文件
  std::unordered_map<int, std::string> idx2cls;
  std::unordered_map<std::string, std::string> cls2syn;
  
  std::ifstream clsfile("../../tmp/resnet/imagenet_classes.txt");
  if (clsfile.is_open()) {
    int idx = 0;
    std::string cls;
    while (clsfile >> cls) {
      idx2cls[idx++] = cls;
    }
  }
  assert(!idx2cls.empty());

  std::ifstream synfile("../../tmp/resnet/imagenet_synsets.txt");
  if (synfile.is_open()) {
    std::string line;
    while (std::getline(synfile, line)) {
      size_t pos = line.find(' ');
      cls2syn[line.substr(0, pos)] = line.substr(pos + 1);
    }
  }
  assert(!cls2syn.empty());

  const uint32_t batch = 1;
  std::vector<sftensor> inputs(batch);
  for (uint32_t b = 0; b < batch; ++b) {
    // 借助openCV读取图片
    cv::Mat image = cv::imread(path);
    // 预处理图像
    sftensor input = PreprocessImg(image);
    inputs.at(b) = input;
  }

  // 构建计算图
  const std::string& param_path = "../../tmp/resnet/demo/resnet18_batch1.pnnx.param";
  const std::string& weight_path = "../../tmp/resnet/demo/resnet18_batch1.pnnx.bin";
  RuntimeGraph graph(param_path, weight_path);
  graph.Build("pnnx_input_0", "pnnx_output_0");

  // 推理
  TICK(resnet_infer)
  const auto outputs = graph.Forward(inputs, false);
  TOCK(resnet_infer)

  assert(outputs.size() == batch);
  
  // softmax
  std::vector<sftensor> outputs_softmax(batch);
  Softmax softmax(0);
  softmax.Forward(outputs, outputs_softmax);
  assert(outputs_softmax.size() == batch);

  for (int b = 0; b < batch; ++b) {
    const sftensor& output = outputs_softmax.at(b);
    assert(output->size() == 1 * 1000);
    // 找到类别概率最大的种类
    float max_prob = -1.f;
    int max_idx = -1;
    for (int i = 0; i < output->size(); ++i) {
      float prob = output->index(i);
      if (max_prob <= prob) {
        max_prob = prob;
        max_idx = i;
      }
    }
    std::string syn_str = cls2syn[idx2cls[max_idx]];
    std::cout << "Class: "<< syn_str << "\nProb.: " << max_prob << std::endl; 
  }

  return 0;
}