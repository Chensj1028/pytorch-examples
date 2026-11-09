//
//  main.h
//  transfer-learning
//
//  Created by Kushashwa Ravi Shrimali on 15/08/19.
// 由 Kushashwa Ravi Shrimali 于 2019 年 8 月 15 日创建。
//

#ifndef main_h
#define main_h

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <dirent.h>
#include <torch/script.h>

// Function to return image read at location given as type torch::Tensor
// 读取指定位置的图片并返回 torch::Tensor。
// Resizes image to (224, 224, 3)
// 将图片缩放为 (224, 224, 3)。
torch::Tensor read_data(std::string location);

// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
// 将整数标签转换为 torch::Tensor；二分类用 0、1，n 分类用 0 到 n-1。
torch::Tensor read_label(int label);

// Function returns vector of tensors (images) read from the list of images in a folder
// 读取目录中的图片列表，返回图片张量向量。
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images);

// Function returns vector of tensors (labels) read from the list of labels
// 读取标签列表，返回标签张量向量。
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels);

// Function to load data from given folder(s) name(s) (folders_name)
// 从 folders_name 指定的目录加载数据。
// Returns pair of vectors of string (image locations) and int (respective labels)
// 返回图片路径字符串向量和对应整数标签向量组成的数据对。
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name);

// Function to train the network on train data
// 在训练数据上训练网络。
template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size);

// Function to test the network on test data
// 在测试数据上评估网络。
template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size);

// Custom Dataset class
// 自定义数据集类。
class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    /* data */
    // 数据。
    // Should be 2 tensors
    // 应包含两个张量。
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;
public:
    CustomDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
        states = process_images(list_images);
        labels = process_labels(list_labels);
        ds_size = states.size();
    };
    
    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        // 这里应返回 {torch::Tensor, torch::Tensor}。
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    };
    
    torch::optional<size_t> size() const override {
        return ds_size;
    };
};

#endif /* main_h */
