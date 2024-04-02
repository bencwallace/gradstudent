#include <filesystem>
#include <iostream>
#include <map>
#include <vector>

#include "gradstudent.h"

const size_t num_examples = 100;

gradstudent::Tensor read_mnist_labels(std::filesystem::path path) {
  std::ifstream file(path, std::ios::binary | std::ios::in);
  file.ignore(8);
  return gradstudent::parse_numpy_data<unsigned char>(
      gradstudent::array_t{num_examples}, file);
}

gradstudent::Tensor read_mnist_images(std::filesystem::path path) {
  std::ifstream file(path, std::ios::binary | std::ios::in);
  file.ignore(16);
  return gradstudent::parse_numpy_data<unsigned char>(
      gradstudent::array_t{num_examples, 28, 28, 1}, file);
}

std::pair<gradstudent::Tensor, gradstudent::Tensor>
load_mnist(std::filesystem::path path) {
  auto labels_path = path / "t10k-labels-idx1-ubyte";
  auto images_path = path / "t10k-images-idx3-ubyte";
  return {read_mnist_labels(labels_path), read_mnist_images(images_path)};
}

std::map<std::string, gradstudent::Tensor>
load_weights(std::filesystem::path path) {
  std::map<std::string, gradstudent::Tensor> weights;

  // TODO: get the shapes right
  weights.insert(
      {"conv1.weight",
       gradstudent::permute(gradstudent::read_numpy(path / "conv1.weight.npy"),
                            {0, 2, 3, 1})});
  weights.insert(
      {"conv1.bias", gradstudent::read_numpy(path / "conv1.bias.npy")});
  weights.insert(
      {"conv2.weight",
       gradstudent::permute(gradstudent::read_numpy(path / "conv2.weight.npy"),
                            {0, 2, 3, 1})});
  weights.insert(
      {"conv2.bias", gradstudent::read_numpy(path / "conv2.bias.npy")});
  weights.insert(
      {"fc1.weight", gradstudent::read_numpy(path / "fc1.weight.npy")});
  weights.insert({"fc1.bias", gradstudent::read_numpy(path / "fc1.bias.npy")});
  weights.insert(
      {"fc2.weight", gradstudent::read_numpy(path / "fc2.weight.npy")});
  weights.insert({"fc2.bias", gradstudent::read_numpy(path / "fc2.bias.npy")});
  weights.insert(
      {"fc3.weight", gradstudent::read_numpy(path / "fc3.weight.npy")});
  weights.insert({"fc3.bias", gradstudent::read_numpy(path / "fc3.bias.npy")});

  return weights;
}

class InferenceRunner {
public:
  InferenceRunner(const std::map<std::string, gradstudent::Tensor> &weights)
      : weights_(weights) {}

  gradstudent::Tensor conv(const gradstudent::Tensor &x, size_t i) {
    const auto &w = weights_.at("conv" + std::to_string(i) + ".weight");
    const auto &b = weights_.at("conv" + std::to_string(i) + ".bias");
    const auto &x1 = gradstudent::conv(x, w, 2);
    const auto &x2 = gradstudent::permute(x1, {1, 2, 0});
    const auto &x3 = x2 + b;
    return x3;
  }

  gradstudent::Tensor conv_block(const gradstudent::Tensor &x, size_t i) {
    const auto &x1 = conv(x, i);
    const auto &x2 = gradstudent::relu(x1);
    const auto &x3 = gradstudent::permute(x2, {2, 0, 1});
    const auto &x4 = gradstudent::maxPool(x3, {2, 2});
    const auto &x5 = gradstudent::permute(x4, {1, 2, 0});
    return x5;
  }

  gradstudent::Tensor fc(const gradstudent::Tensor &x, size_t i) {
    const auto &w = weights_.at("fc" + std::to_string(i) + ".weight");
    const auto &b = weights_.at("fc" + std::to_string(i) + ".bias");
    const auto &x1 = gradstudent::dot(w, x);
    const auto &x2 = x1 + b;
    return x2;
  }

  gradstudent::Tensor infer(const gradstudent::Tensor &input) {
    const auto &x0 = input;
    const auto &x1 = conv_block(x0, 1);
    const auto &x2 = conv_block(x1, 2);
    const auto &x3 = gradstudent::flatten(
        gradstudent::permute(x2, {2, 0, 1})); // TODO: permute before flattening
    const auto &x4 = fc(x3, 1);
    const auto &x5 = fc(x4, 2);
    const auto &x6 = fc(x5, 3);
    const auto &x7 = gradstudent::argmax(x6);
    return x7;
  }

  gradstudent::Tensor run_inference(const gradstudent::Tensor &input) {
    size_t n = input.shape()[0];
    gradstudent::Tensor result(gradstudent::array_t{n});
    for (size_t i = 0; i < n; ++i) {
      result[i] = static_cast<double>(infer(slice(input, {i})));
    }
    return result;
  }

private:
  std::map<std::string, gradstudent::Tensor> weights_;
};

double accuracy(const gradstudent::Tensor &preds,
                const gradstudent::Tensor &labels) {
  size_t n = preds.shape()[0];
  size_t correct = 0;
  for (size_t i = 0; i < n; ++i) {
    if (preds[i] == labels[i]) {
      ++correct;
    }
  }
  return static_cast<double>(correct) / n;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <weights_path>"
              << " <data_path>" << std::endl;
    return 1;
  }
  std::string weights_path(argv[1]);
  auto weights = load_weights(weights_path);

  std::cout << "Loading data" << std::endl;
  std::string data_path(argv[2]);
  auto data = load_mnist(data_path);

  std::cout << "Running inference" << std::endl;
  auto runner = InferenceRunner(weights);
  auto preds = runner.run_inference(data.second);

  std::cout << "Computing accuracy" << std::endl;
  std::cout << "Accuracy: " << accuracy(preds, data.first) << std::endl;

  return 0;
}
