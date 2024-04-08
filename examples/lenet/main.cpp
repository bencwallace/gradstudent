#include <algorithm>
#include <cmath>
#include <execution>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <thread>

#include "gradstudent/ops.h"
#include "gradstudent/utils.h"

const size_t img_dim = 28;
const size_t num_examples = 10000;

const size_t mnist_labels_header_size = 8;
const size_t mnist_images_header_size = 16;

gs::Tensor read_mnist_labels(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::in);
  file.ignore(mnist_labels_header_size);
  return gs::parse_numpy_data<unsigned char>(gs::array_t{num_examples}, file);
}

gs::Tensor read_mnist_images(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary | std::ios::in);
  file.ignore(mnist_images_header_size);
  return gs::parse_numpy_data<unsigned char>(
      gs::array_t{num_examples, img_dim, img_dim, 1}, file);
}

std::pair<gs::Tensor, gs::Tensor>
load_mnist(const std::filesystem::path &path) {
  auto labels_path = path / "t10k-labels-idx1-ubyte";
  auto images_path = path / "t10k-images-idx3-ubyte";

  auto labels = read_mnist_labels(labels_path);
  auto images = read_mnist_images(images_path);

  images = (1. / 255.) * images;
  images = 2 * (images - 0.5);

  return {labels, images};
}

std::map<std::string, gs::Tensor>
load_weights(const std::filesystem::path &path) {
  std::map<std::string, gs::Tensor> weights;

  // TODO: get the shapes right
  weights.insert(
      {"conv1.weight",
       gs::permute(gs::read_numpy(path / "conv1.weight.npy"), {0, 2, 3, 1})});
  weights.insert({"conv1.bias", gs::read_numpy(path / "conv1.bias.npy")});
  weights.insert(
      {"conv2.weight",
       gs::permute(gs::read_numpy(path / "conv2.weight.npy"), {0, 2, 3, 1})});
  weights.insert({"conv2.bias", gs::read_numpy(path / "conv2.bias.npy")});
  weights.insert({"fc1.weight", gs::read_numpy(path / "fc1.weight.npy")});
  weights.insert({"fc1.bias", gs::read_numpy(path / "fc1.bias.npy")});
  weights.insert({"fc2.weight", gs::read_numpy(path / "fc2.weight.npy")});
  weights.insert({"fc2.bias", gs::read_numpy(path / "fc2.bias.npy")});
  weights.insert({"fc3.weight", gs::read_numpy(path / "fc3.weight.npy")});
  weights.insert({"fc3.bias", gs::read_numpy(path / "fc3.bias.npy")});

  return weights;
}

class InferenceRunner {
public:
  InferenceRunner(const std::map<std::string, gs::Tensor> &weights)
      : weights_(weights) {}

  gs::Tensor conv(const gs::Tensor &x, size_t i) {
    const auto &w = weights_.at("conv" + std::to_string(i) + ".weight");
    const auto &b = weights_.at("conv" + std::to_string(i) + ".bias");
    const auto &x1 = gs::conv(x, w, 2);
    const auto &x2 = gs::permute(x1, {1, 2, 0});
    const auto &x3 = x2 + b;
    return x3;
  }

  gs::Tensor conv_block(const gs::Tensor &x, size_t i) {
    const auto &x1 = conv(x, i);
    const auto &x2 = gs::relu(x1);
    const auto &x3 = gs::permute(x2, {2, 0, 1});
    const auto &x4 = gs::maxPool(x3, {2, 2});
    const auto &x5 = gs::permute(x4, {1, 2, 0});
    return x5;
  }

  gs::Tensor fc(const gs::Tensor &x, size_t i) {
    const auto &w = weights_.at("fc" + std::to_string(i) + ".weight");
    const auto &b = weights_.at("fc" + std::to_string(i) + ".bias");
    const auto &x1 = gs::dot(w, x);
    const auto &x2 = x1 + b;
    return x2;
  }

  gs::Tensor infer(const gs::Tensor &input) {
    const auto &x0 = input;
    const auto &x1 = conv_block(x0, 1);
    const auto &x2 = conv_block(x1, 2);
    const auto &x3 = gs::flatten(gs::permute(x2, {2, 0, 1}));
    const auto &x4 = fc(x3, 1);
    const auto &x5 = fc(x4, 2);
    const auto &x6 = fc(x5, 3);
    const auto &x7 = gs::argmax(x6);
    return x7;
  }

  gs::Tensor run_inference(const gs::Tensor &input, size_t num_workers = 0) {
    num_workers =
        num_workers > 0 ? num_workers : std::thread::hardware_concurrency();
    num_workers = std::min(
        num_workers, static_cast<size_t>(std::thread::hardware_concurrency()));

    size_t n = input.shape()[0];
    gs::Tensor result(gs::array_t{n});

    size_t workload = std::ceil(static_cast<float>(n) / num_workers);
    std::vector<size_t> chunks(num_workers, 0);
    for (size_t i = 0; i < num_workers; ++i) {
      chunks.insert(chunks.begin(), i);
    }
    std::for_each(std::execution::par, std::begin(chunks), std::end(chunks),
                  [&](size_t i) {
                    auto in = gs::truncate(input, {i * workload},
                                           {std::min((i + 1) * workload, n)});
                    auto res = gs::truncate(result, {i * workload},
                                            {std::min((i + 1) * workload, n)});
                    for (size_t i = 0; i < in.shape()[0]; ++i) {
                      slice(res, {i}) =
                          static_cast<double>(infer(slice(in, {i})));
                    }
                  });

    return result;
  }

private:
  std::map<std::string, gs::Tensor> weights_;
};

double accuracy(const gs::Tensor &preds, const gs::Tensor &labels) {
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
  if (argc < 3) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::cerr << "Usage: " << argv[0] << " <weights_path>"
              << " <data_path> [<max_workers>]";
    return 1;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::string weights_path(argv[1]);
  auto weights = load_weights(weights_path);

  std::cout << "Loading data\n";
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::string data_path(argv[2]);
  auto data = load_mnist(data_path);

  std::cout << "Running inference\n";
  size_t num_workers = 0;
  if (argc > 3) {
    num_workers = std::stoi(argv[3]);
  }
  auto runner = InferenceRunner(weights);
  auto preds = runner.run_inference(data.second, num_workers);

  std::cout << "Computing accuracy\n";
  std::cout << "Accuracy: " << accuracy(preds, data.first) << '\n';

  return 0;
}
