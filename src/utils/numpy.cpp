#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"
#include "utils.h"

namespace gradstudent {

struct numpy_header {
  std::string descr;
  bool fortran_order;
  array_t shape;
};

bool parse_substring(std::pair<std::string, size_t> &result,
                     const std::string &str) {
  size_t value_start = str.find('\'');
  if (value_start == std::string::npos) {
    return false;
  }
  size_t value_end = str.find('\'', value_start + 1);
  if (value_end == std::string::npos) {
    return false;
  }
  result.first = str.substr(value_start + 1, value_end - value_start - 1);
  result.second = value_end;
  return true;
}

bool parse_tuple(std::pair<std::string, size_t> &result,
                 const std::string &str) {
  size_t value_start = str.find('(');
  if (value_start == std::string::npos) {
    return false;
  }
  size_t value_end = str.find(')', value_start + 1);
  if (value_end == std::string::npos) {
    return false;
  }
  result.first = str.substr(value_start + 1, value_end - value_start - 1);
  result.second = value_end;
  return true;
}

bool parse_other(std::pair<std::string, size_t> &result,
                 const std::string &str) {
  size_t value_start = str.find_first_not_of(" \t\n\r\f\v");
  if (value_start == std::string::npos) {
    return false;
  }
  size_t value_end = str.find_first_of(" \t\n\r\f\v", value_start + 1);
  if (value_end == std::string::npos) {
    return false;
  }
  result.first = str.substr(value_start, value_end - value_start - 1);
  result.second = value_end;
  return true;
}

std::pair<std::string, size_t> parse_value(const std::string &str) {
  std::pair<std::string, size_t> result;
  char symbol = str[str.find_first_not_of(" \t\n\r\f\v")];
  bool success;
  if (symbol == '\'') {
    success = parse_substring(result, str);
  } else if (symbol == '(') {
    success = parse_tuple(result, str);
  } else {
    success = parse_other(result, str);
  }
  if (!success) {
    throw std::runtime_error("Cannot parse value: " + str);
  }
  return result;
}

std::unordered_map<std::string, std::string>
parse_dict(const std::string &str) {
  // parse a string representation of a python/json-like dict
  std::unordered_map<std::string, std::string> result;
  size_t start = 0;
  while (start < str.size()) {
    size_t key_start = str.find('\'', start);
    if (key_start == std::string::npos) {
      break;
    }
    size_t key_end = str.find("\':", key_start + 1);
    if (key_end == std::string::npos) {
      break;
    }
    std::string key = str.substr(key_start + 1, key_end - key_start - 1);

    key_end += 2; // length of: ':
    const auto &[value, value_end] = parse_value(str.substr(key_end + 1));
    result[key] = value;
    start = key_end + value_end + 2;
  }
  return result;
}

std::vector<size_t> parse_int_tuple(const std::string &str) {
  // parse a string representation of a python/json-like tuple of integers
  std::vector<size_t> result;
  size_t start = 0;
  while (start < str.size()) {
    size_t end = str.find(',', start);
    if (end == std::string::npos) {
      end = str.size();
    }
    result.push_back(std::stoul(str.substr(start, end - start)));
    start = end + 1;
  }
  return result;
}

numpy_header parse_numpy_header(std::istream &file) {
  numpy_header result;

  // expect magic character and numpy string
  std::string numpy(6, '\0');
  file.read(&numpy[0], 6);
  if (numpy != "\x93NUMPY") {
    throw std::runtime_error("Unsupported file format. Header: " + numpy);
  }

  // expect version 1.0
  unsigned char version[2];
  file.read(reinterpret_cast<char *>(version), 2);
  if (version[0] != '\x01' || version[1] != '\x00') {
    throw std::runtime_error("Unsupported file version: " +
                             std::string(reinterpret_cast<char *>(version), 2));
  }

  // parse header length
  unsigned char header_len_bytes[2];
  file.read(reinterpret_cast<char *>(&header_len_bytes), 2);
  unsigned short header_len = (header_len_bytes[1] << 8) | header_len_bytes[0];

  // parse header
  std::string header(header_len, '\0');
  file.read(&header[0], header_len);
  auto header_dict = parse_dict(header);

  // parse descr (dtype)
  auto it = header_dict.find("descr");
  if (it == header_dict.end()) {
    throw std::runtime_error("Cannot find 'descr' in header: " + header);
  }
  result.descr = it->second;

  // parse order
  it = header_dict.find("fortran_order");
  if (it == header_dict.end()) {
    throw std::runtime_error("Cannot find 'fortran_order' in header: " +
                             header);
  }
  result.fortran_order = it->second == "True";

  // parse shape
  it = header_dict.find("shape");
  if (it == header_dict.end()) {
    throw std::runtime_error("Cannot find 'shape' in header: " + header);
  }
  result.shape = parse_int_tuple(it->second);

  return result;
}

Tensor read_numpy(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::in);
  if (!file) {
    throw std::runtime_error("Cannot open file: " + filename);
  }

  auto header = parse_numpy_header(file);
  if (header.fortran_order) {
    throw std::runtime_error("Fortran order is not supported.");
  }
  if (header.descr == "<f8") {
    return parse_numpy_data<double>(header.shape, file);
  } else {
    throw std::runtime_error("Unsupported dtype: " + header.descr);
  }
}

} // namespace gradstudent
