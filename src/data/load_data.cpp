#include <string>
#include <fstream>
#include <armadillo>
#include <utility>
#include <glog/logging.h>
#include "data/load_data.hpp"

namespace TinyInfer {

arma::fmat CSVDataLoader::LoadData(const std::string& file_path, const char split_char) {
  arma::fmat data;
  if (file_path.empty()) {
    LOG(ERROR) << "CSV file path is empty: " << file_path;
    return data;
  }

  std::ifstream in(file_path);
  if (!in.is_open() || !in.good()) {
    LOG(ERROR) << "File open failed: " << file_path;
    return data;
  }

  std::string line_str;
  std::stringstream line_stream;

  const auto& [rows, cols] = CSVDataLoader::GetMatrixSize(in, split_char);
  data.zeros(rows, cols);  // 初始化全0的fmat，行列数为(rows, cols)

  size_t row = 0;
  while (in.good()) {
    // 读取一行到line_str
    std::getline(in, line_str);
    if (line_str.empty()) {
      break;
    }

    std::string token;
    line_stream.clear();
    
    line_stream.str(line_str);
    size_t col = 0;
    while (line_stream.good()) {
      // 读取一行中的每个字符到token
      std::getline(line_stream, token, split_char);
      try {
        data.at(row, col) = std::stof(token);
      }
      catch (std::exception& e) {
        DLOG(ERROR) << "Parse CSV File meet error: " << e.what() << " row: " << row << " col: " << col;
      }
      col += 1;
      CHECK(col <= cols) << "There are excessive elements on the column";
    }

    row += 1;
    CHECK(row <= rows) << "There are excessive elements on the row";
  }
  
  return data;
}

std::pair<size_t, size_t> CSVDataLoader::GetMatrixSize(std::ifstream& file, char split_char) {
  bool load_ok = file.good();
  file.clear();
  size_t fn_rows = 0;
  size_t fn_cols = 0;
  const std::ifstream::pos_type start_pos = file.tellg();

  std::string token;
  std::string line_str;
  std::stringstream line_stream;

  while (file.good() && load_ok) {
    // 读入一行到line_str
    std::getline(file, line_str);
    if (line_str.empty()) {
      break;
    }

    // 将string对象转化为stringstream对象
    line_stream.clear();
    line_stream.str(line_str);
    
    size_t line_cols = 0;
    std::string row_token;
    while (line_stream.good()) {
      // 读取一行中的每个字符到row_token
      std::getline(line_stream, row_token, split_char);
      ++line_cols;
    }

    if (line_cols > fn_cols) {
      fn_cols = line_cols;
    }

    ++fn_rows;
  }
  file.clear();
  file.seekg(start_pos);
  return {fn_rows, fn_cols};
}

}  // namespace TinyInfer