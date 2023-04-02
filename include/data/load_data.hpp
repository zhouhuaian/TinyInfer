#ifndef TINY_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
#define TINY_INFER_INCLUDE_DATA_LOAD_DATA_HPP_

#include <string>
#include <armadillo>

namespace TinyInfer {

class CSVDataLoader {
public:
  /**
   * 读取csv文件中保存的矩阵
   * @param file_path 文件路径
   * @param split_char 分隔符
   * @return 初始化的矩阵
   */
  static arma::fmat LoadData(const std::string& file_path, char split_char = ',');

private:
  /**
   * 获取csv保存的矩阵维度
   * @param file csv文件流
   * @param split_char 分割符
   * @return 矩阵维度
   */
  static std::pair<size_t, size_t> GetMatrixSize(std::ifstream& file, char split_char);
};

}  // namespace TinyInfer

#endif  // TINY_INFER_INCLUDE_DATA_LOAD_DATA_HPP_
