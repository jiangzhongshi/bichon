#include <doctest.h>

#include <Eigen/Core>
#include <fstream>
#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

template <typename T>
bool read(std::string path, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat)
{
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Index Index;

    std::ifstream in(path, std::ios::in | std::ios::binary);
    if (!in.good()) {
        // logger().error("Failed to open file: {}", path);
        in.close();

        return false;
    }

    Index rows = 0, cols = 0;
    in.read((char*)(&rows), sizeof(Index));
    in.read((char*)(&cols), sizeof(Index));

    mat.resize(rows, cols);
    in.read((char*)mat.data(), rows * cols * sizeof(T));
    in.close();

    return true;
}

TEST_CASE("read-io")
{
    std::string path = "/Users/zhongshi/Downloads/test/u_2.txt";
    Eigen::MatrixXd mat;
    read(path, mat);
    spdlog::info("mat \n{}", mat);
}