#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>

#include "tensor.hpp"

class DataLoader
{
private:
    Tensor train_images;
    std::vector<unsigned short> train_labels;

    Tensor test_images;
    std::vector<unsigned short> test_labels;

    void shuffle_data();
    void load_data(const std::string &dataset_path);
    void normalize_data();

public:
    DataLoader(const std::string &dataset_path);

    Tensor get_batch(const int &batch_idx, const int &batch_size);

    int num_train() const;
    int num_test() const;

    unsigned short get_train_label(int index) const;
};

#endif // DATASET_HPP