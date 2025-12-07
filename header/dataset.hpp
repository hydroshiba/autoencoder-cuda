#pragma once
#include <vector>
#include "tensor.hpp"
using namespace std;

class DataLoader
{
    private:
        Tensor train_images;
        vector<ushort> train_labels;

        Tensor test_images;
        vector<ushort> test_labels;

        void shuffle_data();
        void load_data(const string& dataset_path);
        void normalize_data();
    public:
        DataLoader(const string& dataset_path);

        vector<Tensor> get_batch(const int& batch_idx, const int& batch_size);

        int num_train() const;
        int num_test() const;
};