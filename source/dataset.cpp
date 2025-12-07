#include "dataset.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

DataLoader::DataLoader(const std::string& dataset_path)
{
    load_data(dataset_path);
    normalize_data();
    shuffle_data();
}

void DataLoader::load_data(const std::string& dataset_path)
{
    constexpr int image_size = 32;
    constexpr int channels = 3;
    constexpr int pixels_per_image = image_size * image_size;
    constexpr int record_bytes = 1 + channels * pixels_per_image; // 1 label + image bytes
    constexpr int train_count = 50000;
    constexpr int test_count = 10000;

    train_images.resize(train_count, channels, image_size, image_size);
    train_labels.assign(train_count, 0);
    test_images.resize(test_count, channels, image_size, image_size);
    test_labels.assign(test_count, 0);

    auto load_split = [&](const std::vector<std::string>& files, Tensor& images, std::vector<ushort>& labels, int max_samples)
    {
        std::array<unsigned char, record_bytes> buffer{};
        int sample_idx = 0;

        for (const auto& file : files)
        {
            const std::string path = dataset_path + "/" + file;
            std::ifstream in(path, std::ios::binary);
            if (!in)
            {
                throw std::runtime_error("Failed to open CIFAR-10 file: " + path);
            }

            while (sample_idx < max_samples && in.read(reinterpret_cast<char*>(buffer.data()), buffer.size()))
            {
                const unsigned char label = buffer[0];

                for (int c = 0; c < channels; ++c)
                {
                    const int channel_offset = 1 + c * pixels_per_image;
                    for (int h = 0; h < image_size; ++h)
                    {
                        for (int w = 0; w < image_size; ++w)
                        {
                            const int idx = channel_offset + h * image_size + w;
                            images(sample_idx, c, h, w) = static_cast<float>(buffer[idx]);
                        }
                    }
                }

                labels[sample_idx] = static_cast<ushort>(label);
                ++sample_idx;
            }

            if (!in.eof())
            {
                throw std::runtime_error("Unexpected read error while parsing " + path);
            }
        }

        if (sample_idx != max_samples)
        {
            throw std::runtime_error("Loaded " + std::to_string(sample_idx) + " samples, expected " + std::to_string(max_samples));
        }
    };

    const std::vector<std::string> train_files = {
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"};

    const std::vector<std::string> test_files = {"test_batch.bin"};

    load_split(train_files, train_images, train_labels, train_count);
    load_split(test_files, test_images, test_labels, test_count);
}

void DataLoader::normalize_data()
{
    auto normalize = [](Tensor& images)
    {
        const float scale = 1.0f / 255.0f;
        const int N = images.batch();
        const int C = images.channels();
        const int H = images.height();
        const int W = images.width();

        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int h = 0; h < H; ++h)
                {
                    for (int w = 0; w < W; ++w)
                    {
                        images(n, c, h, w) *= scale;
                    }
                }
            }
        }
    };

    normalize(train_images);
    normalize(test_images);
}

void DataLoader::shuffle_data()
{
    const int N = train_images.batch();
    if (N <= 0)
    {
        return;
    }

    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    std::mt19937 rng(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), rng);

    Tensor shuffled(N, train_images.channels(), train_images.height(), train_images.width());
    std::vector<ushort> shuffled_labels(N);

    for (int new_idx = 0; new_idx < N; ++new_idx)
    {
        const int old_idx = indices[new_idx];
        for (int c = 0; c < train_images.channels(); ++c)
        {
            for (int h = 0; h < train_images.height(); ++h)
            {
                for (int w = 0; w < train_images.width(); ++w)
                {
                    shuffled(new_idx, c, h, w) = train_images(old_idx, c, h, w);
                }
            }
        }
        shuffled_labels[new_idx] = train_labels[old_idx];
    }

    train_images = shuffled;
    train_labels.swap(shuffled_labels);
}

int DataLoader::num_train() const
{
    return train_images.batch();
}

int DataLoader::num_test() const
{
    return test_images.batch();
}

vector<Tensor> DataLoader::get_batch(const int& batch_idx, const int& batch_size)
{
    const int total = train_images.batch();
    if (batch_idx < 0 || batch_size <= 0 || batch_idx * batch_size >= total)
    {
        throw std::out_of_range("Invalid batch index or size");
    }

    std::vector<Tensor> batch;
    const int start = batch_idx * batch_size;
    const int end = std::min(start + batch_size, total);
    batch.reserve(end - start);

    for (int n = start; n < end; ++n)
    {
        Tensor sample(1, train_images.channels(), train_images.height(), train_images.width());
        for (int c = 0; c < train_images.channels(); ++c)
        {
            for (int h = 0; h < train_images.height(); ++h)
            {
                for (int w = 0; w < train_images.width(); ++w)
                {
                    sample(0, c, h, w) = train_images(n, c, h, w);
                }
            }
        }
        batch.push_back(sample);
    }
    return batch;
}