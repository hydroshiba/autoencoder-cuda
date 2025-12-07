#pragma once

#include <cstddef>
#include <vector>

class Tensor
{
    private:
        std::vector<float> host_data;
        float* device_data;
        int N, C, H, W;
        bool on_gpu;

        std::size_t index(int n, int c, int h, int w) const;

    public:
        Tensor();
        Tensor(int n, int c, int h, int w);
        Tensor(const Tensor& other);
        Tensor& operator=(const Tensor& other);
        ~Tensor();

        void resize(int n, int c, int h, int w);

        int batch() const;
        int channels() const;
        int height() const;
        int width() const;

        size_t size() const;

        void to_gpu();
        void to_cpu();
        bool is_gpu() const;

        float& operator()(int n, int c, int h, int w);
        const float& operator()(int n, int c, int h, int w) const;

};