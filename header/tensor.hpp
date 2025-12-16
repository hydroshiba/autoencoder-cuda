#pragma once

#include <cstddef>
#include <vector>

class Tensor
{
private:
    std::vector<float> host_data;
    float *device_data;
    int N, C, H, W;
    bool on_gpu;

public:
    // Constructors and destructor
    Tensor(bool on_gpu = false);
    Tensor(int n, int c, int h, int w, bool on_gpu = false);
    Tensor(const Tensor &other);
    Tensor &operator=(const Tensor &other);
    ~Tensor();

    // Information and manipulation
    void resize(int n, int c, int h, int w);
    int batch() const;
    int channels() const;

    int height() const;
    int width() const;

    size_t size() const;

    // Get element functions
    std::size_t index(int n, int c, int h, int w) const;

    // Access to underlying data -> element-wise
    float &operator()(int n, int c, int h, int w);
    const float &operator()(int n, int c, int h, int w) const;

    // Other functions
    Tensor &to_gpu();
    Tensor &to_cpu();
    bool is_gpu() const;

    float *data();
    const float *data() const;

    void clear_tensor();

    friend class Layer;
};