#include "layer.hpp"

#include <cstddef>
#include <cmath>
#include <random>
#include <stdexcept>

namespace
{
    float kaiming_uniform_limit(int in_channels, int kernel_size)
    {
        if (in_channels <= 0 || kernel_size <= 0)
        {
            return 1.0f;
        }

        const float fan_in = static_cast<float>(in_channels * kernel_size * kernel_size);
        return std::sqrt(6.0f / fan_in);
    }
}

Conv2D::Conv2D(const int &in_ch, const int &out_ch, const int &k, const int &s, const int &p)
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k), stride(s), padding(p)
{
    if (in_channels <= 0 || out_channels <= 0)
    {
        throw std::invalid_argument("Conv2D requires positive channel counts");
    }
    if (kernel_size <= 0)
    {
        throw std::invalid_argument("Conv2D requires a positive kernel size");
    }
    if (stride <= 0)
    {
        throw std::invalid_argument("Conv2D stride must be positive");
    }
    if (padding < 0)
    {
        throw std::invalid_argument("Conv2D padding cannot be negative");
    }

    weights.resize(out_channels, in_channels, kernel_size, kernel_size);
    biases.resize(1, out_channels, 1, 1);

    std::mt19937 rng(std::random_device{}());
    const float limit = kaiming_uniform_limit(in_channels, kernel_size);
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (std::size_t idx = 0; idx < weights.size(); ++idx)
    {
        // weights(idx) = dist(rng);
    }
    for (int oc = 0; oc < out_channels; ++oc)
    {
        biases(0, oc, 0, 0) = 0.0f;
    }
}

Tensor Conv2D::forward_cpu(const Tensor &input)
{
    if (input.channels() != in_channels)
    {
        throw std::invalid_argument("Conv2D forward input channel mismatch");
    }

    const int batch_size = input.batch();
    const int input_height = input.height();
    const int input_width = input.width();

    if (input_height <= 0 || input_width <= 0)
    {
        throw std::invalid_argument("Conv2D forward expects positive spatial dimensions");
    }

    const int numerator_h = input_height + 2 * padding - kernel_size;
    const int numerator_w = input_width + 2 * padding - kernel_size;

    if (numerator_h < 0 || numerator_w < 0)
    {
        throw std::invalid_argument("Conv2D configuration yields negative output dimensions");
    }

    if (numerator_h % stride != 0 || numerator_w % stride != 0)
    {
        throw std::invalid_argument("Conv2D stride does not divide the padded dimensions");
    }

    const int out_height = numerator_h / stride + 1;
    const int out_width = numerator_w / stride + 1;

    Tensor output(batch_size, out_channels, out_height, out_width);

    for (int n = 0; n < batch_size; ++n)
    {
        for (int oc = 0; oc < out_channels; ++oc)
        {
            for (int oh = 0; oh < out_height; ++oh)
            {
                for (int ow = 0; ow < out_width; ++ow)
                {
                    float sum = biases(0, oc, 0, 0);

                    for (int ic = 0; ic < in_channels; ++ic)
                    {
                        for (int kh = 0; kh < kernel_size; ++kh)
                        {
                            for (int kw = 0; kw < kernel_size; ++kw)
                            {
                                const int ih = oh * stride + kh - padding;
                                const int iw = ow * stride + kw - padding;

                                if (ih < 0 || ih >= input_height || iw < 0 || iw >= input_width)
                                {
                                    continue;
                                }

                                sum += input(n, ic, ih, iw) * weights(oc, ic, kh, kw);
                            }
                        }
                    }

                    output(n, oc, oh, ow) = sum;
                }
            }
        }
    }

    return output;
}