#include "layer.hpp"

Tensor UpSample2D::forward_cpu(const Tensor &input)
{
    cached_input = input;

    int N = input.batch();
    int C = input.channels();
    int H = input.height();
    int W = input.width();

    int out_h = H * scale_factor;
    int out_w = W * scale_factor;

    Tensor output(N, C, out_h, out_w);

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < out_h; h++)
            {
                for (int w = 0; w < out_w; w++)
                {

                    int ih = h / scale_factor;
                    int iw = w / scale_factor;

                    output(n, c, h, w) = input(n, c, ih, iw);
                }
            }
        }
    }

    return output;
}

Tensor UpSample2D::backward_cpu(const Tensor &grad_output)
{
    Tensor grad_input(cached_input.batch(), cached_input.channels(),
                        cached_input.height(), cached_input.width());

    std::fill(grad_input.data(),
              grad_input.data() + grad_input.size(),
              0.0f);

    int out_h = cached_input.height() * scale_factor;
    int out_w = cached_input.width() * scale_factor;

    for (int n = 0; n < cached_input.batch(); n++)
    {
        for (int c = 0; c < cached_input.channels(); c++)
        {
            for (int h = 0; h < out_h; h++)
            {
                for (int w = 0; w < out_w; w++)
                {

                    int ih = h / scale_factor;
                    int iw = w / scale_factor;

                    grad_input(n, c, ih, iw) += grad_output(n, c, h, w);
                }
            }
        }
    }

    return grad_input;
}
Tensor UpSample2D::forward_gpu(const Tensor &input)
{
    // GPU version: currently fallback to CPU
    return forward_cpu(input);
}

Tensor UpSample2D::backward_gpu(const Tensor &grad_output)
{
    // GPU version: currently fallback to CPU
    return backward_cpu(grad_output);
}