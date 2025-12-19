#include "layer.hpp"

UpSample2D::UpSample2D(int scale_factor_)
    : scale_factor(scale_factor_)
{
}

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
    int input_h = cached_input.height();
    int input_w = cached_input.width();
    int input_c = cached_input.channels();
    int input_n = cached_input.batch();

    Tensor grad_input(input_n, input_c, input_h, input_w);

    std::fill(grad_input.data(),
              grad_input.data() + grad_input.size(),
              0.0f);

    int out_h = input_h * scale_factor;
    int out_w = input_w * scale_factor;

    for (int n = 0; n < input_n; n++)
    {
        for (int c = 0; c < input_c; c++)
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