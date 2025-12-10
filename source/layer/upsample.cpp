#include "layer.hpp"

Tensor UpSample2D::forward_cpu(const Tensor &input)
{
    cached_input = input;

    int N = input.N;
    int C = input.C;
    int H = input.H;
    int W = input.W;

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

                    output.host_data[output.index(n, c, h, w)] =
                        input.host_data[input.index(n, c, ih, iw)];
                }
            }
        }
    }

    return output;
}

Tensor UpSample2D::backward_cpu(const Tensor &grad_output)
{
    Tensor grad_input(cached_input.N, cached_input.C,
                      cached_input.H, cached_input.W);

    std::fill(grad_input.host_data,
              grad_input.host_data + grad_input.size(),
              0.0f);

    int out_h = cached_input.H * scale_factor;
    int out_w = cached_input.W * scale_factor;

    for (int n = 0; n < cached_input.N; n++)
    {
        for (int c = 0; c < cached_input.C; c++)
        {
            for (int h = 0; h < out_h; h++)
            {
                for (int w = 0; w < out_w; w++)
                {

                    int ih = h / scale_factor;
                    int iw = w / scale_factor;

                    grad_input.host_data[grad_input.index(n, c, ih, iw)] += grad_output.host_data[grad_output.index(n, c, h, w)];
                }
            }
        }
    }

    return grad_input;
}
