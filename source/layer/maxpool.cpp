#include "layer.hpp"

Tensor MaxPool2D::forward_cpu(const Tensor &input)
{
    cached_input = input;

    int N = input.N;
    int C = input.C;
    int H = input.H;
    int W = input.W;

    int out_h = H / pool_size;
    int out_w = W / pool_size;

    Tensor output(N, C, out_h, out_w);
    mask_indices.resize(N * C * out_h * out_w);

    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int oh = 0; oh < out_h; oh++)
            {
                for (int ow = 0; ow < out_w; ow++)
                {

                    float max_val = -1e9;
                    int max_idx = -1;

                    for (int kh = 0; kh < pool_size; kh++)
                    {
                        for (int kw = 0; kw < pool_size; kw++)
                        {

                            int ih = oh * pool_size + kh;
                            int iw = ow * pool_size + kw;

                            int idx = input.index(n, c, ih, iw);
                            float v = input.host_data[idx];

                            if (v > max_val)
                            {
                                max_val = v;
                                max_idx = idx;
                            }
                        }
                    }

                    output.host_data[output.index(n, c, oh, ow)] = max_val;
                    mask_indices[output.index(n, c, oh, ow)] = max_idx;
                }
            }
        }
    }

    return output;
}

Tensor MaxPool2D::backward_cpu(const Tensor &grad_output)
{
    Tensor grad_input = Tensor(cached_input.N, cached_input.C,
                               cached_input.H, cached_input.W);

    std::fill(grad_input.host_data,
              grad_input.host_data + grad_input.size(),
              0.0f);

    for (size_t i = 0; i < mask_indices.size(); i++)
    {
        int idx = mask_indices[i];
        grad_input.host_data[idx] += grad_output.host_data[i];
    }

    return grad_input;
}
