#include "layer.hpp"

Tensor MaxPool2D::forward_cpu(const Tensor &input)
{
    cached_input = input;

    int N = input.batch();
    int C = input.channels();
    int H = input.height();
    int W = input.width();  

    int out_h = H / pool_size;
    int out_w = W / pool_size;

    Tensor output(N, C, out_h, out_w);
    max_indices.resize(N * C * out_h * out_w);

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
                            float v = input(n, c, ih, iw);

                            if (v > max_val)
                            {
                                max_val = v;
                                max_idx = idx;
                            }
                        }
                    }

                    output(n, c, oh, ow) = max_val;
                    max_indices[output.index(n, c, oh, ow)] = max_idx;
                }
            }
        }
    }

    return output;
}

Tensor MaxPool2D::backward_cpu(const Tensor &grad_output)
{
    Tensor grad_input = Tensor(cached_input.batch(), cached_input.channels(),
                                cached_input.height(), cached_input.width());

    std::fill(grad_input.data(),
              grad_input.data() + grad_input.size(),
              0.0f);

    for (size_t i = 0; i < max_indices.size(); i++)
    {
        int idx = max_indices[i];
        grad_input.data()[idx] += grad_output.data()[i];
    }

    return grad_input;
}
