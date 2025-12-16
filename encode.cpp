#include <iostream>
#include <cmath>

#include "header/autoencoder.hpp"
#include "header/dataset.hpp"

void compare_tensor(const Tensor& a, const Tensor& b, const std::string& name)
{
    Tensor ta = a;
    Tensor tb = b;

    ta.to_cpu();
    tb.to_cpu();

    const float* pa = ta.data();
    const float* pb = tb.data();

    size_t n = ta.size();

    float max_diff = 0.0f;
    float mean_diff = 0.0f;

    for (size_t i = 0; i < n; ++i)
    {
        float d = std::abs(pa[i] - pb[i]);
        max_diff = std::max(max_diff, d);
        mean_diff += d;
    }
    mean_diff /= n;

    std::cout << "[COMPARE] " << name  << " | mean = " << mean_diff  << " | max = " << max_diff << std::endl;
}

int main()
{
    std::cout << "[MAIN] Start inference check\n";

    Autoencoder::Base* cpu = new Autoencoder::CPU();
    Autoencoder::Base* gpu = new Autoencoder::GPU();

    cpu->load_model("model/checkpoint_gpu.dat");
    gpu->load_model("model/checkpoint_gpu.dat");

    DataLoader loader("cifar-10-batches-bin");

    Tensor input = loader.get_batch(1, 1);

    // ===== ENCODE =====
    Tensor z_cpu = cpu->encode(input);
    Tensor z_gpu = gpu->encode(input);

    compare_tensor(z_cpu, z_gpu, "ENCODE");

    // ===== DECODE =====
    Tensor out_cpu = cpu->decode(z_cpu);
    Tensor out_gpu = gpu->decode(z_gpu);

    compare_tensor(out_cpu, out_gpu, "DECODE");

    std::cout << "[MAIN] Done\n";

    delete cpu;
    delete gpu;

    return 0;
}
