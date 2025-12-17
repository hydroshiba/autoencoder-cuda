// #include <iostream>
// #include <cmath>

// #include "header/autoencoder.hpp"
// #include "header/dataset.hpp"

// void compare_tensor(const Tensor& a, const Tensor& b, const std::string& name)
// {
//     Tensor ta = a;
//     Tensor tb = b;

//     ta.to_cpu();
//     tb.to_cpu();

//     const float* pa = ta.data();
//     const float* pb = tb.data();

//     size_t n = ta.size();

//     float max_diff = 0.0f;
//     float mean_diff = 0.0f;

//     for (size_t i = 0; i < n; ++i)
//     {
//         float d = std::abs(pa[i] - pb[i]);
//         max_diff = std::max(max_diff, d);
//         mean_diff += d;
//     }
//     mean_diff /= n;

//     std::cout << "[COMPARE] " << name  << " | mean = " << mean_diff  << " | max = " << max_diff << std::endl;
// }

// int main()
// {
//     std::cout << "[MAIN] Start inference check\n";

//     Autoencoder::Base* cpu = new Autoencoder::CPU();
//     Autoencoder::Base* gpu = new Autoencoder::GPU();

//     cpu->load_model("model/checkpoint_gpu.dat");
//     gpu->load_model("model/checkpoint_gpu.dat");

//     DataLoader loader("cifar-10-batches-bin");

//     Tensor input = loader.get_batch(1, 1);

//     // ===== ENCODE =====
//     Tensor z_cpu = cpu->encode(input);
//     Tensor z_gpu = gpu->encode(input);

//     compare_tensor(z_cpu, z_gpu, "ENCODE");

//     // ===== DECODE =====
//     Tensor out_cpu = cpu->decode(z_cpu);
//     Tensor out_gpu = gpu->decode(z_gpu);

//     compare_tensor(out_cpu, out_gpu, "DECODE");

//     std::cout << "[MAIN] Done\n";

//     delete cpu;
//     delete gpu;

//     return 0;
// }

#include "header/autoencoder.hpp"
#include "header/dataset.hpp"

#include <fstream>
#include <iostream>
#include <filesystem>

int main()
{
    std::filesystem::create_directories("/content/output");

    Autoencoder::GPU model;
    model.load_model("/content/checkpoint_gpu_full_7h.dat");

    DataLoader loader("/content/cifar-10-batches-bin");

    const int batch_size = 256;
    const int num_samples = loader.num_train();

    std::ofstream feat_out("/content/output/train_features.bin", std::ios::binary);
    std::ofstream label_out("/content/output/train_labels.bin", std::ios::binary);

    if (!feat_out || !label_out)
    {
        std::cerr << "Failed to open output files\n";
        return 1;
    }

    for (int i = 0; i * batch_size < num_samples; ++i)
    {
        Tensor batch = loader.get_batch(i, batch_size);

        batch.to_gpu();

        Tensor z = model.encode(batch);
        z.to_cpu();

        feat_out.write(
            reinterpret_cast<char*>(z.data()),
            z.size() * sizeof(float)
        );

        for (int j = 0; j < z.batch(); ++j)
        {
            unsigned short lbl =
                loader.get_train_label(i * batch_size + j);
            label_out.write(reinterpret_cast<char*>(&lbl), sizeof(lbl));
        }
    }

    feat_out.close();
    label_out.close();

    std::cout << "Feature extraction done.\n";
    return 0;
}
