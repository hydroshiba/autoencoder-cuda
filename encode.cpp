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
    // =====================================================
    // 0. SETUP
    // =====================================================
    const std::string output_dir = "/content/output";
    std::filesystem::create_directories(output_dir);

    const int batch_size = 256;

    // =====================================================
    // 1. LOAD MODEL
    // =====================================================
    Autoencoder::GPU model;
    model.load_model("/content/checkpoint_gpu_full_7h.dat");

    // =====================================================
    // 2. LOAD DATASET
    // =====================================================
    DataLoader loader("/content/cifar-10-batches-bin");

    // =====================================================
    // 3. ENCODE TRAIN SET (50,000)
    // =====================================================
    const int num_train = loader.num_train();

    std::ofstream train_feat_out(
        output_dir + "/train_features.bin",
        std::ios::binary
    );
    std::ofstream train_label_out(
        output_dir + "/train_labels.bin",
        std::ios::binary
    );

    if (!train_feat_out || !train_label_out)
    {
        std::cerr << "❌ Failed to open TRAIN output files\n";
        return 1;
    }

    std::cout << "[ENCODE] Train set: " << num_train << " samples\n";

    for (int i = 0; i * batch_size < num_train; ++i)
    {
        Tensor batch = loader.get_batch(i, batch_size);
        batch.to_gpu();

        Tensor z = model.encode(batch);
        z.to_cpu();

        int N = z.batch();
        int C = z.channels();   // 128
        int H = z.height();     // 8
        int W = z.width();      // 8
        int D = C * H * W;      // 8192

        const float* z_data = z.data();

        // Write features
        for (int n = 0; n < N; ++n)
        {
            train_feat_out.write(
                reinterpret_cast<const char*>(z_data + n * D),
                D * sizeof(float)
            );
        }

        // Write labels
        for (int j = 0; j < N; ++j)
        {
            unsigned short lbl =
                loader.get_train_label(i * batch_size + j);

            train_label_out.write(
                reinterpret_cast<char*>(&lbl),
                sizeof(lbl)
            );
        }

        if (i % 20 == 0)
            std::cout << "  Train batch " << i << "\n";
    }

    train_feat_out.close();
    train_label_out.close();

    // =====================================================
    // 4. ENCODE TEST SET (10,000)
    // =====================================================
    const int num_test = loader.num_test();

    std::ofstream test_feat_out(
        output_dir + "/test_features.bin",
        std::ios::binary
    );
    std::ofstream test_label_out(
        output_dir + "/test_labels.bin",
        std::ios::binary
    );

    if (!test_feat_out || !test_label_out)
    {
        std::cerr << "❌ Failed to open TEST output files\n";
        return 1;
    }

    std::cout << "[ENCODE] Test set: " << num_test << " samples\n";

    for (int i = 0; i * batch_size < num_test; ++i)
    {
        Tensor batch = loader.get_test_batch(i, batch_size);
        batch.to_gpu();

        Tensor z = model.encode(batch);
        z.to_cpu();

        int N = z.batch();
        int D = z.channels() * z.height() * z.width(); // 8192

        const float* z_data = z.data();

        // Write features
        for (int n = 0; n < N; ++n)
        {
            test_feat_out.write(
                reinterpret_cast<const char*>(z_data + n * D),
                D * sizeof(float)
            );
        }

        // Write labels
        for (int j = 0; j < N; ++j)
        {
            unsigned short lbl =
                loader.get_test_label(i * batch_size + j);

            test_label_out.write(
                reinterpret_cast<char*>(&lbl),
                sizeof(lbl)
            );
        }

        if (i % 10 == 0)
            std::cout << "  Test batch " << i << "\n";
    }

    test_feat_out.close();
    test_label_out.close();

    // =====================================================
    // 5. DONE
    // =====================================================
    std::cout << "✅ Feature extraction DONE\n";
    std::cout << "Output files:\n";
    std::cout << " - train_features.bin\n";
    std::cout << " - train_labels.bin\n";
    std::cout << " - test_features.bin\n";
    std::cout << " - test_labels.bin\n";

    return 0;
}
