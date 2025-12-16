#include "header/autoencoder.hpp"
#include "header/dataset.hpp"

#include <fstream>
#include <iostream>

int main()
{
    Autoencoder::GPU model;
    model.load_model("model/checkpoint.dat");

    DataLoader loader("cifar-10-batches-bin");

    const int batch_size = 256;
    const int num_samples = loader.num_train();

    std::ofstream feat_out("output/train_features.bin", std::ios::binary);
    std::ofstream label_out("output/train_labels.bin", std::ios::binary);

    for (int i = 0; i * batch_size < num_samples; ++i)
    {
        Tensor batch = loader.get_batch(i, batch_size);

        if (dynamic_cast<Autoencoder::GPU*>(&model))
            batch.to_gpu();

        Tensor z = model.encode(batch);

        z.to_cpu();

        feat_out.write(
            reinterpret_cast<char*>(z.data()),
            z.size() * sizeof(float)
        );

        for (int j = 0; j < z.batch(); ++j)
        {
            unsigned short lbl = loader.train_labels[i * batch_size + j];
            label_out.write(reinterpret_cast<char*>(&lbl), sizeof(lbl));
        }
    }

    std::cout << "Feature extraction done.\n";
}
