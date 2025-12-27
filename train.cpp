#include <iostream>

#include "header/trainer.hpp"
#include "header/dataset.hpp"
#include "header/autoencoder.hpp"

int main()
{
    std::cout << "[MAIN] Program started\n";

    Autoencoder::Base *model = new Autoencoder::CPU();
    std::cout << "[MAIN] Model created\n";

    DataLoader *data_loader = new DataLoader("cifar-10-batches-bin");
    std::cout << "[MAIN] DataLoader created\n";

    Trainer trainer(model, data_loader, "./config.yaml");
    std::cout << "[MAIN] Trainer created\n";

    trainer.train();
    std::cout << "[MAIN] Training finished\n";

    trainer.save_checkpoint("checkpoint.dat");
    std::cout << "[MAIN] Checkpoint saved\n";

    return 0;
}