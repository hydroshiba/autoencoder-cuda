#include "header/trainer.hpp"
#include "header/dataset.hpp"
#include "header/autoencoder.hpp"

int main()
{
    // Example usage of Trainer class
    Autoencoder::Base* model = new Autoencoder::CPU();
    DataLoader* data_loader = new DataLoader("path/to/dataset");
    Trainer trainer(model, data_loader, "./config.yaml");

    trainer.train();

    trainer.save_checkpoint("checkpoint.dat");

    return 0;
}
