#include <iostream>
#include "header/trainer.hpp"
#include "header/dataset.hpp"
#include "header/autoencoder.hpp"

int main()
{
    std::cout << "[MAIN] Inference process started\n";

    Autoencoder::Base *model = new Autoencoder::CPU();
    std::cout << "[MAIN] Model created\n";

    model->load_model("checkpoint.dat");
    std::cout << "[MAIN] Checkpoint loaded\n";

    DataLoader *data_loader = new DataLoader("cifar-10-batches-bin");
    Tensor testImage = data_loader->get_batch(1,1);

    Tensor inferedEncoding = model->encode(testImage);

    // TODO: Add evaluation code here
   
    std::cout << "[MAIN] Evaluation finished\n";

    return 0;
}