// #include <iostream>

// #include "header/trainer.hpp"
// #include "header/dataset.hpp"
// #include "header/autoencoder.hpp"

// int main()
// {
//     std::cout << "[MAIN] Program started\n";

//     Autoencoder::Base *model = new Autoencoder::CPU();
//     std::cout << "[MAIN] Model created\n";

//     DataLoader *data_loader = new DataLoader("cifar-10-batches-bin");
//     std::cout << "[MAIN] DataLoader created\n";

//     Trainer trainer(model, data_loader, "./config.yaml");
//     std::cout << "[MAIN] Trainer created\n";

//     trainer.train();
//     std::cout << "[MAIN] Training finished\n";

//     trainer.save_checkpoint("checkpoint.dat");
//     std::cout << "[MAIN] Checkpoint saved\n";

//     return 0;
// }


#include <iostream>
#include <vector>

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

    /* ===============================
       LOAD CHECKPOINT
       =============================== */
    trainer.load_checkpoint("model/checkpoint_gpu_full_7h.dat");
    std::cout << "[MAIN] Checkpoint loaded\n";

    /* ===============================
       INFERENCE FIRST 3 BATCHES
       =============================== */
    const size_t NUM_BATCHES = 3;
    const size_t PRINT_COUNT = 180;

    for (size_t batch_idx = 0; batch_idx < NUM_BATCHES; ++batch_idx)
    {

        // lấy batch thứ batch_idx
        auto input = data_loader->get_batch(batch_idx, 1);

        // inference (forward only)
        auto output = model->forward(input);

        std::cout << "\n[INFERENCE] Batch " << batch_idx
                  << " - First 180 values:\n";

        size_t count = std::min(PRINT_COUNT, output.size());
        for (size_t i = 0; i < count; ++i)
        {
            std::cout << output.data()[i] << ' ';
        }

        std::cout << "\n----------------------------------\n";
    }

    delete model;
    delete data_loader;

    std::cout << "[MAIN] Done\n";
    return 0;
}
