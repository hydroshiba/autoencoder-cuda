#include "trainer.hpp"
#include "dataset.hpp"

#include <iostream>
#include <fstream>

Trainer::Trainer(Autoencoder *model, DataLoader *data_loader, const std::string &config_path)
{
    this->model = model;
    this->data_loader = data_loader;

    load_config(config_path);
}

void Trainer::load_config(const std::string &path)
{
    // TODO
}

float Trainer::compute_loss(const Tensor &output, const Tensor &target)
{
    if (output.size() != target.size())
    {
        std::cerr << "[Trainer] Error: Output and target size mismatch in compute_loss." << std::endl;
        return -1.0f;
    }

    float loss = 0.0f;

    for (size_t i = 0; i < output.size(); ++i)
    {
        float diff = output.host_data[i] - target.host_data[i];
        loss += diff * diff;
    }

    return loss / output.size();
}

void Trainer::train()
{
    std::cout << "[Trainer] Start training..." << std::endl;

    for (int e = 0; e < epochs; ++e)
    {
        train_one_epoch(e);
    }

    std::cout << "[Trainer] Training completed." << std::endl;
}

void Trainer::train_one_epoch(int epoch_idx)
{
    int num_batches = data_loader->num_train / batch_size;
    float total_loss = 0.0f;

    for (int i = 0; i < num_batches; ++i)
    {
        // Get batch
        Tensor input = data_loader->get_batch(i, batch_size);

        // (Optional) Use GPU
        if (use_gpu)
        {
            input.to_gpu();
        }

        // Forward
        Tensor output = model->forward(input);

        // Compute loss
        float loss = compute_loss(output, input);
        total_loss += loss;

        // Backward
        model->backward(output);

        // Update weights
        model->update(learning_rate);

        // Log
        if (i % 10 == 0)
        {
            std::cout << "[Epoch " << epoch_idx + 1 << " | Batch " << i << "/" << num_batches << "] Loss = " << loss << std::endl;
        }
    }

    std::cout << ">>> Epoch " << epoch_idx + 1 << " | Average Loss = " << total_loss / num_batches << std::endl;
}

void Trainer::save_checkpoint(const std::string &path)
{
    model->save_model(path);
}

void Trainer::load_checkpoint(const std::string &path)
{
    model->load_model(path);
}