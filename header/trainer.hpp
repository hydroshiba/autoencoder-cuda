#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <string>
#include "autoencoder.hpp"
#include "dataset.hpp"

class Trainer
{
private:
    int epochs;
    int batch_size;
    float learning_rate;
    bool use_gpu;

    Autoencoder *model;
    DataLoader *data_loader;

public:
    Trainer(Autoencoder *model, DataLoader *data_loader, const std::string &config_path);

    void load_config(const std::string &path);

    void train();
    void train_one_epoch(int epoch_idx);

    float compute_loss(const Tensor &output, const Tensor &target);

    void save_checkpoint(const std::string &path);
    void load_checkpoint(const std::string &path);
};

#endif