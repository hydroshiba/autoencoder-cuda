#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <string>
#include <fstream>

#include "autoencoder.hpp"
#include "dataset.hpp"
#include "tensor.hpp"

class Trainer
{
private:
    int epochs;
    int batch_size;
    float learning_rate;
    bool use_gpu;

    Autoencoder::Base *model;
    DataLoader *data_loader;

    std::ofstream log_file;

public:
    Trainer(Autoencoder::Base *model, DataLoader *data_loader, const std::string &config_path);

    void load_config(const std::string &path);

    void train();
    
    float train_one_epoch(int epoch_idx, float& epoch_time_ms);

    float compute_loss(const Tensor &output, const Tensor &target);

    void save_checkpoint(const std::string &path);
    void load_checkpoint(const std::string &path);
};

#endif