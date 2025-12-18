#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <string>
#include <fstream>

#include "config.hpp"
#include "autoencoder.hpp"
#include "dataset.hpp"
#include "loss.cuh"

#include "utils/timer.cuh"

class Trainer {
private:
    Timer timer;

    const size_t batch_size;
    const size_t epochs;
    const float learning_rate;

public:
    Trainer() : batch_size(Config::batch_size), epochs(Config::epochs), learning_rate(Config::learning_rate) {}
    template <typename Tag, typename Optimizer, typename Loss>
    void fit(Autoencoder<Tag> &model, Dataset &dataset, Optimizer &optimizer, Loss loss = Loss::MSE);
};

#include "trainer.tpp"

#endif