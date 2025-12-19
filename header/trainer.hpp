#ifndef TRAINER_HPP
#define TRAINER_HPP

#include <string>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

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
    
    // CUDA stream support for concurrent batch processing
    int num_streams;
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> batch_complete_events;
    std::vector<cudaEvent_t> update_complete_events;
    
    // Per-stream loss buffers (device and pinned host)
    std::vector<float*> d_loss_buffers;
    std::vector<float*> h_loss_pinned;

public:
    Trainer(Autoencoder::Base *model, DataLoader *data_loader, const std::string &config_path);
    ~Trainer();

    void load_config(const std::string &path);

    void train();
    
    float train_one_epoch(int epoch_idx, float& epoch_time_ms);

    float compute_loss(const Tensor &output, const Tensor &target);
    float compute_loss_async(const Tensor &output, const Tensor &target, 
                            cudaStream_t stream, int stream_id);

    void save_checkpoint(const std::string &path);
    void load_checkpoint(const std::string &path);
    
private:
    void init_streams();
    void cleanup_streams();
};

#endif