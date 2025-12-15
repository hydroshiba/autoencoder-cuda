#include "trainer.hpp"
#include "dataset.hpp"

#include <iostream>
#include <fstream>

#include <cuda_runtime.h>

__global__ void mse_grad_kernel(const float *output, const float *input, float *grad, int N, float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        grad[i] = scale * (output[i] - input[i]);
    }
}

Trainer::Trainer(Autoencoder::Base *model, DataLoader *data_loader, const std::string &config_path)
{
    this->model = model;
    this->data_loader = data_loader;
    // Default use_gpu based on model type; load_config can override
    use_gpu = (dynamic_cast<Autoencoder::GPU *>(model) != nullptr);

    load_config(config_path);
}

void Trainer::load_config(const std::string &path)
{
    // TODO
    // For simplicity, we hardcode some parameters here
    epochs = 100;
    batch_size = 64;
    learning_rate = 0.001f;
    // leave use_gpu as detected unless config wants to force it; for now keep detected value
}

// float Trainer::compute_loss(const Tensor &output, const Tensor &target)
// {
//     if (output.size() != target.size())
//     {
//         std::cerr << "[Trainer] Error: Output and target size mismatch in compute_loss." << std::endl;
//         return -1.0f;
//     }

//     float loss = 0.0f;

//     const float* output_data = output.data();
//     const float* target_data = target.data();

//     if (! use_gpu)
//     {
//         for (int i = 0; i < output.size(); ++i)
//         {
//             float diff = output_data[i] - target_data[i];
//             loss += diff * diff;
//         }
//     }
//     else{
//         // GPU loss computation
//         float* d_output;
//         float* d_target;
//         float* d_loss;
//         float h_loss = 0.0f;

//         CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
//         CHECK(cudaMalloc(&d_target, target.size() * sizeof(float)));
//         CHECK(cudaMalloc(&d_loss, sizeof(float)));

//         CHECK(cudaMemcpy(d_output, output_data, output.size() * sizeof(float), cudaMemcpyHostToDevice));
//         CHECK(cudaMemcpy(d_target, target_data, target.size() * sizeof(float), cudaMemcpyHostToDevice));
//         CHECK(cudaMemset(d_loss, 0, sizeof(float)));

//         int threads = 256;
//         int blocks = (output.size() + threads - 1) / threads;
//         compute_mse_loss_kernel<<<blocks, threads>>>(d_output, d_target, d_loss, output.size());
//         CHECK(cudaDeviceSynchronize());

//         CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

//         loss = h_loss;

//         cudaFree(d_output);
//         cudaFree(d_target);
//         cudaFree(d_loss);
//     }

//     return loss / output.size();
// }

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
    int num_batches = data_loader->num_train() / batch_size;
    float total_loss = 0.0f;

    for (int i = 0; i < num_batches; ++i)
    {
        std::cout << "[DEBUG] Batch " << i << " starting..." << std::endl;

        // Get batch
        Tensor input = data_loader->get_batch(i, batch_size);
        std::cout << "[DEBUG] Batch " << i << " loaded" << std::endl;

        // (Optional) Use GPU
        if (use_gpu)
        {
            input.to_gpu();
        }

        // Forward pass
        std::cout << "[DEBUG] Batch " << i << " forward starting..." << std::endl;
        Tensor output = model->forward(input);
        std::cout << "[DEBUG] Batch " << i << " forward done" << std::endl;

        // Compute loss
        // float loss = compute_loss(output, input);
        // total_loss += loss;
        // std::cout << "[DEBUG] Batch " << i << " loss computed: " << loss << std::endl;

        // Compute MSE gradient: grad = 2 * (output - input) / N
        const int N = output.size();
        const float scale = 2.0f / N;
        Tensor grad_loss(output.batch(), output.channels(), output.height(), output.width());
        if (use_gpu)
        {
            // Ensure tensors on device
            Tensor out_gpu = output;
            Tensor in_gpu = input;
            out_gpu.to_gpu();
            in_gpu.to_gpu();
            grad_loss.to_gpu();

            int threads = 256;
            int blocks = (N + threads - 1) / threads;
            mse_grad_kernel<<<blocks, threads>>>(out_gpu.data(), in_gpu.data(), grad_loss.data(), N, scale);
            cudaDeviceSynchronize();
        }
        else
        {
            for (int j = 0; j < N; ++j)
            {
                grad_loss.data()[j] = scale * (output.data()[j] - input.data()[j]);
            }
        }
        std::cout << "[DEBUG] Batch " << i << " loss gradient computed" << std::endl;

        // Backward pass
        std::cout << "[DEBUG] Batch " << i << " backward starting..." << std::endl;
        model->backward(grad_loss);
        std::cout << "[DEBUG] Batch " << i << " backward done" << std::endl;

        // Update weights
        std::cout << "[DEBUG] Batch " << i << " update starting..." << std::endl;
        model->update(learning_rate);
        std::cout << "[DEBUG] Batch " << i << " update done" << std::endl;

        // Log
        if (i % 10 == 0)
        {
            // std::cout << "[Epoch " << epoch_idx + 1 << " | Batch " << i << "/" << num_batches << "] Loss = " << loss << std::endl;
        }
    }

    // std::cout << ">>> Epoch " << epoch_idx + 1 << " | Average Loss = " << total_loss / num_batches << std::endl;
}

void Trainer::save_checkpoint(const std::string &path)
{
    model->save_model(path);
}

void Trainer::load_checkpoint(const std::string &path)
{
    model->load_model(path);
}