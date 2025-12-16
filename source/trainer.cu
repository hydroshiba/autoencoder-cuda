#include "trainer.hpp"
#include "dataset.hpp"
#include "autoencoder.hpp"

#include <iostream>
#include <stdexcept>
#include <chrono>

#include <cuda_runtime.h>

/* ================= CUDA KERNEL ================= */

__global__ void mse_loss_kernel(
    const float *output,
    const float *target,
    float *loss,
    int N)
{
    __shared__ float cache[256];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int cache_idx = threadIdx.x;

    float temp = 0.0f;
    if (tid < N)
    {
        float diff = output[tid] - target[tid];
        temp = diff * diff;
    }

    cache[cache_idx] = temp;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (cache_idx < stride)
            cache[cache_idx] += cache[cache_idx + stride];
        __syncthreads();
    }

    if (cache_idx == 0)
        atomicAdd(loss, cache[0]);
}

__global__ void mse_grad_kernel(
    const float *output,
    const float *input,
    float *grad,
    int N,
    float scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        grad[i] = scale * (output[i] - input[i]);
}

/* ================= TRAINER ================= */

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
    epochs = 20;
    learning_rate = 0.001f;

    if (use_gpu)
        batch_size = 64; // GPU
    else
        batch_size = 32; // CPU

    std::cout << "[Trainer] Mode: " << (use_gpu ? "GPU" : "CPU") << std::endl;
    std::cout << "[Trainer] Batch size = " << batch_size << std::endl;
    std::cout << "[Trainer] Epochs = " << epochs << std::endl;
}

/* ================= LOSS ================= */

float Trainer::compute_loss(const Tensor &output, const Tensor &target)
{
    if (output.size() != target.size())
        throw std::runtime_error("compute_loss: size mismatch");

    const int N = output.size();

    /* -------- CPU -------- */
    if (!use_gpu)
    {
        float loss = 0.0f;
        const float *out = output.data();
        const float *tgt = target.data();

        for (int i = 0; i < N; ++i)
        {
            float diff = out[i] - tgt[i];
            loss += diff * diff;
        }
        return loss / N;
    }
    /* -------- GPU -------- */
    float *d_loss;
    float h_loss = 0.0f;

    cudaMalloc(&d_loss, sizeof(float));
    cudaMemset(d_loss, 0, sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    mse_loss_kernel<<<blocks, threads>>>(
        output.data(),
        target.data(),
        d_loss,
        N);

    cudaDeviceSynchronize();

    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_loss);

    return h_loss / N;
}

/* ================= TRAIN LOOP ================= */

void Trainer::train()
{
    log_file.open("training_log.csv");
    log_file << "epoch,loss,time_ms\n";

    std::cout << "[Trainer] Start training..." << std::endl;

    for (int e = 0; e < epochs; ++e)
    {
        float epoch_time = 0.0f;
        float avg_loss = train_one_epoch(e, epoch_time);

        log_file << e + 1 << "," << avg_loss << "," << epoch_time << "\n";
    }

    log_file.close();

    std::cout << "[Trainer] Training completed." << std::endl;
}

float Trainer::train_one_epoch(int epoch_idx, float &epoch_time_ms)
{
    using clock = std::chrono::high_resolution_clock;
    auto t_start = clock::now();

    int num_samples;

    if (use_gpu)
        num_samples = data_loader->num_train();
    else
        num_samples = 160;

    const int num_batches = (num_samples + batch_size - 1) / batch_size;

    std::cout << "Number of Batches: " << num_batches << std::endl;

    float total_loss = 0.0f;

    for (int i = 0; i < num_batches; ++i)
    {
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " start" << std::endl;

        // Get batch
        Tensor input = data_loader->get_batch(i, batch_size);

        if (use_gpu)
            input.to_gpu();

        // Forward pass
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " forward starting..." << std::endl;
        Tensor output = model->forward(input);
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " forward done" << std::endl;

        // Compute loss
        float loss = compute_loss(output, input);
        total_loss += loss;

        const int N = output.size();
        const float scale = 2.0f / N;

        Tensor grad(output.batch(), output.channels(), output.height(), output.width());

        if (use_gpu)
        {
            grad.to_gpu();

            int threads = 256;
            int blocks = (N + threads - 1) / threads;

            mse_grad_kernel<<<blocks, threads>>>(
                output.data(),
                input.data(),
                grad.data(),
                N,
                scale);

            cudaDeviceSynchronize();

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA error: "  << cudaGetErrorString(err)  << std::endl;
                return 0;
            }
        }
        else
        {
            for (int j = 0; j < N; ++j)
            {
                grad.data()[j] =
                    scale * (output.data()[j] - input.data()[j]);
            }
        }

        // Backward pass
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " backward starting..." << std::endl;
        model->backward(grad);
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " backward done" << std::endl;

        // Update weights
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " update starting..." << std::endl;
        model->update(learning_rate);
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " update done" << std::endl;

        if (use_gpu)
            cudaDeviceSynchronize();

        /* -------- Log -------- */
        if (i % 10 == 0)
        {
            std::cout << "[Epoch " << epoch_idx + 1 << " | Batch " << i << "/" << num_batches << "] Loss = " << loss << std::endl;
        }
    }

    auto t_end = clock::now();
    epoch_time_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    float avg_loss = total_loss / num_batches;

    std::cout << ">>> Epoch " << epoch_idx + 1 << " | Loss = " << avg_loss << " | Time = " << epoch_time_ms << " ms" << std::endl;

    return avg_loss;
}

void Trainer::save_checkpoint(const std::string &path)
{
    model->save_model(path);
}

void Trainer::load_checkpoint(const std::string &path)
{
    model->load_model(path);
}