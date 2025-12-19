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
    : num_streams(1)
{
    this->model = model;
    this->data_loader = data_loader;

    // Default use_gpu based on model type; load_config can override
    use_gpu = (dynamic_cast<Autoencoder::GPU *>(model) != nullptr);

    load_config(config_path);
    
    if (use_gpu)
    {
        init_streams();
    }
}

Trainer::~Trainer()
{
    if (use_gpu)
    {
        cleanup_streams();
    }
}

void Trainer::init_streams()
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    // Determine optimal stream count based on GPU capabilities
    num_streams = std::min(deviceProp.multiProcessorCount / 8, deviceProp.asyncEngineCount);
    num_streams = std::max(2, std::min(num_streams, 8));  // Clamp to [2, 8]
    
    // Check available memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t mem_per_stream = 200 * 1024 * 1024;  // 200MB estimate per stream
    int max_by_mem = static_cast<int>(free_mem / mem_per_stream / 2);
    num_streams = std::min(num_streams, max_by_mem);
    
    std::cout << "[Trainer] GPU: " << deviceProp.name << std::endl;
    std::cout << "[Trainer] SMs: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "[Trainer] Using " << num_streams << " concurrent streams" << std::endl;
    std::cout << "[Trainer] Sub-batch size per stream: " << batch_size << " images" << std::endl;
    std::cout << "[Trainer] Effective batch size: " << (batch_size * num_streams) << " images" << std::endl;
    
    // Create streams and events
    streams.resize(num_streams);
    batch_complete_events.resize(num_streams);
    update_complete_events.resize(num_streams);
    d_loss_buffers.resize(num_streams, nullptr);
    h_loss_pinned.resize(num_streams, nullptr);
    
    for (int i = 0; i < num_streams; ++i)
    {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&batch_complete_events[i]);
        cudaEventCreate(&update_complete_events[i]);
        
        // Allocate loss buffers
        cudaMalloc(&d_loss_buffers[i], sizeof(float));
        cudaMallocHost(&h_loss_pinned[i], sizeof(float));
    }
    
    std::cout << "[Trainer] Streams initialized successfully" << std::endl;
}

void Trainer::cleanup_streams()
{
    for (int i = 0; i < num_streams; ++i)
    {
        if (streams[i])
            cudaStreamDestroy(streams[i]);
        if (batch_complete_events[i])
            cudaEventDestroy(batch_complete_events[i]);
        if (update_complete_events[i])
            cudaEventDestroy(update_complete_events[i]);
        if (d_loss_buffers[i])
            cudaFree(d_loss_buffers[i]);
        if (h_loss_pinned[i])
            cudaFreeHost(h_loss_pinned[i]);
    }
}

void Trainer::load_config(const std::string &path)
{
    epochs = 20;
    learning_rate = 0.001f;

    if (use_gpu)
        batch_size = 32; // Per-stream batch size
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

float Trainer::compute_loss_async(const Tensor &output, const Tensor &target,
                                  cudaStream_t stream, int stream_id)
{
    if (output.size() != target.size())
        throw std::runtime_error("compute_loss_async: size mismatch");

    const int N = output.size();
    
    cudaMemsetAsync(d_loss_buffers[stream_id], 0, sizeof(float), stream);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    mse_loss_kernel<<<blocks, threads, 0, stream>>>(
        output.data(),
        target.data(),
        d_loss_buffers[stream_id],
        N);
    
    // Async copy to pinned host memory
    cudaMemcpyAsync(h_loss_pinned[stream_id], d_loss_buffers[stream_id],
                    sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // Return placeholder - actual value read after stream sync
    return *h_loss_pinned[stream_id] / N;
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

        // Select stream (single-stream execution for now)
        int stream_id = 0;
        cudaStream_t stream = streams.empty() ? 0 : streams[stream_id];

        // Forward pass
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " forward starting..." << std::endl;
        Tensor output = model->forward(input, stream);
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " forward done" << std::endl;

        float loss = 0.0f;

        // Compute loss (GPU async / CPU sync)
        if (use_gpu && !streams.empty())
        {
            compute_loss_async(output, input, stream, stream_id);
        }

        const int N = output.size();
        const float scale = 2.0f / N;

        Tensor grad(output.batch(), output.channels(), output.height(), output.width());

        if (use_gpu)
        {
            grad.to_gpu();

            int threads = 256;
            int blocks = (N + threads - 1) / threads;

            mse_grad_kernel<<<blocks, threads, 0, stream>>>(
                output.data(),
                input.data(),
                grad.data(),
                N,
                scale);

            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                std::cerr << "CUDA error: "  << cudaGetErrorString(err)  << std::endl;
                return 0;
            }
        }
        else
        {
            float *output_data = output.data();
            float *input_data = input.data();
            float *grad_data = grad.data();
            for (int j = 0; j < N; ++j)
            {
                grad_data[j] = scale * (output_data[j] - input_data[j]);
            }
        }

        // Backward pass
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " backward starting..." << std::endl;
        model->backward(grad, stream);
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " backward done" << std::endl;

        // Update weights
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " update starting..." << std::endl;
        model->update(learning_rate, stream);
        std::cout << "[Epoch " << epoch_idx + 1 << "] Batch " << i << " update done" << std::endl;

        if (use_gpu)
        {
            if (!streams.empty())
            {
                cudaStreamSynchronize(stream);
                loss = *h_loss_pinned[stream_id] / N;
            }
            else
            {
                loss = compute_loss(output, input);
            }
            total_loss += loss;
        }
        else
        {
            loss = compute_loss(output, input);
            total_loss += loss;
        }

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