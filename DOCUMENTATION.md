# Tài Liệu Dự Án: Autoencoder CUDA

## I. TỔNG QUAN DỰ ÁN

### Mục Đích
Dự án xây dựng một **Autoencoder** được tối ưu hóa sử dụng **CUDA** (GPU computing) cho việc nén và giải nén dữ liệu hình ảnh. Mô hình có khả năng chạy trên cả CPU và GPU.

### Tập Dữ Liệu
- **Dataset**: CIFAR-10 (định dạng nhị phân)
  - 50,000 ảnh huấn luyện
  - 10,000 ảnh kiểm tra
  - Kích thước: 32×32 pixels
  - Kênh màu: 3 (RGB)

### Công Nghệ Chính
- **Ngôn ngữ**: C++ 17, CUDA
- **Build System**: CMake 3.20+
- **GPU Architecture**: Compute Capability 7.5 (RTX/Turing)
- **Thư viện**: CUDA Runtime

---

## II. CẤU TRÚC DỰ ÁN

```
autoencoder-cuda/
├── CMakeLists.txt                 # Build configuration
├── config.yaml                    # Configuration file
├── train.cpp                      # Entry point for training
├── encode.cpp                     # Entry point for inference
├── draw.ipynb                     # Jupyter notebook for visualization
├── header/                        # Header files (.hpp, .cuh)
│   ├── autoencoder.hpp           # Autoencoder class definitions
│   ├── trainer.hpp               # Trainer class
│   ├── tensor.hpp                # Tensor data structure
│   ├── layer.hpp                 # Layer classes
│   ├── dataset.hpp               # DataLoader class
│   ├── kernel.cuh                # CUDA kernel declarations
│   └── README.md
├── source/                        # Implementation files
│   ├── tensor.cpp                # Tensor CPU implementation
│   ├── tensor.cu                 # Tensor GPU implementation
│   ├── dataset.cpp               # DataLoader implementation
│   ├── trainer.cu                # Trainer implementation (CUDA)
│   ├── autoencoder/
│   │   ├── base.cpp              # Base autoencoder class
│   │   ├── cpu.cpp               # CPU inference implementation
│   │   └── gpu.cpp               # GPU inference implementation
│   ├── layer/
│   │   ├── layer.cpp             # Base layer class
│   │   ├── conv.cpp              # Convolution (CPU)
│   │   ├── conv.cu               # Convolution (GPU)
│   │   ├── relu.cpp              # ReLU activation (CPU)
│   │   ├── relu.cu               # ReLU activation (GPU)
│   │   ├── maxpool.cpp           # Max pooling (CPU)
│   │   ├── maxpool.cu            # Max pooling (GPU)
│   │   ├── upsample.cpp          # Upsampling (CPU)
│   │   └── upsample.cu           # Upsampling (GPU)
│   └── optimizer/
│       └── sgd.cu                # SGD optimizer (GPU)
├── Build/                         # CMake build output
├── model/                         # Trained models
│   └── checkpoint_gpu_full_7h.dat
└── cifar-10-batches-bin/         # Dataset files
```

---

## III. KIẾN TRÚC MÔ HÌNH

### 3.1 Kiến Trúc Tổng Quan

```
Input (3×32×32)
    ↓
Encoder:
    Conv2D (3→256, k=3, s=1, p=1) + ReLU + MaxPool2D (2×2)
    ↓ (256×16×16)
    Conv2D (256→128, k=3, s=1, p=1) + ReLU + MaxPool2D (2×2)
    ↓ (128×8×8) ← Latent Representation
    
Decoder:
    Conv2D (128→128, k=3, s=1, p=1) + ReLU + UpSample2D (2×)
    ↓ (128×16×16)
    Conv2D (128→256, k=3, s=1, p=1) + ReLU + UpSample2D (2×)
    ↓ (256×32×32)
    Conv2D (256→3, k=3, s=1, p=1)
    ↓
Output (3×32×32)
```

### 3.2 Các Lớp

#### **Conv2D - Tích Chập 2D**
- **Tham số**:
  - `in_channels`: số kênh đầu vào
  - `out_channels`: số kênh đầu ra
  - `kernel_size`: kích thước kernel
  - `stride`: bước dịch chuyển
  - `padding`: độ padding
- **Khởi tạo trọng số**: Kaiming Uniform
- **Hỗ trợ**: CPU (`forward_cpu`) và GPU (`forward_gpu`)

#### **ReLU - Hàm Kích Hoạt**
- Công thức: `f(x) = max(0, x)`
- **Hỗ trợ**: CPU và GPU

#### **MaxPool2D - Pooling Cực Đại**
- **Tham số**: 
  - `pool_size`: kích thước vùng pooling (thường 2×2)
  - `stride`: bước dịch chuyển
- **Giảm kích thước**: từ H×W → H/stride × W/stride

#### **UpSample2D - Lấy Mẫu Lên**
- **Tham số**: `scale_factor` (thường 2)
- **Phương pháp**: Lặp lại pixel (Nearest Neighbor)
- **Công dụng**: Tăng kích thước đặc trưng trong decoder

#### **SGD Optimizer**
- **CUDA Kernel**: `sgd_update_kernel`
- **Công thức**: `params[i] -= lr * grads[i]`
- **Thực thi trên**: GPU device

---

## IV. CÁC LỚP CHÍNH

### 4.1 Tensor Class

**File**: [header/tensor.hpp](header/tensor.hpp), [source/tensor.cpp](source/tensor.cpp), [source/tensor.cu](source/tensor.cu)

**Mục đích**: Đại diện cho mảng dữ liệu 4D (N×C×H×W)

**Thuộc tính**:
```cpp
std::vector<float> host_data;    // Dữ liệu CPU
float* device_data;              // Con trỏ GPU (CUDA memory)
int N, C, H, W;                  // Dimensions
bool on_gpu;                      // Trạng thái thiết bị
```

**Phương thức chính**:
- `resize(n, c, h, w)`: Tạo lại kích thước
- `operator()(n, c, h, w)`: Truy cập phần tử
- `to_gpu()` / `to_cpu()`: Chuyển đổi thiết bị
- `data()`: Lấy con trỏ đến dữ liệu
- `index(n, c, h, w)`: Tính chỉ số tuyến tính

### 4.2 Layer Class

**File**: [header/layer.hpp](header/layer.hpp), [source/layer/](source/layer/)

**Kiến trúc**:
```cpp
class Layer {
    protected:
        Tensor weights, biases;
        Tensor grad_weights, grad_biases, cached_input;
    public:
        virtual Tensor forward_cpu(const Tensor&) = 0;
        virtual Tensor forward_gpu(const Tensor&) = 0;
        virtual Tensor backward_cpu(const Tensor&) = 0;
        virtual Tensor backward_gpu(const Tensor&) = 0;
        void update(float learning_rate);
        void to_gpu();
};
```

**Các lớp dẫn xuất**:
- `Conv2D`: Tích chập 2D
- `ReLU`: Hàm kích hoạt
- `MaxPool2D`: Pooling
- `UpSample2D`: Upsampling

### 4.3 Autoencoder Class

**File**: [header/autoencoder.hpp](header/autoencoder.hpp), [source/autoencoder/base.cpp](source/autoencoder/base.cpp)

**Cấu trúc**:
```cpp
namespace Autoencoder {
    class Base {
        protected:
            std::vector<std::unique_ptr<Layer>> layers;
            int encode_layer;  // Điểm chia encoder/decoder
        public:
            virtual void build();
            Tensor forward(const Tensor&);    // Full: encode + decode
            Tensor encode(const Tensor&);     // Encoder only
            Tensor decode(const Tensor&);     // Decoder only
            void backward(const Tensor&);     // Backpropagation
            virtual void update(float lr) = 0;
            void save_model(const string&);
            void load_model(const string&);
    };
    
    class CPU : public Base { /* CPU implementation */ };
    class GPU : public Base { /* GPU implementation */ };
}
```

**Phương thức chính**:
1. `build()`: Khởi tạo kiến trúc mô hình
2. `forward()`: Forward pass (encode + decode)
3. `encode()`: Forward pass chỉ encoder → latent vector
4. `decode()`: Forward pass chỉ decoder → reconstruct
5. `backward()`: Backward pass cho backpropagation
6. `update()`: Cập nhật trọng số (abstract, triển khai riêng cho CPU/GPU)
7. `save_model()` / `load_model()`: Lưu/tải checkpoint

### 4.4 DataLoader Class

**File**: [header/dataset.hpp](header/dataset.hpp), [source/dataset.cpp](source/dataset.cpp)

**Mục đích**: Tải và xử lý CIFAR-10 dataset

**Phương thức**:
- `load_data()`: Đọc file .bin
- `normalize_data()`: Chuẩn hóa [0, 255] → [0, 1]
- `shuffle_data()`: Xáo trộn dữ liệu huấn luyện
- `get_batch(idx, size)`: Lấy batch dữ liệu

### 4.5 Trainer Class

**File**: [header/trainer.hpp](header/trainer.hpp), [source/trainer.cu](source/trainer.cu)

**Mục đích**: Quản lý quá trình huấn luyện

**Phương thức chính**:
- `train()`: Vòng huấn luyện chính
- `train_one_epoch()`: Huấn luyện 1 epoch
- `compute_loss()`: Tính MSE loss
- `save_checkpoint()` / `load_checkpoint()`: Lưu/tải trạng thái

**Cấu hình**:
```yaml
epochs: 100
batch_size: 64
learning_rate: 0.001
```

---

## V. CÁC KERNEL CUDA

### 5.1 MSE Loss Kernel

**File**: [source/trainer.cu](source/trainer.cu)

```cuda
__global__ void mse_loss_kernel(
    const float *output,
    const float *target,
    float *loss,
    int N)
```

**Chức năng**: Tính Mean Squared Error trên GPU
- Mỗi thread tính `(output[i] - target[i])^2`
- Sử dụng shared memory để reduce kết quả

### 5.2 MSE Gradient Kernel

```cuda
__global__ void mse_grad_kernel(
    const float *output,
    const float *input,
    float *grad,
    int N,
    float scale)
```

**Chức năng**: Tính gradient của MSE loss
- `grad[i] = scale * (output[i] - input[i])`

### 5.3 SGD Update Kernel

**File**: [header/kernel.cuh](header/kernel.cuh), [source/optimizer/sgd.cu](source/optimizer/sgd.cu)

```cuda
__global__ void sgd_update_kernel(
    float* params,
    const float* grads,
    float lr,
    int size)
```

**Chức năng**: Cập nhật tham số sử dụng SGD
- `params[i] -= lr * grads[i]`

### 5.4 Các Kernel Layer

#### **Conv2D Kernel**
- Forward GPU: Convolution 2D trên GPU
- Backward GPU: Gradient w.r.t. input và parameters

#### **ReLU Kernel**
- Forward: `output = max(0, input)`
- Backward: `grad_input = grad_output * (input > 0)`

#### **MaxPool2D Kernel**
- Forward: Tìm max trong vùng pool size × stride
- Backward: Gradient flow chỉ đến position của max

#### **UpSample2D Kernel**
- Forward: Lặp lại mỗi pixel scale_factor lần
- Backward: Cộng gradient từ các pixel lặp lại

---

## VI. QUY TRÌNH HỖ TRỢ

### 6.1 Quá Trình Huấn Luyện

```
1. Khởi tạo DataLoader
   ├── Tải CIFAR-10 file .bin
   ├── Chuẩn hóa [0, 255] → [0, 1]
   └── Xáo trộn dữ liệu

2. Khởi tạo Autoencoder (CPU hoặc GPU)
   └── Gọi build() để xây dựng layers

3. Khởi tạo Trainer
   └── Tải config.yaml (epochs, batch_size, learning_rate)

4. Vòng lặp huấn luyện:
   for epoch in 1..epochs:
       for batch in data:
           a) Forward pass:
              - x_encode = encoder(input)
              - x_decode = decoder(x_encode)
           
           b) Tính loss:
              - MSE(x_decode, input)
           
           c) Backward pass:
              - grad_loss = ∇MSE
              - grad_decode = backward_decoder(grad_loss)
              - grad_encode = backward_encoder(grad_decode)
           
           d) Cập nhật tham số:
              - params -= lr * gradients
       
       e) Lưu log: epoch, loss, time_ms
   
5. Lưu checkpoint
```

### 6.2 Quá Trình Inference (Encode)

```
1. Tải checkpoint từ file
   └── load_model("checkpoint.dat")

2. Lặp từng batch:
   a) Lấy batch dữ liệu
      input = data_loader.get_batch(idx, batch_size)
   
   b) Forward pass (encode only):
      z = model.encode(input)
   
   c) In kết quả
      In 180 giá trị đầu tiên của z
```

### 6.3 Quá Trình Inference (Encode + Decode)

```
1. Full forward pass:
   output = model.forward(input)
   
2. Có thể so sánh input vs output để kiểm tra chất lượng reconstruction
```

---

## VII. BIÊN DỊCH VÀ CHẠY

### 7.1 Yêu Cầu

- Windows 10/11 hoặc Linux
- NVIDIA GPU (Compute Capability ≥ 7.5)
- CUDA Toolkit 11.0+
- CMake 3.20+
- C++ 17 compiler

### 7.2 Biên Dịch

```bash
mkdir -p build
cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### 7.3 Chạy Huấn Luyện

```bash
./Build/Release/train
```

**Kết quả**: Tạo file `training_log.csv` với các cột:
- `epoch`: Số epoch (1-indexed)
- `loss`: MSE loss trung bình
- `time_ms`: Thời gian epoch (milliseconds)

### 7.4 Chạy Inference

```bash
./Build/Release/encode
```

**Kết quả**: In ra 180 giá trị đầu tiên của latent vector cho 3 batch

---

## VIII. CẤU HÌNH

**File**: [config.yaml](config.yaml)

```yaml
epochs: 100              # Số epoch huấn luyện
batch_size: 64          # Kích thước batch
learning_rate: 0.001    # Tỉ lệ học
```

---

## IX. ĐỊNH DẠNG FILE MÔ HÌNH

### Magic Header
```
uint32_t magic = 0x41455631  // "AEV1"
uint32_t layer_count         // Số layers
```

### Per-Layer Data
```
For each layer:
    uint32_t weight_size
    uint32_t bias_size
    float[weight_size] weights
    float[bias_size] biases
```

---

## X. TỐI ƯU HÓA GPU

### 10.1 Cấu Hình CUDA

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75)  # RTX/Turing (7.5)
target_compile_options(train PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: -O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>: -O3>
)
```

### 10.2 Chiến Lược Tối Ưu Hóa

1. **Kernel Tuning**:
   - 256 threads per block cho loss kernel
   - Shared memory reduction cho tính toán global

2. **Memory Management**:
   - `cudaMalloc()` / `cudaFree()` cho GPU buffers
   - `cudaMemcpy()` chỉ khi cần thiết
   - `to_gpu()` / `to_cpu()` cho chuyển đổi dữ liệu

3. **Synchronization**:
   - `cudaDeviceSynchronize()` sau các kernel
   - Lộ trình host-device được tối ưu hóa

---

## XI. ENTRY POINTS

### 11.1 train.cpp

**Mục đích**: Huấn luyện mô hình

**Luồng**:
```cpp
1. Tạo model (CPU)
2. Tạo data loader (CIFAR-10)
3. Tạo trainer
4. Tải checkpoint (optional)
5. Gọi trainer.train()
6. Lưu checkpoint
```

### 11.2 encode.cpp

**Mục đích**: Inference sử dụng checkpoint

**Luồng**:
```cpp
1. Tạo model (CPU)
2. Tạo data loader
3. Tạo trainer
4. Tải checkpoint
5. Lặp 3 batches:
   - Lấy batch dữ liệu
   - Forward pass (inference)
   - In 180 giá trị đầu tiên
6. Cleanup
```

---

## XII. ĐIỂM KHÁC BIỆT CPU vs GPU

| Khía Cạnh | CPU | GPU |
|-----------|-----|-----|
| Class | `Autoencoder::CPU` | `Autoencoder::GPU` |
| File | `source/autoencoder/cpu.cpp` | `source/autoencoder/gpu.cpp` |
| Batch Size | 32 (tối ưu cho bộ nhớ) | 64 (hiệu suất tốt hơn) |
| Forward | `forward_cpu()` | `forward_gpu()` với CUDA |
| Backward | `backward_cpu()` | `backward_gpu()` với CUDA |
| Update | CPU-side SGD | `sgd_update_kernel()` |
| Memory | Host (RAM) | Device (VRAM) |

---

## XIII. FLOW BIỂU ĐỒ

### Forward Pass (Full Autoencoder)

```
Input (3×32×32)
    ↓
Conv2D (3→256, 3×3, p=1)
    ↓
ReLU
    ↓
MaxPool2D (2×2)
    ↓ (256×16×16)
Conv2D (256→128, 3×3, p=1)
    ↓
ReLU
    ↓
MaxPool2D (2×2)
    ↓ (128×8×8) [Latent]
Conv2D (128→128, 3×3, p=1)
    ↓
ReLU
    ↓
UpSample2D (2×)
    ↓ (128×16×16)
Conv2D (128→256, 3×3, p=1)
    ↓
ReLU
    ↓
UpSample2D (2×)
    ↓ (256×32×32)
Conv2D (256→3, 3×3, p=1)
    ↓
Output (3×32×32)
```

### Backward Pass

```
Loss Gradient (3×32×32)
    ↓ backward()
Backward Conv2D (256→3)
    ↓
Backward UpSample2D
    ↓
Backward ReLU
    ↓
Backward Conv2D (128→256)
    ↓
Backward UpSample2D
    ↓
Backward ReLU
    ↓
Backward Conv2D (128→128)
    ↓
Backward MaxPool2D
    ↓
Backward ReLU
    ↓
Backward Conv2D (256→128)
    ↓
Backward MaxPool2D
    ↓
Backward ReLU
    ↓
Backward Conv2D (3→256)
    ↓
Gradients (∇W, ∇b) cho tất cả layers
```

---

## XIV. CÔNG THỨC TOÁN HỌC

### MSE Loss

$$L = \frac{1}{N} \sum_{i=0}^{N-1} (y_i - \hat{y}_i)^2$$

Trong đó:
- $y_i$: output của mô hình
- $\hat{y}_i$: target (input gốc)
- $N$: số phần tử

### Conv2D Forward

$$\text{out}[n,oc,oh,ow] = \sum_{ic=0}^{C_{in}-1} \sum_{kh=0}^{K-1} \sum_{kw=0}^{K-1} W[oc,ic,kh,kw] \times \text{in}[n,ic,oh \times s + kh - p, ow \times s + kw - p] + b[oc]$$

Trong đó:
- $K$: kernel size
- $s$: stride
- $p$: padding
- $W$: weights, $b$: biases

### ReLU

$$f(x) = \max(0, x)$$

### MaxPool2D Forward

$$\text{out}[n,c,oh,ow] = \max_{kh,kw} \text{in}[n,c,oh \times s + kh, ow \times s + kw]$$

### SGD Update

$$w_t = w_{t-1} - \eta \cdot \nabla L(w_{t-1})$$

Trong đó:
- $\eta$: learning rate
- $\nabla L$: gradient của loss

---

## XV. CÁCH SỬ DỤNG NOTEBOOK

**File**: [draw.ipynb](draw.ipynb)

Jupyter notebook cho visualizing results từ `training_log.csv`:
- Plot loss vs epoch
- Thống kê thời gian huấn luyện
- Visualization của reconstructed images (nếu có)

---

## XVI. GỠ LỖI

### Debug Build

```bash
cmake --build . --config Debug
./Build/Debug/debug
```

**Cấu hình**: Tắt tối ưu hóa (-O0), bật debug symbols (-g)

### Kiểm Tra Lỗi CUDA

Sử dụng `CHECK()` macro từ [kernel.cuh](header/kernel.cuh):

```cpp
CHECK(cudaMalloc(...));  // Tự động báo lỗi CUDA
```

### So Sánh CPU vs GPU

Sử dụng hàm `compare_tensor()` trong [encode.cpp](encode.cpp):

```cpp
compare_tensor(z_cpu, z_gpu, "encoding");
// In ra: mean difference, max difference
```

---

## XVII. HẠNG GIỚI

### Hiểu Biết Hiện Tại

✅ Kiến trúc Autoencoder hoàn toàn  
✅ Hỗ trợ CPU và GPU  
✅ CIFAR-10 dataset integration  
✅ Checkpoint save/load  
✅ MSE loss và backpropagation  

### Khả Năng Mở Rộng

🔹 Thêm các activation khác (Sigmoid, Tanh, ...)  
🔹 Thêm Batch Normalization layers  
🔹 Thêm Dropout regularization  
🔹 Thêm các optimizer khác (Adam, Momentum, ...)  
🔹 Variational Autoencoder (VAE)  
🔹 Denoising Autoencoder  

---

## XVIII. TÓM TẮT CÁC FILE

| File | Loại | Mô Tả |
|------|------|-------|
| [header/autoencoder.hpp](header/autoencoder.hpp) | Header | Định nghĩa Autoencoder Base/CPU/GPU |
| [header/trainer.hpp](header/trainer.hpp) | Header | Định nghĩa Trainer |
| [header/tensor.hpp](header/tensor.hpp) | Header | Định nghĩa Tensor |
| [header/layer.hpp](header/layer.hpp) | Header | Định nghĩa Layer classes |
| [header/dataset.hpp](header/dataset.hpp) | Header | Định nghĩa DataLoader |
| [header/kernel.cuh](header/kernel.cuh) | CUDA Header | CUDA kernel declarations |
| [train.cpp](train.cpp) | Main | Entry point huấn luyện |
| [encode.cpp](encode.cpp) | Main | Entry point inference |
| [source/tensor.cpp](source/tensor.cpp) | Source | Tensor CPU implementation |
| [source/tensor.cu](source/tensor.cu) | CUDA | Tensor GPU implementation |
| [source/dataset.cpp](source/dataset.cpp) | Source | DataLoader implementation |
| [source/trainer.cu](source/trainer.cu) | CUDA | Trainer (loss, training loop) |
| [source/autoencoder/base.cpp](source/autoencoder/base.cpp) | Source | Autoencoder base class |
| [source/autoencoder/cpu.cpp](source/autoencoder/cpu.cpp) | Source | Autoencoder CPU |
| [source/autoencoder/gpu.cpp](source/autoencoder/gpu.cpp) | Source | Autoencoder GPU |
| [source/layer/conv.cpp](source/layer/conv.cpp) | Source | Conv2D CPU |
| [source/layer/conv.cu](source/layer/conv.cu) | CUDA | Conv2D GPU |
| [source/layer/relu.cpp](source/layer/relu.cpp) | Source | ReLU CPU |
| [source/layer/relu.cu](source/layer/relu.cu) | CUDA | ReLU GPU |
| [source/layer/maxpool.cpp](source/layer/maxpool.cpp) | Source | MaxPool2D CPU |
| [source/layer/maxpool.cu](source/layer/maxpool.cu) | CUDA | MaxPool2D GPU |
| [source/layer/upsample.cpp](source/layer/upsample.cpp) | Source | UpSample2D CPU |
| [source/layer/upsample.cu](source/layer/upsample.cu) | CUDA | UpSample2D GPU |
| [source/optimizer/sgd.cu](source/optimizer/sgd.cu) | CUDA | SGD optimizer |
| [config.yaml](config.yaml) | Config | Training configuration |
| [CMakeLists.txt](CMakeLists.txt) | Build | CMake build script |
| [draw.ipynb](draw.ipynb) | Notebook | Visualization |
| [README.md](README.md) | Doc | Project overview |

---

## XIX. KẾT LUẬN

Dự án **autoencoder-cuda** là một triển khai hoàn chỉnh của Convolutional Autoencoder với hỗ trợ CPU và GPU. Nó khám phá các khía cạnh chính của deep learning:

1. **Mô hình**: Autoencoder với encoder-decoder symmetry
2. **Dữ liệu**: CIFAR-10 dataset (ảnh 32×32)
3. **Tối ưu hóa**: Sử dụng CUDA để tăng tốc độ tính toán
4. **Framework**: CMake + C++ + CUDA
5. **Khả năng**: Huấn luyện, inference, checkpoint management

Dự án có thể được mở rộng với các variational autoencoders, dense autoencoders, hoặc các mô hình phức tạp khác.
