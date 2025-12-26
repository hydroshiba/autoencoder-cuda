#ifndef TIMER_CUH
#define TIMER_CUH

#include <cuda_runtime.h>
#include <string>
#include <map>
#include <utility>
#include <chrono>
#include <type_traits>

#include "device.hpp"

class Timer {
private:
	std::map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> gpu_timers;
	
	using Clock = std::chrono::high_resolution_clock;
	std::map<std::string, Clock::time_point> cpu_start_times;
	std::map<std::string, float> cpu_elapsed_ms;

public:
	Timer() {}

	~Timer() {
		for(auto& [name, events] : gpu_timers) {
			cudaEventDestroy(events.first);
			cudaEventDestroy(events.second);
		}
	}

	template <typename T>
	void start(const std::string& name, cudaStream_t stream = 0) {
		if constexpr (std::is_same_v<T, Device::CPU>) cpu_start_times[name] = Clock::now();
		else {
			if(gpu_timers.find(name) == gpu_timers.end()) {
				cudaEvent_t begin, end;
				cudaEventCreate(&begin);
				cudaEventCreate(&end);
				gpu_timers[name] = {begin, end};
			}
			cudaEventRecord(gpu_timers[name].first, stream);
		}
	}

	template <typename T>
	void stop(const std::string& name, cudaStream_t stream = 0) {
		if constexpr (std::is_same_v<T, Device::CPU>) {
			auto it = cpu_start_times.find(name);
			if(it != cpu_start_times.end()) {
				auto end_time = Clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - it->second);
				cpu_elapsed_ms[name] = duration.count() / 1000000.0f;
			}
		}
		else {
			auto it = gpu_timers.find(name);
			if(it != gpu_timers.end()) cudaEventRecord(it->second.second, stream);
		}
	}

	template <typename T>
	float elapsed(const std::string& name) {
		if constexpr (std::is_same_v<T, Device::CPU>) {
			auto it = cpu_elapsed_ms.find(name);
			if(it != cpu_elapsed_ms.end()) return it->second;
		}
		else {
			auto it = gpu_timers.find(name);
			if(it != gpu_timers.end()) {
				float milliseconds = 0.0f;
				cudaEventSynchronize(it->second.second);
				cudaEventElapsedTime(&milliseconds, it->second.first, it->second.second);
				return milliseconds;
			}
		}
		return -1.0f;
	}
};

#endif // TIMER_CUH