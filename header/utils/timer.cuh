#include <cuda_runtime.h>
#include <string>
#include <map>
#include <utility>

class Timer {
private:
	std::map<std::string, std::pair<cudaEvent_t, cudaEvent_t>> timers;

public:
	Timer() {}

	~Timer() {
		for(auto& [name, events] : timers) {
			cudaEventDestroy(events.first);
			cudaEventDestroy(events.second);
		}
	}

	void start(const std::string& name, cudaStream_t stream = 0) {
		if(timers.find(name) == timers.end()) {
			cudaEvent_t begin, end;
			cudaEventCreate(&begin);
			cudaEventCreate(&end);
			timers[name] = {begin, end};
		}
		cudaEventRecord(timers[name].first, stream);
	}

	void stop(const std::string& name, cudaStream_t stream = 0) {
		auto it = timers.find(name);
		if(it != timers.end()) {
			cudaEventRecord(it->second.second, stream);
		}
	}

	void synchronize() {
		cudaDeviceSynchronize();
	}

	float elapsed(const std::string& name) {
		auto it = timers.find(name);
		if(it != timers.end()) {
			float elapsed = 0.0f;
			cudaEventElapsedTime(&elapsed, it->second.first, it->second.second);
			return elapsed;
		}
		return -1.0f;
	}
};