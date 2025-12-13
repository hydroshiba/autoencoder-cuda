#ifndef RANDOM_CUH
#define RANDOM_CUH

#include <cmath>
#include <cstdint>

namespace Random {

__host__ __device__ inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

__host__ __device__ inline uint64_t splitmix64(uint64_t& x) {
	uint64_t z = (x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}

// XorShiro128+ PRNG
__host__ __device__ inline uint64_t xorshiro(uint64_t s[2]) {
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
	s[1] = rotl(s1, 37);

	return result;
}

__host__ __device__ inline float box_muller(uint64_t s[2]) {
	constexpr float norm = 1.0f / 18446744073709551616.0f; // 1 / 2^64
	
	float u1 = xorshiro(s) * norm;
	float u2 = xorshiro(s) * norm;

	if(u1 <= 1e-7f) u1 = 1e-7f;

	constexpr float pi = 3.14159265358979323846f;
	return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * pi * u2);
}

}

#endif // RANDOM_CUH