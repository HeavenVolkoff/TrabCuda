#include <iostream>

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __CUDACC__
inline
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
	if (code != cudaSuccess) {
		std::cerr << "Cuda Error at: " << cudaGetErrorString(code) << file << line
		          << sts::endl;

		if (abort) exit(code);
	}
}

#define CUDA_ENVIRONMENT
#define GLOBAL_CALLABLE __host__ __device__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define KERNEL __global__
#define CUDA_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#else
#define CUDA_CALLABLE
#define CUDA_DEVICE
#define CUDA_HOST
#define KERNEL
#endif

#endif //CUDA_UTILS_H
