#include <iostream>

#ifndef CUDA_UTILS_H
	#define CUDA_UTILS_H

	#ifdef __NVCC__

		inline
		void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
			if (code != cudaSuccess) {
				std::cerr << "Cuda Error: " << cudaGetErrorString(code) << std::endl <<
				"At File: " << file << " Line: " << line << std::endl;

				if (abort) exit(code);
			}
		}
		#define CUDA_ERROR_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

		#define CUDA_ENVIRONMENT
		#define KERNEL __global__
		#define CUDA_ANY __host__ __device__
		#define CUDA_HOST __host__
		#define CUDA_DEVICE __device__

	#else

		#define KERNEL
		#define CUDA_ANY
		#define CUDA_HOST
		#define CUDA_DEVICE

	#endif

#endif //CUDA_UTILS_H
