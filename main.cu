#include <new>
#include <ctime>

#include "include/Matrix.hpp"
#include "include/cudaUtils.h"
#include "include/GaussSeidel.hpp"

#define THREAD_BLOCK_SIZE 16

template <typename T>
class A {
public:
    CUDA_DEVICE A(){}

    CUDA_DEVICE const T operator() (T x, T y) const {
        return 500.0f * x * (1.0f - x) * (0.5f - y);
    }
};

template <typename T>
class B {
public:
    CUDA_DEVICE B(){}

    CUDA_DEVICE const T operator() (T x, T y) const {
        return 500.0f * y * (1.0f - y) * (x - 0.5f);
    }
};

typedef float dataType;
typedef Matrix::Matrix<dataType> matrixType;
typedef GaussSeidel::GaussSeidel<A, B, Matrix::Matrix, dataType> gaussType;

template <typename T>
CUDA_DEVICE inline void matrixSetup (
		Matrix::Matrix<T> &m, size_t north, size_t south, size_t west, size_t east
) {
	m.zero();
	m.fillColumn(0, west);
	m.fillColumn(m.columns - 1, east);
	m.fillRow(0, north);
	m.fillRow(m.rows - 1, south);
}

KERNEL void setup(dataType *data, void *matrixMemLocation,
	void *gSMemLocation, size_t rows, size_t columns, size_t north,
	size_t south, size_t west, size_t east
) {
	matrixType* matrix = new(matrixMemLocation) matrixType(rows, columns, data);
	gaussType* gS = new(gSMemLocation) gaussType(*matrix);

	matrixSetup(*matrix, north, south, west, east);
};

KERNEL void redDot (gaussType *gS, size_t rowLimit, size_t columnLimit) {
	size_t column = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	size_t row = ((blockIdx.y * blockDim.y) + threadIdx.y) * 2 + (column % 2 == 0 ? 1 : 2);

	if (column < rowLimit && row < columnLimit) gS->updateElement(row, column);
}

KERNEL void blueDot (gaussType *gS, size_t rowLimit, size_t columnLimit) {
	size_t column = (blockIdx.x * blockDim.x) + threadIdx.x + 1;
	size_t row = ((blockIdx.y * blockDim.y) + threadIdx.y) * 2 + (column % 2 == 0 ? 2 : 1);

	if (column < rowLimit && row < columnLimit) gS->updateElement(row, column);
}

int main (int argc, char** argv) {
	gaussType *deviceGS;
	matrixType *deviceMatrix;
	dataType *deviceMatrixData;
	size_t columns, rows, laps, north, south, west, east, matrixDataSize;
	char print;

	if (argc == 8) {
		rows = atoi(argv[0]);
		columns = atoi(argv[1]);
	} else if (argc == 7) {
		rows = columns = atoi(argv[1]);
	} else {
		std::cout << "USAGE: " << argv[0]
		          << " <size/rows> [columns] <loop count>"
		          << " Temperatures: <north> <east> <south> <west>"
		          << std::endl;
		exit(-1);
	}

	west = atoi(argv[--argc]);
	south = atoi(argv[--argc]);
	east = atoi(argv[--argc]);
	north = atoi(argv[--argc]);
	laps = atoi(argv[--argc]);

	matrixDataSize = rows * columns * sizeof(dataType);

	CUDA_ERROR_CHECK(cudaMalloc(
		(void**) &deviceMatrixData, matrixDataSize
	));
	CUDA_ERROR_CHECK(cudaMalloc((void**) &deviceGS, sizeof(gaussType)));
	CUDA_ERROR_CHECK(cudaMalloc((void**) &deviceMatrix, sizeof(matrixType)));

	setup<<<1,1>>>(deviceMatrixData, deviceMatrix, deviceGS, rows, columns,
			north, south, west, east
	);
	CUDA_ERROR_CHECK(cudaPeekAtLastError());

	dim3 threadNum(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	dim3 blockNum(
		(columns + threadNum.x - 3) / threadNum.x,
		(((rows - 1) / 2) + threadNum.y - 1) / threadNum.y
	);
	std::cout << "Block Thread Size: " << threadNum.x << " X " << threadNum.y <<
				std::endl << "Block Size: " << blockNum.x << " X " <<
				blockNum.y << std::endl;

	clock_t time = clock();
	while (laps > 0) {
		redDot<<<blockNum, threadNum>>>(deviceGS, rows - 1, columns - 1);
		CUDA_ERROR_CHECK(cudaPeekAtLastError());
		blueDot<<<blockNum, threadNum>>>(deviceGS, rows - 1, columns - 1);
		CUDA_ERROR_CHECK(cudaPeekAtLastError());
		--laps;
	}
	CUDA_ERROR_CHECK(cudaDeviceSynchronize());
	time = (clock() - time);

	std::cout << "Kernel time: " << time << " cycles" << std::endl;
	std::cout << "Print Matrix (y/N): ";
    std::cin >> print;

    if (print == 'y') {
    	dataType* hostMatrixData = (dataType*) malloc(matrixDataSize);
		CUDA_ERROR_CHECK(cudaMemcpy(hostMatrixData, deviceMatrixData,
			matrixDataSize, cudaMemcpyDeviceToHost
		));
		matrixType hostMatrix(rows, columns, hostMatrixData);
		std::cout << hostMatrix;
		free(hostMatrixData);
    }

    CUDA_ERROR_CHECK(cudaFree(deviceMatrixData));
    CUDA_ERROR_CHECK(cudaFree(deviceGS));
    CUDA_ERROR_CHECK(cudaFree(deviceMatrix));

	return 0;
}
