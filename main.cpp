#include <string>
#include "include/cudaUtils.h"
#include "include/Matrix.hpp"
#include "include/GaussSeidel.hpp"

#define THREAD_BLOCK_SIZE 16
typedef float dataType;
typedef Matrix::Matrix<dataType> matrixType;
typedef GaussSeidel::GaussSeidel<A, B, matrixType, dataType> gaussType;

template <typename T>
class A {
public:
	CUDA_DEVICE A(){}

	CUDA_DEVICE const T operator() (T x, T y) const {
		return 500 * x * (1 - x) * (0.5 - y);
	}
};

template <typename T>
class B {
public:
	CUDA_DEVICE B(){}

	CUDA_DEVICE const T operator() (T x, T y) const {
		return 500 * y * (1 - y) * (x - 0.5);
	}
};

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

//#ifdef CUDA_ENVIRONMENT
KERNEL void setup(dataType *data, matrixType *matrix, gaussType *gS,
                  size_t rows, size_t columns, size_t north, size_t south,
                  size_t west, size_t east
) {
	*matrix = matrixType(data, rows, columns);
	*gS = gaussType(m);

	matrixSetup(m, north, south, west, east);
};

KERNEL void deviceStep(gaussType *gS, matrixType *matrix, size_t laps) {
	size_t threadIdX = blockIdx.x * blockDim.x + threadIdx.x;
	size_t threadIdY = blockIdx.y * blockDim.y + threadIdx.y;
	bool readThread = false;
	bool blueThread = false;
	if (threadIdX != 0 && threadIdX < matrix->rows - 1 && threadIdY != 0 &&
			threadIdY < matrix->columns - 1
	) {
		if ((threadIdX + threadIdY) % 2 === 0) blueThread = true;
		else readThread = true;
	}

	while (--laps > -1) {
		if (readThread) {
			gS->updateElement(threadIdY, threadIdX);
		}

		__synchronize();

		if (blueThread) {
			gS->updateElement(threadIdY, threadIdX);
		}

		__synchronize();
	}
};
//#endif

int main (int argc, char** argv) {
	size_t columns, rows, laps, north, south, west, east, argIdx;

	if (argc == 7) {
		rows = atoi(argv[0]);
		columns = atoi(argv[1]);
	} else if (argc == 6) {
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

#ifdef CUDA_ENVIRONMENT

	dataType *deviceMatrixData;
	gaussType *deviceGS;
	matrixType *deviceMatrix;

	CUDA_ERROR_CHECK(cudaMalloc(
		(void**) &deviceMatrixData, rows * columns * sizeof(dataType)
	))
	CUDA_ERROR_CHECK(cudaMalloc((void**) &deviceGS, sizeof(gaussType)))
	CUDA_ERROR_CHECK(cudaMalloc((void**) &deviceMatrix, sizeof(matrixType)))

	setup<<<1,1>>>(deviceMatrixData, deviceMatrix, deviceGS, rows, columns,
			north, south, west, east
	);

	dim3 threadNum(THREAD_BLOCK_SIZE, THREAD_BLOCK_SIZE);
	dim3 blockNum(
		(rows + threadNum.x - 1) / threadNum.x,
		(columns + threadNum.y -1) / threadNum.y
	);

	deviceStep<<<blockNum, threadNum>>>(deviceGS);

	cudaDeviceSynchronize();

#endif

	matrixType m(rows, columns);
	gaussType gS(m);
	matrixSetup(m, north, south, west, east);

	for (size_t i = 0; i < laps; ++i) deviceGS.step();

#endif

	return 0;
}