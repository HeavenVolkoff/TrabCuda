#include <new>
#include <omp.h>
#include <chrono>

#include "include/Matrix.hpp"
#include "include/equation.hpp" 
#include "include/cudaUtils.hpp"
#include "include/GaussSeidel.hpp"

#define THREAD_BLOCK_SIZE 16

typedef float dataType;
typedef Matrix::Matrix<dataType> matrixType;
typedef GaussSeidel::GaussSeidel<equation::A, equation::B, Matrix::Matrix, dataType> gaussType;

template <typename T>
inline void matrixSetup (
		Matrix::Matrix<T> &m, size_t north, size_t south, size_t west, size_t east
) {
	m.zero();
	m.fillColumn(0, west);
	m.fillColumn(m.columns - 1, east);
	m.fillRow(0, north);
	m.fillRow(m.rows - 1, south);
}

int main (int argc, char** argv) {
	gaussType *deviceGS;
	matrixType *deviceMatrix;
	dataType *deviceMatrixData;
	size_t columns, rows, laps, north, south, west, east, rowIdx, colIdx, rowStart;
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

	matrixType matrix(rows, columns);
	matrixSetup(matrix, north, south, west, east);
	gaussType gS(matrix);

	omp_set_num_threads(omp_get_max_threads());

	auto start = std::chrono::steady_clock::now();
	while (laps > 0) {
		#pragma omp parallel private(rowStart)
		{ 
			rowStart = ((omp_get_thread_num() + 1) % 2) + 1;

			#pragma omp for private(colIdx, rowIdx) schedule(guided) collapse(2)
		    for (colIdx = 1; colIdx < (matrix.columns - 1); ++colIdx)
				for (rowIdx = rowStart; rowIdx < (matrix.rows - 1); rowIdx += 2) {
					gS.updateElement(rowIdx, colIdx);
					if (rowIdx >= (matrix.rows - 1)) rowStart = (rowStart % 2) + 1;
				}
		}

		#pragma omp parallel private(rowStart)
		{ 
			rowStart = ((omp_get_thread_num() + 2) % 2) + 1;

			#pragma omp for private(colIdx, rowIdx) schedule(guided) collapse(2)
		    for (colIdx = 1; colIdx < (matrix.columns - 1); ++colIdx)
				for (rowIdx = rowStart; rowIdx < (matrix.rows - 1); rowIdx += 2) {
					gS.updateElement(rowIdx, colIdx);
					if (rowIdx >= (matrix.rows - 1)) rowStart = (rowStart % 2) + 1;
				}
		}

		--laps;
	}
	std::chrono::duration<double, std::milli> duration = std::chrono::steady_clock::now() - start;

	std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
	std::cout << "Print Matrix (y/N): ";
    std::cin >> print;
    if (print == 'y') std::cout << matrix;

	return 0;
}
