#include <ctgmath>
#include <iostream>

#include "./Matrix.hpp"

#ifndef GAUSS_SEIDEL_HPP
#define GAUSS_SEIDEL_HPP

namespace GaussSeidel {
	using Matrix::Matrix;

	template<typename T>
	constexpr T calculateJump(T val) {
		return 1 / (val - 1);
	}

	template<
			template<typename> class A,
			template<typename> class B,
			template<typename> class M, typename T
	>
	class GaussSeidel {
	private:
		static const A<T> a;
		static const B<T> b;

		M<T> &matrix;
		const T jumpRow;
		const T jumpColumn;

		constexpr const T w (size_t row, size_t column) {
			const T temp = oij(row, column);
			return 2 / (1 + sqrt(1 - (temp * temp)));
		}

		constexpr const T oij (size_t row, size_t column) {
			return 2 * (
					(sqrt(east(row, column) * west(row, column)) * cos(jumpColumn * M_PI))
			       +
					(sqrt(north(row, column) * south(row, column)) * cos(jumpRow * M_PI))
			);
		}

		constexpr const T east(size_t row, size_t column) {
			return (2 - (jumpColumn * a(jumpColumn * column, jumpRow * row)))
			       / (4 * (1 + ((jumpColumn * jumpColumn) / (jumpRow * jumpRow))));
		}

		constexpr const T west(size_t row, size_t column) {
			return (2 + (jumpColumn * a(jumpColumn * column, jumpRow * row)))
			       / (4 * (1 + ((jumpColumn * jumpColumn) / (jumpRow * jumpRow))));
		}

		constexpr const T south(size_t row, size_t column) {
			return (2 + (jumpRow * b(jumpColumn * column, jumpRow * row)))
			       / (4 * (1 + ((jumpRow * jumpRow) / (jumpColumn * jumpColumn))));
		}

		constexpr const T north(size_t row, size_t column) {
			return (2 - (jumpRow * b(jumpColumn * column, jumpRow * row)))
			       / (4 * (1 + ((jumpRow * jumpRow) / (jumpColumn * jumpColumn))));
		}

	public:
		CUDA_DEVICE constexpr GaussSeidel(M<T> &m)
				: matrix(m), jumpRow(calculateJump<T>(m.rowLength)),
				  jumpColumn(calculateJump<T>(m.columnLength)) {}

		CUDA_DEVICE constexpr void updateElement(size_t row, size_t column) {
			T localW = w(row, column);

			matrix(row, column) = (1 - localW) * matrix(row, column) +
			                      localW * ((north(row, column) * matrix(row - 1, column)) +
			                                (south(row, column) * matrix(row + 1, column)) +
			                                (west(row, column) * matrix(row, column - 1)) +
			                                (east(row, column) * matrix(row, column + 1)));

//			std::cout << std::endl << "Row: " << row << std::endl << "Column: "
//			          << column << std::endl << "JumpRow: " << jumpRow << std::endl
//			          << "JumpColumn: " << jumpColumn << std::endl << "A: "
//			          << a(jumpColumn * column, jumpRow * row) << std::endl << "B: "
//			          << b(jumpColumn * column, jumpRow * row) << std::endl
//			          << "north: " << north(row, column) << " * "
//			          << matrix(row - 1, column) << std::endl << "south: "
//			          << south(row, column) << " * " << matrix(row + 1, column)
//			          << std::endl << "west: " << west(row, column) << " * "
//			          << matrix(row, column - 1) << std::endl << "east: "
//			          << east(row, column) << " * " << matrix(row, column + 1)
//			          << std::endl << matrix << std::endl;
		}

		constexpr void step() {
			size_t rowIdx = 0, colIdx = 0, rowStart = 0;

			for (colIdx = 1, rowStart = 2; colIdx < (matrix.columns - 1);
			     rowStart = (rowStart % 2) + 1, ++colIdx)
				for (rowIdx = rowStart; rowIdx < (matrix.rows - 1); rowIdx += 2)
					updateElement(rowIdx, colIdx);

			for (colIdx = 1, rowStart = 1; colIdx < (matrix.columns - 1);
			     rowStart = (rowStart % 2) + 1, ++colIdx)
				for (rowIdx = rowStart; rowIdx < (matrix.rows - 1); rowIdx += 2)
					updateElement(rowIdx, colIdx);
//
//			for (rowIdx = 1; rowIdx < (matrix.rows - 1); ++rowIdx)
//				for (colIdx = 1; colIdx < (matrix.columns - 1); ++colIdx)
//					updateElement(rowIdx, colIdx);

//			for (rowIdx = 1, colIdx = 1; rowIdx < matrix.rows-1; ++rowIdx, colIdx = 1)
//				for (; colIdx < matrix.columns - 1; colIdx += 2)
//					updateElement(rowIdx, colIdx);
//
//			for (rowIdx = 1, colIdx = 2; rowIdx < matrix.rows -1; ++rowIdx, colIdx = 2)
//				for (; colIdx < matrix.columns - 1; colIdx += 2)
//					updateElement(rowIdx, colIdx);
		}
	};

	template<
			template<typename> class A,
			template<typename> class B,
			template<typename> class M, typename T
	>
	const A<T> GaussSeidel<A, B, M, T>::a;

	template<
			template<typename> class A,
			template<typename> class B,
			template<typename> class M, typename T
	>
	const B<T> GaussSeidel<A, B, M, T>::b;
}

#endif