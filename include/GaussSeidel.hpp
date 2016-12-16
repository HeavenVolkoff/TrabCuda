#include <ctgmath>

#include "./Matrix.hpp"
#include "./cudaUtils.hpp"

#ifndef GAUSS_SEIDEL_HPP
#define GAUSS_SEIDEL_HPP

namespace GaussSeidel {
	using Matrix::Matrix;

	template<typename T>
    CUDA_DEVICE constexpr T calculateJump(T val) {
		return 1.0f / (val - 1.0f);
	}

	template<
			template<typename> class A,
			template<typename> class B,
			template<typename> class M,
			typename T
	>
	class GaussSeidel {
	private:
		static const A<T> a;
		static const B<T> b;

		M<T> &matrix;
		const T jumpRow;
		const T jumpColumn;

    CUDA_DEVICE constexpr const T oij (size_t row, size_t column) const {
        return 2.0f * (
                (std::sqrt(east(row, column) * west(row, column)) * std::cos(jumpColumn * M_PI))
                +
                (std::sqrt(north(row, column) * south(row, column)) * std::cos(jumpRow * M_PI))
        );
    }

    CUDA_DEVICE constexpr const T w (size_t row, size_t column) const {
			return 2.0f / (1.0f + std::sqrt(1.0f - (std::pow(oij(row, column), 2))));
		}

    CUDA_DEVICE constexpr const T east(size_t row, size_t column) const {
			return (2.0f - (jumpColumn * a(jumpColumn * column, jumpRow * row)))
			       / (4.0f * (1.0f + ((jumpColumn * jumpColumn) / (jumpRow * jumpRow))));
		}

    CUDA_DEVICE constexpr const T west(size_t row, size_t column) const {
			return (2.0f + (jumpColumn * a(jumpColumn * column, jumpRow * row)))
			       / (4.0f * (1.0f + ((jumpColumn * jumpColumn) / (jumpRow * jumpRow))));
		}

    CUDA_DEVICE constexpr const T south(size_t row, size_t column) const {
			return (2.0f + (jumpRow * b(jumpColumn * column, jumpRow * row)))
			       / (4.0f * (1.0f + ((jumpRow * jumpRow) / (jumpColumn * jumpColumn))));
		}

    CUDA_DEVICE constexpr const T north(size_t row, size_t column) const {
			return (2.0f - (jumpRow * b(jumpColumn * column, jumpRow * row)))
			       / (4.0f * (1.0f + ((jumpRow * jumpRow) / (jumpColumn * jumpColumn))));
		}

	public:
		CUDA_DEVICE constexpr GaussSeidel(M<T> &m)
				: matrix(m), jumpRow(calculateJump<T>(m.rowLength)),
				  jumpColumn(calculateJump<T>(m.columnLength)) {}

		CUDA_DEVICE void updateElement(size_t row, size_t column) {
			T localW = w(row, column);
			matrix(row, column) = ((1.0f - localW) * matrix(row, column)) +
														(localW * (
															(north(row, column) * matrix(row + 1, column)) +
			                        (south(row, column) * matrix(row - 1, column)) +
			                        (west(row, column) * matrix(row, column - 1)) +
			                        (east(row, column) * matrix(row, column + 1))
			                      ));
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