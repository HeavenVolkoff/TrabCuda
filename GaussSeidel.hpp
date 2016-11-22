#include "./Matrix.hpp"

#ifndef GAUSS_SEIDEL_HPP
#define GAUSS_SEIDEL_HPP

namespace GaussSeidel {
	using Matrix::Matrix;

	template <typename T>
	constexpr T calculateJump (T val) {
		return 1 / (val + 1);
	}

	template <
		template <typename> class A, template <typename> class B,
		template<size_t, size_t, typename> class M,
		size_t columnLength, size_t rowLength, typename T
	>
	class GaussSeidel {
	private:
		M<columnLength, rowLength, T> const &matrix;
		static constexpr T jumpRow = calculateJump<T>(columnLength);
		static constexpr T jumpColumn = calculateJump<T>(rowLength);
		static const A<T> a;
		static const B<T> b;

		static constexpr const T east(size_t row, size_t column) {
			return (2 - jumpColumn * a(jumpColumn * column, jumpRow * row))
				/ (4 * (1 + jumpColumn * jumpColumn / jumpRow * jumpRow));
		}

		static constexpr const T west(size_t row, size_t column) {
			return (2 + jumpColumn * a(jumpColumn * column, jumpRow * row))
				/ (4 * (1 + jumpColumn * jumpColumn / jumpRow * jumpRow));
		}

		static constexpr const T south(size_t row, size_t column) {
			return (2 + jumpRow * b(jumpColumn * column, jumpRow * row))
				/ (4 * (1 + jumpRow * jumpRow / jumpColumn * jumpColumn));
		}

		static constexpr const T north(size_t row, size_t column) {
			return (2 - jumpRow * b(jumpColumn * column, jumpRow * row))
				/ (4 * (1 + jumpRow * jumpRow / jumpColumn * jumpColumn));
		}

		static constexpr const T step(size_t row, size_t column) {
			matrix(row, column) =
				north(row, column) * matrix(row + 1, column) +
				south(row, column) * matrix(row - 1, column) +
				west(row, column) * matrix(row, column - 1) +
				east(row, column) * matrix(row, column + 1)
		}

	public:
		constexpr GaussSeidel(M<columnLength, rowLength, T> const &m)
			: matrix(m) {}
	};

	template <
		template <typename> class A, template <typename> class B,
		template<size_t, size_t, typename> class M,
		size_t cL, size_t rL, typename T
	>
	constexpr GaussSeidel<A, B, M, cL, rL, T> instantiate(
		M<cL, rL, T> const &m
	) {
		return GaussSeidel<A, B, M, cL, rL, T>(m);
	}
}

#endif