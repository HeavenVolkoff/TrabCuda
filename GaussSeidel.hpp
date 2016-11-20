#include "./Matrix.hpp"

#ifndef GAUSS_SEIDEL_HPP
#define GAUSS_SEIDEL_HPP

namespace GaussSeidel {
	using Matrix::Matrix;

	inline double calculateJump (auto val) {
		return 1 / (static_cast<double>(val) + 1);
	}

	template <
		template <typename> class A, template <typename> class B,
		template<size_t, size_t, typename> class M,
		size_t columnLength, size_t rowLength, typename T
	>
	class GaussSeidel {
	private:
		M<columnLength, rowLength, T> const &matrix;
		A<T> a;
		B<T> b;

	public:
		GaussSeidel(M<columnLength, rowLength, T> const &m)
			: matrix(m), a(), b() {}
	};

	template <
		template <typename> class _A, template <typename> class _B,
		template<size_t, size_t, typename> class _M,
		size_t cL, size_t rL, typename _T
	>
	GaussSeidel<_A, _B, _M, cL, rL, _T> instantiate(_M<cL, rL, _T> const &m){
		return GaussSeidel<_A, _B, _M, cL, rL, _T>(m);
	}
}

#endif