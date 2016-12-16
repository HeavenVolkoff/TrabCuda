#include "./cudaUtils.hpp"

#ifndef EQUATION_HPP
#define EQUATION_HPP

namespace equation {
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
}

#endif