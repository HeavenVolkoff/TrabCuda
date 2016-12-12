#include <string>
#include <iostream>
#include "include/Matrix.hpp"
#include "include/GaussSeidel.hpp"

template <typename T>
class A {
public:
	A(){}

	const T operator() (T x, T y) const {
		return 500 * x * (1 - x) * (0.5 - y);
	}
};

template <typename T>
class B {
public:
	B(){}

	const T operator() (T x, T y) const {
		return 500 * y * (1 - y) * (x - 0.5);
	}
};

int main () {
	Matrix::Matrix<double> m(200, 200);
	auto gS = GaussSeidel::instantiate<A, B>(m);

	m.zero();
	m.fillColumn(m.columns - 1, 10);
	m.fillRow(0, 5);
	m.fillRow(m.rows - 1, 5);
	// m.fillColumn(0, 0);

	for (size_t i = 0; i < 50000; ++i) gS.step();

	std::cout << m;
}