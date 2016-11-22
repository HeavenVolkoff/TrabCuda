#include <string>
#include <iostream>
#include "./Matrix.hpp"
#include "./GaussSeidel.hpp"

template <typename T>
class A {
public:
	const T operator() (T x, T y) const {
		return 500 * x * (1 - x) * (0.5 - y);
	}
};

template <typename T>
class B {
public:
	const T operator() (T x, T y) const {
		return 500 * y * (1 - y) * (x - 0.5);
	}
};

int main () {
	Matrix::Matrix<20, 20, double> m;
	auto gS = GaussSeidel::instantiate<A, B>(m);

	m.zero();
	m.fillRow(0, 5);
	m.fillRow(m.rows - 1, 5);
	// m.fillColumn(0, 0);
	m.fillColumn(m.columns - 1, 10);

	gS.step();

	std::cout << m;
}