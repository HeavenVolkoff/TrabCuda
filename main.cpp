#include <string>
#include <iostream>
#include "./Matrix.hpp"
#include "./GaussSeidel.hpp"

template <typename T>
class A {
public:
	T operator() (T x, T y) { return 500 * x * (1 - x) * (0.5 - y); }
};

template <typename T>
class B {
public:
	T operator() (T x, T y) { return 500 * y * (1 - y) * (x - 0.5); }
};

int main () {
	Matrix::Matrix<10, 20, double> m;
	auto gS = GaussSeidel::instantiate<A, B>(m);

	m.zero();
	m.fillColumn(0, 1);
	m.fillColumn(m.columns - 1, 9);
	m.fillRow(0, 2);
	m.fillRow(m.rows - 1, 5);
	std::cout << m;
}