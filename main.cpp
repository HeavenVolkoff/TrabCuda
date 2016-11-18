#include "./Matrix.hpp"
#include <iostream>
#include <string>

inline double calculateJump (auto val) {
	return 1 / (static_cast<double>(val) + 1);
}

int main () {
	Matrix<10, 10, double> m;
	m.zero();
	m.fillColumn(0, 1);
	m.fillColumn(m.cols - 1, 9);
	m.fillRow(0, 2);
	m.fillRow(m.cols - 1, 5);
	std::cout << m;
}