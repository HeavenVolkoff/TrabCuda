#include <string>
#include <iostream>
#include "./Matrix.hpp"

int main () {
	Matrix::Matrix<300, 200, double> m;
	m.zero();
	m.fillColumn(0, 1);
	m.fillColumn(m.columns - 1, 9);
	m.fillRow(0, 2);
	m.fillRow(m.rows - 1, 5);
	std::cout << m;
}