#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <iostream>

#include "./cudaUtils.h"

#ifndef MATRIX_HPP
#define MATRIX_HPP

namespace Matrix {
	/**
	 * Matrix Class
	 * This code was heavily influenced by:
	 * https://raw.githubusercontent.com/VMML/vmmlib/master/vmmlib/matrix.hpp
	 */
	template<typename T = double>
	class Matrix {
	private:
		/** Members **/
		T* const array;

		/**
		 * Convert matrix to a string representation
		 * @return [Matrix's string representation]
		 */
		CUDA_HOST std::string toString() const {
			std::stringstream toString;

			toString << std::showpos << std::fixed;
			toString.precision(5);

			for (size_t rowIndex = 0; rowIndex < rows; ++rowIndex) {
				toString << '[' << at(rowIndex, 0);

				for (size_t colIndex = 1; colIndex < columns; ++colIndex) {
					toString << ", " << at(rowIndex, colIndex);
				}
				toString << "]\n";
			}

			return toString.str();
		}

	public:
		const size_t size, rows, columnLength, rowLength, columns;

		/**
		 * Constructor
		 */
		CUDA_ANY constexpr Matrix(size_t columnLength, size_t rowLength)
			: rows(columnLength), columns(rowLength),
			  size(columnLength * rowLength),
			  array(new T[columnLength * rowLength]),
			  columnLength(columnLength), rowLength(rowLength){}

		CUDA_ANY constexpr Matrix(size_t columnLength, size_t rowLength, T* const array)
			: rows(columnLength), columns(rowLength),
			  size(columnLength * rowLength),
			  array(array),
			  columnLength(columnLength), rowLength(rowLength){}

		/** Getters **/
		/**
		 * Access value at a determined matrix's index
		 * @param  rowIndex [The matrix row position]
		 * @param  colIndex [The matrix column position]
		 * @return          [Value at [Row][Column] position]
		 */
		CUDA_ANY T const &at(size_t rowIndex, size_t colIndex) const {
			return array[rowIndex * rowLength + colIndex];
		}

		CUDA_ANY T &at(size_t rowIndex, size_t colIndex) {
			return const_cast<T &>(
					const_cast<Matrix<T> const &>(*this)
							.at(rowIndex, colIndex)
			);
		}

		// Operator overload - Function call
		CUDA_ANY T const &operator()(size_t rowIndex, size_t colIndex) const {
			return at(rowIndex, colIndex);
		}

		CUDA_ANY T &operator()(size_t rowIndex, size_t colIndex) {
			return at(rowIndex, colIndex);
		}

		/** Setters **/
		/**
		 * Fill matrix with single value
		 * @param fillValue [Value used to fill matrix]
		 * @param rowIndex  [Start row]
		 * @param numOfRows [Number of rows to be filled]
		 * @param colIndex  [Start columns]
		 * @param numOfCols [Number of columns to be filled]
		 */
		CUDA_ANY void fill(T fillValue, size_t rowIndex, size_t numOfRows,
		          size_t colIndex, size_t numOfCols
		) {
			for (size_t rowCounter = 0; rowCounter < numOfRows; ++rowCounter)
				for (size_t colCounter = 0; colCounter < numOfCols; ++colCounter)
					at(rowIndex + rowCounter, colIndex + colCounter) = fillValue;
		}

		/**
		 * Overload of fill
		 * @param fillValue [Value used to fill matrix]
		 * @param rowIndex [Start row]
		 * @param colIndex [Start columns]
		 */
		CUDA_ANY void fill (
				T fillValue, size_t rowIndex = 0, size_t colIndex = 0
		) {
			return fill(fillValue, rowIndex, rows, colIndex, columns);
		}

		/**
		 * Zero-fill matrix
		 */
		CUDA_ANY void zero() { return fill(static_cast<T>(0.0)); }

		/**
		 * Fill a matrix's column
		 * @param index     [Which column]
		 * @param fillValue [Value to fill column]
		 */
		CUDA_ANY void fillColumn(size_t index, T fillValue) {
			fill(fillValue, 0, columnLength, index, 1);
		}

		/**
		 * Fill a matrix's row
		 * @param index     [Which row]
		 * @param fillValue [Value to fill row]
		 */
		CUDA_ANY void fillRow(size_t index, T fillValue) {
			fill(fillValue, index, 1, 0, rowLength);
		}

		/**
		 * Convert operator overload
		 * Declare how a Matrix should be interpreted when converted to T*
		 */
		CUDA_ANY operator const T *() const { return array; }

		CUDA_ANY operator T *() { return array; }

		/**
		 * Convert operator overload
		 * Declare how a Matrix should be interpreted when converted to std::string
		 */
		CUDA_HOST operator const std::string() const { return toString(); }

		CUDA_HOST operator std::string() { return toString(); }
	};

	/**
	 * Operator Overload - <<
	 */
	template<typename T>
	CUDA_HOST std::ostream &operator<<(std::ostream &stream, const Matrix<T> &m) {
		return stream << static_cast<std::string const &>(m);
	}
}

#endif
