#include <array>
#include <string>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#ifndef MATRIX_HPP
#define MATRIX_HPP
#define MATRIX_TO_STRING_LIMIT 20

namespace Matrix {
	/**
	 * Matrix Class
	 * This code was heavily influenced by:
	 * https://raw.githubusercontent.com/VMML/vmmlib/master/vmmlib/matrix.hpp
	 */
	template<size_t columnLength, size_t rowLength, typename T = double>
	class Matrix {
	private:
		/** Members **/
		T array[columnLength * rowLength];

		/** Methods */
		/**
		 * Get the values of a matrix's column
		 * @param  colIndex [Which column]
		 * @return       [Column's values]
		 */
		std::array<T, columnLength> getColumnAtIndex(size_t colIndex) const {
			if (colIndex >= columns)
				throw std::out_of_range("Index out of bounds");

			std::array<T, columnLength> column;

			for (size_t rowCounter = 0; rowCounter < rows; ++rowCounter)
				column[rowCounter] = at(rowCounter, colIndex);

			return column;
		}

		/**
		 * Get the values of a matrix's row
		 * @param  index [Which row]
		 * @return       [Row's values]
		 */
		std::array<T, rowLength> getRowAtIndex(size_t index) const {
			if (index >= columnLength)
				throw std::out_of_range("Index out of bounds");

			std::array<T, columnLength> row;

			std::memcpy(row, &array[rowLength * index], rowLength * sizeof(T));

			return row;
		}

		/**
		 * Convert matrix to a string representation
		 * @return [Matrix's string representation]
		 */
		std::string toString() const {
			std::string toString;
			size_t numOfCols = columns;
			size_t numOfRows = rows;
			bool colContinue = false;
			bool rowContinue = false;

			if (columnLength > MATRIX_TO_STRING_LIMIT) {
				numOfRows = MATRIX_TO_STRING_LIMIT;
				rowContinue = true;
			}

			if (rowLength > MATRIX_TO_STRING_LIMIT) {
				numOfCols = MATRIX_TO_STRING_LIMIT;
				colContinue = true;
			}

			for (size_t rowIndex = 0; rowIndex < numOfRows; ++rowIndex) {
				toString += '[' + std::to_string(at(rowIndex, 0));

				for (size_t colIndex = 1; colIndex < numOfCols; ++colIndex) {
					toString += ", " + std::to_string(at(rowIndex, colIndex));
				}

				if (colContinue) { toString += ", . . ."; }
				toString += "]\n";
			}

			if (rowContinue) { toString += ".\n.\n.\n"; }

			return toString;
		}

	public:
		static const size_t size = columnLength * rowLength;
		static const size_t rows = columnLength;
		static const size_t columns = rowLength;

		/**
		 * Constructor
		 */
		Matrix() : array() {}

		/** Interators **/
		const T* begin() const { return array; };

		const T* end() const { return array + size; };

		T* begin() {
			return const_cast<T*>(
					const_cast<Matrix<rows, columns, T> const &>(*this).begin()
			);
		};

		T* end() {
			return const_cast<T*>(
					const_cast<Matrix<rows, columns, T> const &>(*this).end()
			);
		};

		std::reverse_iterator<const T*> rbegin() const {
			return array + size - 1;
		};

		std::reverse_iterator<const T*> rend() const {
			return array - 1;
		};

		std::reverse_iterator<T*> rbegin() {
			return const_cast<std::reverse_iterator<T*>>(
					const_cast<Matrix<rows, columns, T> const &>(*this).rbegin()
			);
		}

		std::reverse_iterator<T*> rend() {
			return const_cast<std::reverse_iterator<T*>>(
					const_cast<Matrix<rows, columns, T> const &>(*this).rend()
			);
		}

		/** Getters **/
		/**
		 * Access value at a determined matrix's index
		 * @param  rowIndex [The matrix row position]
		 * @param  colIndex [The matrix column position]
		 * @return          [Value at [Row][Column] position]
		 */
		const T &at(size_t rowIndex, size_t colIndex) const {
			if (rowIndex >= rows || colIndex >= columns)
				throw std::out_of_range("Index out of bounds");

			return array[rowIndex * rowLength + colIndex];
		}

		T &at(size_t rowIndex, size_t colIndex) {
			return const_cast<T &>(
					const_cast<Matrix<rows, columns, T> const &>(*this)
							.at(rowIndex, colIndex)
			);
		}

		// Operator overload - Function call
		const T &operator()(size_t rowIndex, size_t colIndex) const {
			return at(rowIndex, colIndex);
		}

		T &operator()(size_t rowIndex, size_t colIndex) {
			return at(rowIndex, colIndex);
		}

		const std::array<T, columnLength> getColumn(size_t index) const {
			return getColumnAtIndex(index);
		}

		std::array<T, columnLength> getColumn(size_t index) {
			return getColumnAtIndex(index);
		}

		const std::array<T, rowLength> getRow(size_t index) const {
			return getRowAtIndex(index);
		}

		std::array<T, rowLength> getRow(size_t index) {
			return getRowAtIndex(index);
		}

		/** Setters **/
		/**
		 * Fill matrix with single value
		 * @param fillValue [Value used to fill matrix]
		 * @param rowIndex  [Start row]
		 * @param numOfRows [Number of rows to be filler]
		 * @param colIndex  [Start columns]
		 * @param numOfCols [Number of columns to be filler]
		 */
		void fill(T fillValue, size_t rowIndex = 0, size_t numOfRows = rows,
		          size_t colIndex = 0, size_t numOfCols = columns
		) {
			if (rowIndex < 0 || rowIndex >= rows ||
			    colIndex < 0 || colIndex >= columns ||
			    numOfRows < 0 || numOfRows > (rows - rowIndex) ||
			    numOfCols < 0 || numOfCols > (columns - colIndex)
					) { throw std::out_of_range("Index out of bounds"); }

			for (size_t rowCounter = 0; rowCounter < numOfRows; ++rowCounter)
				for (size_t colCounter = 0; colCounter < numOfCols; ++colCounter)
					at(rowIndex + rowCounter, colIndex + colCounter) = fillValue;
		}

		/**
		 * Zero-fill matrix
		 */
		void zero() { fill(static_cast<T>(0.0)); }

		/**
		 * Set the value of a matrix's column
		 * @param colIndex  [Which column]
		 * @param column [Column's new values]
		 */
		void setColumn(size_t colIndex, const std::array<T, columnLength>& column) {
			if (colIndex >= columns)
				throw std::out_of_range("Index out of bounds");

			for (size_t rowCounter = 0; rowCounter < rows; ++rowCounter)
				at(rowCounter, colIndex) = column[rowCounter];
		}

		/**
		 * Fill a matrix's column
		 * @param index     [Which column]
		 * @param fillValue [Value to fill column]
		 */
		void fillColumn(size_t index, T fillValue) {
			fill(fillValue, 0, columnLength, index, 1);
		}

		/**
		 * Set the value of a matrix's row
		 * @param index  [Which row]
		 * @param column [Row's new values]
		 */
		void setRow(size_t index, const std::array<T, rowLength>& row) {
			if (index >= rows)
				throw std::out_of_range("Index out of bounds");

			std::memcpy(array + rowLength * index, row, rowLength * sizeof(T));
		}

		/**
		 * Fill a matrix's row
		 * @param index     [Which row]
		 * @param fillValue [Value to fill row]
		 */
		void fillRow(size_t index, T fillValue) {
			fill(fillValue, index, 1);
		}

		/**
		 * Convert operator overload
		 * Declare how a Matrix should be interpreted when converted to T*
		 */
		operator const T *() const { return array; }

		operator T *() { return array; }

		/**
		 * Convert operator overload
		 * Declare how a Matrix should be interpreted when converted to std::string
		 */
		operator const std::string() const { return toString(); }

		operator std::string() { return toString(); }
	};

/**
 * Operator Overload - <<
 */
	template<size_t M, size_t N, typename T>
	std::ostream &operator<<(std::ostream &stream, const Matrix<M, N, T> &m) {
		return stream << static_cast<std::string const &>(m);
	}
}

#endif
