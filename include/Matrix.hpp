#include <array>
#include <string>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#ifndef MATRIX_HPP
#define MATRIX_HPP
#define MATRIX_TO_STRING_LIMIT 200

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

		/** Methods */
		/**
		 * Get the values of a matrix's column
		 * @param  colIndex [Which column]
		 * @return       [Column's values]
		 */
		T* getColumnAtIndex(size_t colIndex) const {
			if (colIndex >= columns)
				throw std::out_of_range("Index out of bounds");

			T* column = new T[columnLength];

			for (size_t rowCounter = 0; rowCounter < rows; ++rowCounter)
				column[rowCounter] = at(rowCounter, colIndex);

			return column;
		}

		/**
		 * Get the values of a matrix's row
		 * @param  index [Which row]
		 * @return       [Row's values]
		 */
		T* getRowAtIndex(size_t index) const {
			if (index >= columnLength)
				throw std::out_of_range("Index out of bounds");

			T* row = new T[rowLength];

			std::memcpy(row, &array[rowLength * index], rowLength * sizeof(T));

			return row;
		}

		/**
		 * Convert matrix to a string representation
		 * @return [Matrix's string representation]
		 */
		std::string toString() const {
			std::stringstream toString;
			size_t numOfCols = columns;
			size_t numOfRows = rows;
			bool colContinue = false;
			bool rowContinue = false;

			toString << std::showpos << std::fixed;
			toString.precision(5);

			if (columnLength > MATRIX_TO_STRING_LIMIT) {
				numOfRows = MATRIX_TO_STRING_LIMIT;
				rowContinue = true;
			}

			if (rowLength > MATRIX_TO_STRING_LIMIT) {
				numOfCols = MATRIX_TO_STRING_LIMIT;
				colContinue = true;
			}

			for (size_t rowIndex = 0; rowIndex < numOfRows; ++rowIndex) {
				toString << '[' << at(rowIndex, 0);

				for (size_t colIndex = 1; colIndex < numOfCols; ++colIndex) {
					toString << ", " << at(rowIndex, colIndex);
				}

				if (colContinue) { toString << ", . . ."; }
				toString << "]\n";
			}

			if (rowContinue) { toString << ".\n.\n.\n"; }

			return toString.str();
		}

	public:
		size_t const size, rows, columnLength, rowLength, columns;

		/**
		 * Constructor
		 */
		constexpr Matrix(size_t columnLength, size_t rowLength)
			: rows(columnLength), columns(rowLength),
			  size(columnLength * rowLength), array(new T[columnLength * rowLength]),
			  columnLength(columnLength), rowLength(rowLength){}

		/** Iterators **/
		T* const begin() const { return array; };

		T* const end() const { return array + size; };

		std::reverse_iterator<T* const> rbegin() const {
			return std::reverse_iterator<T* const>(array + size - 1);
		};

		std::reverse_iterator<T* const> rend() const {
			return std::reverse_iterator<T* const>(array - 1);
		};

		/** Getters **/
		/**
		 * Access value at a determined matrix's index
		 * @param  rowIndex [The matrix row position]
		 * @param  colIndex [The matrix column position]
		 * @return          [Value at [Row][Column] position]
		 */
		T const &at(size_t rowIndex, size_t colIndex) const {
			if (rowIndex >= rows || colIndex >= columns)
				throw std::out_of_range("Index out of bounds");

			return array[rowIndex * rowLength + colIndex];
		}

		T &at(size_t rowIndex, size_t colIndex) {
			return const_cast<T &>(
					const_cast<Matrix<T> const &>(*this)
							.at(rowIndex, colIndex)
			);
		}

		// Operator overload - Function call
		T const &operator()(size_t rowIndex, size_t colIndex) const {
			return at(rowIndex, colIndex);
		}

		T &operator()(size_t rowIndex, size_t colIndex) {
			return at(rowIndex, colIndex);
		}

		T* const getColumn(size_t index) const {
			return getColumnAtIndex(index);
		}

		T* getColumn(size_t index) {
			return getColumnAtIndex(index);
		}

		T* const getRow(size_t index) const {
			return getRowAtIndex(index);
		}

		T* getRow(size_t index) {
			return getRowAtIndex(index);
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
		void fill(T fillValue, size_t rowIndex, size_t numOfRows,
		          size_t colIndex, size_t numOfCols
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
		 * Overload of fill
		 * @param fillValue [Value used to fill matrix]
		 * @param rowIndex [Start row]
		 * @param colIndex [Start columns]
		 */
		void fill (T fillValue, size_t rowIndex = 0, size_t colIndex = 0) {
			return fill(fillValue, rowIndex, rows, colIndex, columns);
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
		template<size_t columnLength>
		void setColumn(size_t colIndex, const std::array<T, columnLength>& column) {
			if (colIndex >= columns)
				throw std::out_of_range("Index out of bounds");

			size_t limit = rows < columnLength ? rows : columnLength;
			for (size_t rowCounter = 0; rowCounter < limit; ++rowCounter)
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
		template<size_t rowLength>
		void setRow(size_t index, const std::array<T, rowLength>& row) {
			if (index >= rows)
				throw std::out_of_range("Index out of bounds");

			size_t limit = rows < columnLength ? rows : columnLength;
			std::memcpy(array + (this->rowLength * index),
			            static_cast<const void*>(row),
			            limit * sizeof(T)
			);
		}

		/**
		 * Fill a matrix's row
		 * @param index     [Which row]
		 * @param fillValue [Value to fill row]
		 */
		void fillRow(size_t index, T fillValue) {
			fill(fillValue, index, 1, 0, rowLength);
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
	template<typename T>
	std::ostream &operator<<(std::ostream &stream, const Matrix<T> &m) {
		return stream << static_cast<std::string const &>(m);
	}
}

#endif
