#include <string>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <stdexcept>

/**
 * Matrix Class
 * This code was heavily influenced by:
 * https://raw.githubusercontent.com/VMML/vmmlib/master/vmmlib/matrix.hpp
 */
template<size_t M, size_t N, typename T = double>
class Matrix {
private:
	/** Members **/
	T array[M * N];

	/** Mtehods */
	/**
	 * Get the values of a matrix's column
	 * @param  index [Which column]
	 * @return       [Column's values]
	 */
	T* getColumnAtIndex(size_t index) const {
		if (index >= N)
			throw std::out_of_range("Index out of bounds");

		T* column = static_cast<T*>(malloc(M * sizeof(T)));
		memcpy(column, &array[M * index], M * sizeof(T));
		return column;
	}

	/**
	 * Get the values of a matrix's row
	 * @param  index [Which row]
	 * @return       [Row's values]
	 */
	T* getRowAtIndex(size_t index) const {
		if (index >= M)
			throw std::out_of_range("Index out of bounds");

		T* row = static_cast<T*>(malloc(N * sizeof(T)));
		memcpy(row, &array[N * index], N * sizeof(T));
		return row;
	}

	std::string toString() const {
		std::string toString;
		size_t numOfCols = N;
		bool colContinue = false;

		if (N > 100) {
			numOfCols = 100;
			colContinue = true;
		}

		for(size_t rowIndex = 0; rowIndex < M; ++rowIndex) {
			toString += '[' +  std::to_string(at(rowIndex, 0));

			for(size_t colIndex = 1; colIndex < numOfCols; ++colIndex) {
				toString +=
					", " + std::to_string(at(rowIndex, colIndex));
			}

			if (colContinue) toString += ", ...";
			toString += "]\n";

			if (rowIndex > 100)	{
				toString += "...";
				break;
			}
		}

		return toString;
	}
public:
	static const size_t rows = M;
	static const size_t cols = N;
	static const size_t size = M * N;

	/**
	 * Constructor
	 */
	Matrix() : array() {}

	/** Getters **/
	/**
	 * Access value at a determined matrix's index
	 * @param  rowIndex [The matrix row position]
	 * @param  colIndex [The matrix column position]
	 * @return          [Value at [Row][Column] position]
	 */
	inline const T& at(size_t rowIndex, size_t colIndex) const {
		if (rowIndex >= M || colIndex >= N)
			throw std::out_of_range("Index out of bounds");

		return array[rowIndex * M + colIndex];
	}
	inline T& at(size_t rowIndex, size_t colIndex) {
		return const_cast<T&>(
			const_cast<Matrix<M, N, T> const&>(*this).at(rowIndex, colIndex)
		);
	}

	// Operator overload - Function call
	const inline T& operator()(size_t rowIndex, size_t colIndex) const {
		return at(rowIndex, colIndex);
	}
	inline T& operator()(size_t rowIndex, size_t colIndex) {
		return at(rowIndex, colIndex);
	}

	const T* getColumn(size_t index) const { return getColumnAtIndex(index); }
	T* getColumn(size_t index) { return getColumnAtIndex(index); }

	const T* getRow(size_t index) const { return getRowAtIndex(index); }
	T* getRow(size_t index) { return getRowAtIndex(index); }

	/** Setters **/
	/**
	 * Fill matrix with single value
	 * @param fillValue [Value used to fill matrix]
	 * @param rowIndex  [Start row]
	 * @param numOfRows [Number of rows to be filler]
	 * @param colIndex  [Start columns]
	 * @param numOfCols [Number of columns to be filler]
	 */
	void fill(T fillValue, size_t rowIndex = 0, size_t numOfRows = M,
		size_t colIndex = 0, size_t numOfCols = N
	) {
		if (rowIndex < 0 || rowIndex >= M ||
			colIndex < 0 || colIndex >= N ||
			numOfRows < 0 || numOfRows > (M - rowIndex) ||
			numOfCols < 0 || numOfCols > (N - colIndex)
		) {
			throw std::out_of_range("Index out of bounds");
		}

		for(size_t rowCounter = 0; rowCounter < numOfRows; ++rowCounter)
			for(size_t colCounter = 0; colCounter < numOfCols; ++colCounter)
				at(
					rowIndex + rowCounter, colIndex + colCounter
				) = fillValue;
	}

	/**
	 * Zero-fill matrix
	 */
	void zero() { fill(static_cast<T>(0.0)); }

	/**
	 * Set the value of a matrix's column
	 * @param index  [Which column]
	 * @param column [Column's new values]
	 */
	void setColumn(size_t index, T* column) {
		if (index >= N)
			throw std::out_of_range("Index out of bounds");

		memcpy(array + M * index, column, M * sizeof(T));
	}

	/**
	 * Fill a matrix's column
	 * @param index     [Which column]
	 * @param fillValue [Value to fill column]
	 */
	void fillColumn(size_t index, T fillValue) {
		fill(fillValue, 0, M, index, 1);
	}

	/**
	 * Set the value of a matrix's row
	 * @param index  [Which row]
	 * @param column [Row's new values]
	 */
	void setRow(size_t index, T* row) {
		if (index >= M)
			throw std::out_of_range("Index out of bounds");

		static_assert(index < M, "Invalid Index");
		memcpy(array + N * index, row, N * sizeof(T));
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
	operator T*() { return array; }

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
std::ostream& operator<<(std::ostream &strm, const Matrix<M, N, T> &m) {
  return strm << static_cast<std::string>(m);
}