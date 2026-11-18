#include "linear_algebra.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>

namespace LinearAlgebra
{

Matrix GenerateEmptyMatrix(std::size_t rows, std::size_t cols)
{
	Matrix m(rows, Vector(cols));

	return m;
}

Matrix GenerateEmptySquareMatrix(std::size_t n) { return GenerateEmptyMatrix(n, n); }

Matrix GenerateMatrixWithRandomValues(std::size_t rows, std::size_t cols)
{
	Matrix m(rows, Vector(cols));

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0, 9);

	for (auto &r : m) {
		for (auto &v : r) {
			v = dist(gen);
		}
	}

	return m;
}

Matrix GenerateSquareMatrixWithRandomValues(std::size_t n) { return GenerateMatrixWithRandomValues(n, n); }

void PrintMatrix(const Matrix &m, bool finalNewline, bool roundDecimals, bool skipZeroes)
{
	for (const auto &v : m) {
		PrintVector(v, false, roundDecimals, skipZeroes);
	}

	if (finalNewline) {
		std::cout << "\n";
	}
}

void PrintVector(const Vector &v, bool finalNewline, bool roundDecimals, bool skipZeroes)
{
	for (const auto &i : v) {
		if (skipZeroes && std::round(i) == 0) {
			continue;
		}
		if (roundDecimals) {
			std::cout << std::fixed << std::setprecision(0) << i << " ";
		} else {
			std::cout << i << " ";
		}
	}
	std::cout << "\n";

	if (finalNewline) {
		std::cout << "\n";
	}
}

Matrix MultiplyMatrixes(const Matrix &a, const Matrix &b)
{
	std::size_t m = a.size();
	std::size_t p = a[0].size();

	assert(p == b.size());

	std::size_t n = b[0].size();

	Matrix c(m, std::vector<std::int64_t>(n, 0.0));

#pragma omp parallel for
	for (std::size_t i = 0; i < m; ++i) {
		for (std::size_t k = 0; k < p; ++k) {
			for (std::size_t j = 0; j < n; ++j) {
				c[i][j] += a[i][k] * b[k][j];
			}
		}
	}

	return c;
}

Vector GenerateVectorWithRandomValues(std::size_t n)
{
	Vector v(n);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<int> dist(0, 9);

	for (auto &i : v) {
		i = dist(gen);
	}

	return v;
}

Vector GenerateRademacherVector(std::uint32_t n)
{
	Vector v(n);
	static std::mt19937_64 rng{std::random_device{}()};
	std::uniform_int_distribution<int> coin(0, 1);
	for (auto &x : v) {
		x = coin(rng) ? 1.0 : -1.0;
	}
	return v;
}

Vector MultiplyMatrixAndVector(const Matrix &m, const Vector &v)
{
	assert(!m.empty());
	assert(m[0].size() == v.size());

	std::size_t rows = m.size();
	std::size_t cols = v.size();

	Vector result(rows, 0);

	for (std::size_t i = 0; i < rows; ++i) {
		for (std::size_t j = 0; j < cols; ++j) {
			result[i] += m[i][j] * v[j];
		}
	}

	return result;
}

Vector MultiplyVectorAndMatrix(const Vector &v, const Matrix &m)
{
	assert(!m.empty());
	assert(v.size() == m.size());

	std::size_t rows = m.size();
	std::size_t cols = m[0].size();

	Vector result(cols, 0);

#pragma omp parallel for
	for (std::size_t j = 0; j < cols; ++j) {
		for (std::size_t i = 0; i < rows; ++i) {
			result[j] += v[i] * m[i][j];
		}
	}

	return result;
}

Vector MultiplyScalarAndVector(double scalar, const Vector &v)
{
	Vector product = v;
	for (auto &x : product) {
		x *= scalar;
	}
	return product;
}

Vector SubtractVectors(const Vector &a, const Vector &b)
{
	assert(a.size() == b.size());

	Vector c(a.size(), 0);

	for (std::size_t i = 0; i < a.size(); ++i) {
		c[i] = a[i] - b[i];
	}

	return c;
}

Matrix DiagonalizeMatrix(const Matrix &m)
{
	std::size_t rows = m.size();
	std::size_t cols = m[0].size();
	Matrix diags(cols, Vector(rows));

	for (std::size_t i = 0; i < cols; ++i) {
		for (std::size_t j = 0; j < rows; ++j) {
			diags[i][j] = m[j][(j + i) % cols];
		}
	}

	return diags;
}

Matrix AppendVectorToMatrixAsRow(const Vector &v, const Matrix &m)
{
	Matrix result = m;
	result.push_back(v);
	return result;
}

bool AreVectorsEqual(const Vector &a, const Vector &b, bool roundDecimals)
{
	if (a.size() != b.size()) {
		return false;
	}

	for (std::size_t i = 0; i < a.size(); ++i) {
		double lhs = roundDecimals ? std::round(a[i]) : a[i];
		double rhs = roundDecimals ? std::round(b[i]) : b[i];

		if (lhs != rhs) {
			return false;
		}
	}

	return true;
}

bool AreMatrixesEqual(const Matrix &a, const Matrix &b, bool roundDecimals)
{
	for (std::size_t i = 0; i < a.size(); ++i) {
		if (!AreVectorsEqual(a[i], b[i], roundDecimals)) {
			return false;
		}
	}

	return true;
}

bool FreivaldsAlgorithm(const Matrix &a, const Matrix &b, const Matrix &c, std::size_t n,
                        std::uint32_t numberOfIterations)
{
	bool result = true;

	for (std::uint32_t i = 0; i < numberOfIterations; ++i) {
		Vector r = GenerateVectorWithRandomValues(n);

		Vector br = MultiplyMatrixAndVector(b, r);
		Vector abr = MultiplyMatrixAndVector(a, br);
		Vector cr = MultiplyMatrixAndVector(c, r);

		Vector abrcr = SubtractVectors(abr, cr);

		for (const auto &z : abrcr) {
			if (z != 0) {
				return false;
			}
		}
	}

	return result;
}

Matrix PadMatrix(Matrix &m, std::size_t n)
{
	Matrix result = m;
	result.resize(n, Vector(n, 0.0));
	for (auto &row : result) {
		row.resize(n, 0.0);
	}
	return result;
}

Matrix PadMatrix(Matrix &m, std::size_t rows, std::size_t cols)
{
	Matrix result = m;
	result.resize(rows, Vector(cols, 0.0));
	for (auto &row : result) {
		row.resize(cols, 0.0);
	}
	return result;
}

Matrix TileMatrixVertically(const Matrix &m, std::size_t targetRows)
{
	if (m.empty()) {
		return {};
	}
	std::size_t srcRows = m.size();
	std::size_t cols = m[0].size();
	Matrix tiled(targetRows, Vector(cols, 0));
	for (std::size_t i = 0; i < targetRows; ++i) {
		tiled[i] = m[i % srcRows];
	}
	return tiled;
}

Vector ReplicateToBatch(const Vector &v, std::size_t batchSize)
{
	Vector out(batchSize);
	for (std::size_t i = 0; i < batchSize; ++i) {
		out[i] = v[i % v.size()];
	}
	return out;
}

Matrix Transpose(const Matrix &m)
{
	std::size_t rows = m.size();
	std::size_t cols = m[0].size();
	Matrix mt(cols, Vector(rows));

	for (std::size_t i = 0; i < rows; ++i) {
		for (std::size_t j = 0; j < cols; ++j) {
			mt[j][i] = m[i][j];
		}
	}

	return mt;
}

bool IsPrime(std::int64_t num)
{
	if (num <= 1) {
		return false;
	}
	if (num <= 3) {
		return true;
	}
	if (num % 2 == 0 || num % 3 == 0) {
		return false;
	}
	for (std::int64_t i = 5; i * i <= num; i += 6) {
		if (num % i == 0 || num % (i + 2) == 0) {
			return false;
		}
	}
	return true;
}

LinearAlgebra::Matrix SliceMatrix(const LinearAlgebra::Matrix &mat, std::size_t r, std::size_t c, std::size_t size)
{
	LinearAlgebra::Matrix res(size, LinearAlgebra::Vector(size, 0.0));
	for (std::size_t i = 0; i < size; ++i) {
		for (std::size_t j = 0; j < size; ++j) {
			res[i][j] = mat[r + i][c + j];
		}
	}
	return res;
}

void StitchMatrix(LinearAlgebra::Matrix &mat, const LinearAlgebra::Matrix &sub, std::size_t r, std::size_t c,
                  std::size_t size)
{
	for (std::size_t i = 0; i < size; ++i) {
		for (std::size_t j = 0; j < size; ++j) {
			mat[r + i][c + j] = sub[i][j];
		}
	}
}

} // namespace LinearAlgebra
