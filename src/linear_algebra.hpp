#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

namespace LinearAlgebra
{

using Matrix = std::vector<std::vector<std::int64_t>>;
using Vector = std::vector<std::int64_t>;

Matrix GenerateEmptyMatrix(std::size_t rows, std::size_t cols);
Matrix GenerateEmptySquareMatrix(std::size_t n);
Matrix GenerateMatrixWithRandomValues(std::size_t rows, std::size_t cols);
Matrix GenerateSquareMatrixWithRandomValues(std::size_t n);
void PrintMatrix(const Matrix &m, bool finalNewline = true, bool roundDecimals = true, bool skipZeroes = false);
void PrintVector(const Vector &v, bool finalNewline = true, bool roundDecimals = true, bool skipZeroes = false);
Matrix MultiplyMatrixes(const Matrix &a, const Matrix &b);
Vector GenerateVectorWithRandomValues(std::size_t n);
Vector GenerateRademacherVector(std::uint32_t n);
Vector MultiplyMatrixAndVector(const Matrix &m, const Vector &v);
Vector MultiplyVectorAndMatrix(const Vector &v, const Matrix &m);
Vector MultiplyScalarAndVector(double scalar, const Vector &v);
Vector SubtractVectors(const Vector &a, const Vector &b);
Matrix DiagonalizeMatrix(const Matrix &m);
Matrix AppendVectorToMatrixAsRow(const Vector &v, const Matrix &m);
bool AreVectorsEqual(const Vector &a, const Vector &b, bool roundDecimals = true);
bool AreMatrixesEqual(const Matrix &a, const Matrix &b, bool roundDecimals = true);
bool FreivaldsAlgorithm(const Matrix &a, const Matrix &b, const Matrix &c, std::size_t n,
                        std::uint32_t numberOfIterations = 1);
Matrix PadMatrix(Matrix &m, std::size_t n);
Matrix PadMatrix(Matrix &m, std::size_t rows, std::size_t cols);
Matrix TileMatrixVertically(const Matrix &m, std::size_t targetRows);
Vector ReplicateToBatch(const Vector &v, std::size_t batchSize);
Matrix Transpose(const Matrix &m);
bool IsPrime(std::int64_t num);
Matrix SliceMatrix(const Matrix &mat, std::size_t r, std::size_t c, std::size_t size);
void StitchMatrix(Matrix &mat, const Matrix &sub, std::size_t r, std::size_t c, std::size_t size);

template <typename T>
std::vector<T> FlattenRowMajor(const std::vector<std::vector<T>> &m)
{
	if (m.empty()) {
		return {};
	}

	const std::size_t n = m.size();
	const std::size_t d = m[0].size();

	std::vector<T> flatVector(n * d);

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < d; ++j) {
			flatVector[i * d + j] = m[i][j];
		}
	}

	return flatVector;
}

template <typename T>
std::vector<std::vector<T>> ConvertVectorToSquareMatrix(const std::vector<T> &v)
{
	const std::size_t size = std::sqrt(v.size());

	std::vector<std::vector<T>> m(size, std::vector<T>(size));

	for (std::size_t i = 0; i < size; ++i) {
		for (std::size_t j = 0; j < size; ++j) {
			m[i][j] = v[i * size + j];
		}
	}

	return m;
}

template <typename T>
void WriteMatrixToFile(const std::vector<std::vector<T>> &m, const std::filesystem::path &path)
{
	std::ofstream file;
	file.open(path);
	if (!file.is_open()) {
		return;
	}

	for (const auto &v : m) {
		for (const auto &i : v) {
			file << i << " ";
		}
		file << "\n";
	}

	file.close();
}

template <typename T>
void WriteMatrixToFile(const std::vector<T> &v, const std::filesystem::path &path)
{
	std::ofstream file;
	file.open(path);
	if (!file.is_open()) {
		return;
	}

	std::size_t size = std::sqrt(v.size());

	for (std::size_t i = 0; i < v.size(); ++i) {
		file << v.at(i) << " ";
		if ((i + 1) % size == 0) {
			file << "\n";
		}
	}

	file.close();
}

} // namespace LinearAlgebra
