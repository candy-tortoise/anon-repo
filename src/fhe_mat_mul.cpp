#include "fhe_mat_mul.hpp"

#include <cassert>
#include <iostream>

namespace fhe
{

std::vector<std::int32_t> ComputeRequiredRotationIndices(std::size_t d)
{
	const std::int64_t n = static_cast<std::int64_t>(d);
	std::vector<std::int32_t> indices;
	if (n < 2) {
		return indices;
	}
	indices.reserve(static_cast<std::size_t>(4 * (n - 1)));

	for (std::int64_t i = 1; i < n; ++i) {
		indices.push_back(static_cast<std::int32_t>(i));            // Left shift
		indices.push_back(static_cast<std::int32_t>(i * n));        // Left shift (Up)
		indices.push_back(static_cast<std::int32_t>(-(n - i)));     // Right shift
		indices.push_back(static_cast<std::int32_t>(-(n - i) * n)); // Right shift (Down)
	}

	std::sort(indices.begin(), indices.end());

	assert(std::adjacent_find(indices.begin(), indices.end()) == indices.end());
	assert(indices.size() == 4 * (d - 1));

	return indices;
}

// ---------------------------------------------------------
// COMMON ROTATIONS (Used by both Fast and Robust algorithms)
// ---------------------------------------------------------
Ciphertext<DCRTPoly> RotateCiphertextA(int s, CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                       std::size_t d)
{
	if (s == 0) {
		return cipher_matrix;
	}

	auto left_cipher = cc->EvalRotate(cipher_matrix, s);
	auto right_cipher = cc->EvalRotate(cipher_matrix, -((int)d - s));

	std::vector<std::int64_t> mask_left(d * d, 0);
	std::vector<std::int64_t> mask_right(d * d, 0);

	for (std::size_t r = 0; r < d; ++r) {
		for (std::size_t c = 0; c < d - s; ++c) {
			mask_left[r * d + c] = 1;
		}
		for (std::size_t c = d - s; c < d; ++c) {
			mask_right[r * d + c] = 1;
		}
	}

	auto plain_left = cc->MakePackedPlaintext(mask_left);
	auto plain_right = cc->MakePackedPlaintext(mask_right);

	return cc->EvalAdd(cc->EvalMult(left_cipher, plain_left), cc->EvalMult(right_cipher, plain_right));
}

Ciphertext<DCRTPoly> ChangeCipherToMatrixA(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                           std::size_t d)
{
	std::vector<Ciphertext<DCRTPoly>> cipher_vector;
	for (std::size_t r = 0; r < d; ++r) {
		if (r == 0) {
			std::vector<int64_t> mask(d * d, 0);
			for (std::size_t c = 0; c < d; ++c) {
				mask[r * d + c] = 1;
			}
			auto plain_mask = cc->MakePackedPlaintext(mask);
			cipher_vector.push_back(cc->EvalMult(cipher_matrix, plain_mask));
			continue;
		}

		auto left_cipher = cc->EvalRotate(cipher_matrix, r);
		auto right_cipher = cc->EvalRotate(cipher_matrix, -((int)d - r));

		std::vector<std::int64_t> mask_left(d * d, 0);
		std::vector<std::int64_t> mask_right(d * d, 0);

		for (std::size_t c = 0; c < d - r; ++c) {
			mask_left[r * d + c] = 1;
		}
		for (std::size_t c = d - r; c < d; ++c) {
			mask_right[r * d + c] = 1;
		}

		auto plain_left = cc->MakePackedPlaintext(mask_left);
		auto plain_right = cc->MakePackedPlaintext(mask_right);

		cipher_vector.push_back(cc->EvalMult(left_cipher, plain_left));
		cipher_vector.push_back(cc->EvalMult(right_cipher, plain_right));
	}
	auto dest = cc->EvalAddMany(cipher_vector);
	cc->RelinearizeInPlace(dest);
	cc->RescaleInPlace(dest);
	return dest;
}

std::vector<Ciphertext<DCRTPoly>> FlattenAndEncryptMatrix(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
                                                          lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys,
                                                          LinearAlgebra::Matrix &m, bool diagonalize)
{
	std::vector<Ciphertext<DCRTPoly>> ciphertexts;
	LinearAlgebra::Matrix matrix;

	std::size_t numRows = diagonalize ? m.size() : m[0].size();
	std::size_t numCols = diagonalize ? m[0].size() : m.size();

	if (diagonalize) {
		matrix = LinearAlgebra::DiagonalizeMatrix(m);
	} else {
		LinearAlgebra::Matrix mt(numRows, LinearAlgebra::Vector(numCols));
		for (std::size_t i = 0; i < numCols; ++i) {
			for (std::size_t j = 0; j < numRows; ++j) {
				mt[j][i] = m[i][j];
			}
		}
		matrix = mt;
	}

	for (auto &row : matrix) {
		auto replicated = LinearAlgebra::ReplicateToBatch(row, cc->GetRingDimension() / 2);
		auto pt = cc->MakePackedPlaintext(replicated);
		auto ct = cc->Encrypt(keys.publicKey, pt);
		ciphertexts.push_back(ct);
	}

	return ciphertexts;
}

LinearAlgebra::Matrix Decrypt(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
                              lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &m,
                              std::size_t n)
{
	lbcrypto::Plaintext p;
	LinearAlgebra::Matrix matrix(n, LinearAlgebra::Vector(n, 0.0));

	cc->Decrypt(keys.secretKey, m, &p);
	p->SetLength(n * n);

	const auto &packed = p->GetPackedValue();

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			matrix[i][j] = packed[i * n + j];
		}
	}

	return matrix;
}

Ciphertext<DCRTPoly> FastRotateCiphertextB(int i, CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                           std::size_t d)
{
	if (i == 0) {
		return cipher_matrix;
	}
	return cc->EvalRotate(cipher_matrix, i * d);
}

Ciphertext<DCRTPoly> FastChangeCipherToMatrixB(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                               std::size_t d)
{
	std::vector<Ciphertext<DCRTPoly>> cipher_vector;
	for (std::size_t c = 0; c < d; ++c) {
		if (c == 0) {
			std::vector<std::int64_t> mask(d * d, 0);
			for (std::size_t r = 0; r < d; ++r) {
				mask[r * d + c] = 1;
			}
			auto plain_mask = cc->MakePackedPlaintext(mask);
			cipher_vector.push_back(cc->EvalMult(cipher_matrix, plain_mask));
			continue;
		}
		auto shifted = cc->EvalRotate(cipher_matrix, c * d);
		std::vector<std::int64_t> mask(d * d, 0);
		for (std::size_t r = 0; r < d; ++r) {
			mask[r * d + c] = 1;
		}
		auto plain_mask = cc->MakePackedPlaintext(mask);
		cipher_vector.push_back(cc->EvalMult(shifted, plain_mask));
	}
	auto dest = cc->EvalAddMany(cipher_vector);
	cc->RelinearizeInPlace(dest);
	cc->RescaleInPlace(dest);
	return dest;
}

Ciphertext<DCRTPoly> FastJiangMatrixMultiplication(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_A,
                                                   Ciphertext<DCRTPoly> &encrypted_B, std::size_t d)
{
	std::vector<Ciphertext<DCRTPoly>> cipher_vector;
	for (std::size_t i = 0; i < d; ++i) {
		auto cipher_tmpA = RotateCiphertextA(i, cc, encrypted_A, d);
		auto cipher_tmpB = FastRotateCiphertextB(i, cc, encrypted_B, d);
		cipher_vector.push_back(cc->EvalMult(cipher_tmpA, cipher_tmpB));
	}
	auto encrypted_C = cc->EvalAddMany(cipher_vector);
	cc->RelinearizeInPlace(encrypted_C);
	cc->RescaleInPlace(encrypted_C);
	return encrypted_C;
}

Ciphertext<DCRTPoly> RobustRotateCiphertextB(int i, CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                             std::size_t d)
{
	if (i == 0) {
		return cipher_matrix;
	}
	auto shifted_up = cc->EvalRotate(cipher_matrix, i * d);
	auto shifted_down = cc->EvalRotate(cipher_matrix, -((int)d - i) * d);
	std::vector<std::int64_t> mask_up(d * d, 0);
	std::vector<std::int64_t> mask_down(d * d, 0);
	for (std::size_t r = 0; r < d - i; ++r) {
		for (std::size_t c = 0; c < d; ++c) {
			mask_up[r * d + c] = 1;
		}
	}
	for (std::size_t r = d - i; r < d; ++r) {
		for (std::size_t c = 0; c < d; ++c) {
			mask_down[r * d + c] = 1;
		}
	}
	auto plain_up = cc->MakePackedPlaintext(mask_up);
	auto plain_down = cc->MakePackedPlaintext(mask_down);
	return cc->EvalAdd(cc->EvalMult(shifted_up, plain_up), cc->EvalMult(shifted_down, plain_down));
}

Ciphertext<DCRTPoly> RobustChangeCipherToMatrixB(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                                 std::size_t d)
{
	std::vector<Ciphertext<DCRTPoly>> cipher_vector;
	for (std::size_t c = 0; c < d; ++c) {
		if (c == 0) {
			std::vector<std::int64_t> mask(d * d, 0);
			for (std::size_t r = 0; r < d; ++r) {
				mask[r * d + c] = 1;
			}
			auto plain_mask = cc->MakePackedPlaintext(mask);
			cipher_vector.push_back(cc->EvalMult(cipher_matrix, plain_mask));
			continue;
		}
		auto shifted_up = cc->EvalRotate(cipher_matrix, c * d);
		auto shifted_down = cc->EvalRotate(cipher_matrix, -((int)d - c) * d);
		std::vector<std::int64_t> mask_up(d * d, 0);
		std::vector<std::int64_t> mask_down(d * d, 0);
		for (std::size_t r = 0; r < d - c; ++r) {
			mask_up[r * d + c] = 1;
		}
		for (std::size_t r = d - c; r < d; ++r) {
			mask_down[r * d + c] = 1;
		}
		auto plain_up = cc->MakePackedPlaintext(mask_up);
		auto plain_down = cc->MakePackedPlaintext(mask_down);
		cipher_vector.push_back(
		    cc->EvalAdd(cc->EvalMult(shifted_up, plain_up), cc->EvalMult(shifted_down, plain_down)));
	}
	auto dest = cc->EvalAddMany(cipher_vector);
	cc->RelinearizeInPlace(dest);
	cc->RescaleInPlace(dest);
	return dest;
}

Ciphertext<DCRTPoly> RobustJiangMatrixMultiplication(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_A,
                                                     Ciphertext<DCRTPoly> &encrypted_B, std::size_t d)
{
	std::vector<Ciphertext<DCRTPoly>> cipher_vector;
	for (std::size_t i = 0; i < d; ++i) {
		auto cipher_tmpA = RotateCiphertextA(i, cc, encrypted_A, d);
		auto cipher_tmpB = RobustRotateCiphertextB(i, cc, encrypted_B, d);
		cipher_vector.push_back(cc->EvalMult(cipher_tmpA, cipher_tmpB));
	}
	auto encrypted_C = cc->EvalAddMany(cipher_vector);
	cc->RelinearizeInPlace(encrypted_C);
	cc->RescaleInPlace(encrypted_C);
	return encrypted_C;
}

Ciphertext<DCRTPoly> OptimalJiangMatrixMultiplication(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_A,
                                                      Ciphertext<DCRTPoly> &encrypted_B, std::size_t d, std::size_t N)
{
	std::size_t slot_capacity = N / 2;
	std::size_t required_slots = d * d;

	if (required_slots == slot_capacity) {
		std::cout << "[FHE Router] Using Fast Hardware-Wrapped Algorithm" << std::endl;
		auto ctA = ChangeCipherToMatrixA(cc, encrypted_A, d);
		auto ctB = FastChangeCipherToMatrixB(cc, encrypted_B, d);
		return FastJiangMatrixMultiplication(cc, ctA, ctB, d);
	} else {
		std::cout << "[FHE Router] Using Robust Explicit-Wrapped Algorithm" << std::endl;
		auto ctA = ChangeCipherToMatrixA(cc, encrypted_A, d);
		auto ctB = RobustChangeCipherToMatrixB(cc, encrypted_B, d);
		return RobustJiangMatrixMultiplication(cc, ctA, ctB, d);
	}
}

Ciphertext<DCRTPoly> RectangularJiangMatrixMultiplication(CryptoContext<DCRTPoly> &cc,
                                                          Ciphertext<DCRTPoly> &encrypted_A_tiled,
                                                          Ciphertext<DCRTPoly> &encrypted_B, std::size_t d,
                                                          std::size_t ell, std::size_t N)
{
	if (ell == 0 || ell > d || (d % ell != 0)) {
		throw std::invalid_argument("RectangularJiangMatrixMultiplication: ell must divide d and be in [1, d]");
	}

	std::size_t slot_capacity = (N > 0) ? (N / 2) : (cc->GetRingDimension() / 2);
	std::size_t required_slots = d * d;
	bool is_power_of_two = (ell & (ell - 1)) == 0;
	if (!is_power_of_two) {
		throw std::invalid_argument("RectangularJiangMatrixMultiplication: ell must be a power of two");
	}

	// Step 1: Pre-transformations
	auto ctA_0 = ChangeCipherToMatrixA(cc, encrypted_A_tiled, d);
	Ciphertext<DCRTPoly> ctB_0;
	if (required_slots == slot_capacity) {
		ctB_0 = FastChangeCipherToMatrixB(cc, encrypted_B, d);
	} else {
		ctB_0 = RobustChangeCipherToMatrixB(cc, encrypted_B, d);
	}

	// Step 2 & 3: Loop ell times
	std::vector<Ciphertext<DCRTPoly>> cipher_vector;
	for (std::size_t i = 0; i < ell; ++i) {
		auto cipher_tmpA = RotateCiphertextA(static_cast<int>(i), cc, ctA_0, d);
		Ciphertext<DCRTPoly> cipher_tmpB;
		if (required_slots == slot_capacity) {
			cipher_tmpB = FastRotateCiphertextB(static_cast<int>(i), cc, ctB_0, d);
		} else {
			cipher_tmpB = RobustRotateCiphertextB(static_cast<int>(i), cc, ctB_0, d);
		}
		cipher_vector.push_back(cc->EvalMult(cipher_tmpA, cipher_tmpB));
	}
	auto ctAB = cc->EvalAddMany(cipher_vector);

	// Step 4: Repeated doubling to sum the (d / ell) row blocks
	std::size_t num_blocks = d / ell;
	for (std::size_t step = 1; step < num_blocks; step <<= 1) {
		auto rotated = cc->EvalRotate(ctAB, static_cast<std::int32_t>(step * ell * d));
		ctAB = cc->EvalAdd(ctAB, rotated);
	}

	cc->RelinearizeInPlace(ctAB);
	cc->RescaleInPlace(ctAB);
	return ctAB;
}

} // namespace fhe
