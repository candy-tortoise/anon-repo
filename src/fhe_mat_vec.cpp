#include "fhe_mat_vec.hpp"

namespace fhe
{

std::vector<std::int32_t> ComputeBSGSKeys(std::size_t d)
{
	std::set<std::int32_t> unique_indices;
	int m1 = std::ceil(std::sqrt(d));
	int m2 = std::ceil((double)d / m1);

	// Baby steps
	for (int k = 1; k < m1; k++) {
		unique_indices.insert(static_cast<std::int32_t>(k));
		unique_indices.insert(static_cast<std::int32_t>(-(d - k)));
	}
	// Giant steps
	for (int j = 1; j < m2; j++) {
		int shift = j * m1;
		unique_indices.insert(static_cast<std::int32_t>(shift));
		unique_indices.insert(static_cast<std::int32_t>(-(d - shift)));
	}
	return std::vector<std::int32_t>(unique_indices.begin(), unique_indices.end());
}

std::vector<std::int64_t> PlaintextRotate(const std::vector<std::int64_t> &vec, int shift)
{
	std::size_t size = vec.size();
	std::int64_t size_i = static_cast<std::int64_t>(size);
	std::vector<std::int64_t> out(size);
	shift = (shift % size_i + size_i) % size_i; // normalize
	for (std::size_t i = 0; i < size; ++i) {
		out[i] = vec[(i + shift) % size_i];
	}
	return out;
}

// ---------------------------------------------------------
// FAST BSGS (Hardware-wrapped)
// ---------------------------------------------------------
Ciphertext<DCRTPoly> FastRotate(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_x, int shift)
{
	if (shift == 0) {
		return encrypted_x;
	}
	return cc->EvalRotate(encrypted_x, shift);
}

Ciphertext<DCRTPoly> FastBabyStepGiantStep(CryptoContext<DCRTPoly> &cc,
                                           const std::vector<std::vector<std::int64_t>> &A_diagonals,
                                           Ciphertext<DCRTPoly> &encrypted_x, int d)
{
	int m1 = std::ceil(std::sqrt(d));
	int m2 = std::ceil((double)d / m1);

	std::vector<Ciphertext<DCRTPoly>> baby_steps(m1);
	baby_steps[0] = encrypted_x;
	for (int k = 1; k < m1; k++) {
		baby_steps[k] = FastRotate(cc, encrypted_x, k);
	}

	std::vector<Ciphertext<DCRTPoly>> giant_steps;
	for (int j = 0; j < m2; j++) {
		std::vector<Ciphertext<DCRTPoly>> inner_sum_vector;
		for (int k = 0; k < m1; k++) {
			int diag_idx = j * m1 + k;
			if (diag_idx >= d) {
				continue;
			}

			auto pt_diag = PlaintextRotate(A_diagonals[diag_idx], -(j * m1));
			auto pt_encoded = cc->MakePackedPlaintext(pt_diag);
			inner_sum_vector.push_back(cc->EvalMult(baby_steps[k], pt_encoded));
		}

		auto inner_sum = cc->EvalAddMany(inner_sum_vector);

		if (j == 0) {
			giant_steps.push_back(inner_sum);
		} else {
			giant_steps.push_back(FastRotate(cc, inner_sum, j * m1));
		}
	}

	return cc->EvalAddMany(giant_steps);
}

// ---------------------------------------------------------
// ROBUST BSGS (Explicitly-wrapped)
// ---------------------------------------------------------
Ciphertext<DCRTPoly> RobustRotate(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_x, int shift, size_t d)
{
	if (shift == 0) {
		return encrypted_x;
	}

	auto left = cc->EvalRotate(encrypted_x, shift);
	auto right = cc->EvalRotate(encrypted_x, -(static_cast<int>(d) - shift));

	std::vector<std::int64_t> mask_left(d, 0);
	std::vector<std::int64_t> mask_right(d, 0);
	for (std::size_t c = 0; c < d - shift; ++c) {
		mask_left[c] = 1;
	}
	for (std::size_t c = d - shift; c < d; ++c) {
		mask_right[c] = 1;
	}

	auto pt_left = cc->MakePackedPlaintext(mask_left);
	auto pt_right = cc->MakePackedPlaintext(mask_right);

	return cc->EvalAdd(cc->EvalMult(left, pt_left), cc->EvalMult(right, pt_right));
}

Ciphertext<DCRTPoly> RobustBabyStepGiantStep(CryptoContext<DCRTPoly> &cc,
                                             const std::vector<std::vector<std::int64_t>> &A_diagonals,
                                             Ciphertext<DCRTPoly> &encrypted_x, int d)
{
	int m1 = std::ceil(std::sqrt(d));
	int m2 = std::ceil((double)d / m1);

	std::vector<Ciphertext<DCRTPoly>> baby_steps(m1);
	baby_steps[0] = encrypted_x;
	for (int k = 1; k < m1; ++k) {
		baby_steps[k] = RobustRotate(cc, encrypted_x, k, d);
	}

	std::vector<Ciphertext<DCRTPoly>> giant_steps;
	for (int j = 0; j < m2; ++j) {
		std::vector<Ciphertext<DCRTPoly>> inner_sum_vector;
		for (int k = 0; k < m1; ++k) {
			int diag_idx = j * m1 + k;
			if (diag_idx >= d) {
				continue;
			}

			auto pt_diag = PlaintextRotate(A_diagonals[diag_idx], -(j * m1));
			auto pt_encoded = cc->MakePackedPlaintext(pt_diag);
			inner_sum_vector.push_back(cc->EvalMult(baby_steps[k], pt_encoded));
		}

		auto inner_sum = cc->EvalAddMany(inner_sum_vector);

		if (j == 0) {
			giant_steps.push_back(inner_sum);
		} else {
			giant_steps.push_back(RobustRotate(cc, inner_sum, j * m1, d));
		}
	}

	return cc->EvalAddMany(giant_steps);
}

// ---------------------------------------------------------
// OPTIMAL ROUTER
// ---------------------------------------------------------
Ciphertext<DCRTPoly> OptimalMatrixVectorMult(CryptoContext<DCRTPoly> &cc,
                                             const std::vector<std::vector<int64_t>> &A_diagonals,
                                             Ciphertext<DCRTPoly> &encrypted_x, int d, std::size_t N)
{
	std::size_t slot_capacity = N / 2;
	if (d == slot_capacity) {
		std::cout << "[FHE Router] Using Fast Hardware-Wrapped BSGS" << std::endl;
		return FastBabyStepGiantStep(cc, A_diagonals, encrypted_x, d);
	} else {
		std::cout << "[FHE Router] Using Robust Explicit-Wrapped BSGS" << std::endl;
		return RobustBabyStepGiantStep(cc, A_diagonals, encrypted_x, d);
	}
}

} // namespace fhe
