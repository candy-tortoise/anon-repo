#include "constants-defs.h"
#include "fhe_context.hpp"
#include "fhe_mat_mul.hpp"
#include "linear_algebra.hpp"
#include "openfhe.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace lbcrypto;
using namespace std::chrono;

struct AttackFHEResult {
	std::string attackName;
	std::string verifierName;
	std::string inputFamily;
	std::size_t totalTrials;
	std::size_t acceptedCount;
	std::size_t incorrectAndAcceptedCount;
	double acceptanceRate;
	double forgerySuccessRate;
	double avgServerTime_s;
	double avgAttackTime_s;
	double avgClientVerifyTime_s;
};

// Uniform field vector in [0, P-1] (strict field distribution)
LinearAlgebra::Vector GenerateUniformFieldVector(std::size_t n, std::int64_t P)
{
	LinearAlgebra::Vector v(n);
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<std::int64_t> dist(0, P - 1);
	for (auto &x : v) {
		x = dist(gen);
	}
	return v;
}

// Uniform field matrix in [0, P-1]
LinearAlgebra::Matrix GenerateUniformFieldMatrix(std::size_t rows, std::size_t cols, std::int64_t P)
{
	LinearAlgebra::Matrix m(rows, LinearAlgebra::Vector(cols));
	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<std::int64_t> dist(0, P - 1);
	for (std::size_t i = 0; i < rows; ++i) {
		for (std::size_t j = 0; j < cols; ++j) {
			m[i][j] = dist(gen);
		}
	}
	return m;
}

// Helper: Modulo P reduction to [0, P-1]
inline std::int64_t ModP(std::int64_t v, std::int64_t P)
{
	std::int64_t r = v % P;
	if (r < 0) {
		r += P;
	}
	return r;
}

// Decrypt an M x M square matrix from packed ciphertext and reduce modulo P
LinearAlgebra::Matrix DecryptSquareMatrix(CryptoContext<DCRTPoly> &cc, KeyPair<DCRTPoly> &keys,
                                          Ciphertext<DCRTPoly> &ct, std::size_t M, std::int64_t P)
{
	Plaintext p;
	cc->Decrypt(keys.secretKey, ct, &p);
	p->SetLength(M * M);
	const auto &packed = p->GetPackedValue();
	LinearAlgebra::Matrix m(M, LinearAlgebra::Vector(M, 0));
	for (std::size_t i = 0; i < M; ++i) {
		for (std::size_t j = 0; j < M; ++j) {
			m[i][j] = ModP(packed[i * M + j], P);
		}
	}
	return m;
}

// Setup BGV CryptoContext with all required rotation keys for matrix dimension M
CryptoContext<DCRTPoly> SetupAttackFHEContext(std::int64_t P, uint32_t N, KeyPair<DCRTPoly> &keys, std::size_t M,
                                              std::size_t d)
{
	CCParams<CryptoContextBGVRNS> parameters;
	parameters.SetMultiplicativeDepth(5);
	parameters.SetPlaintextModulus(P);
	parameters.SetRingDim(N);
	parameters.SetSecurityLevel(HEStd_128_classic);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);

	keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);

	// Collect all required rotation indices for both canvas dimension M and useful dimension d
	auto indices_M = fhe::ComputeRequiredRotationIndices(M);
	auto indices_d = fhe::ComputeRequiredRotationIndices(d);

	std::set<std::int32_t> all_indices(indices_M.begin(), indices_M.end());
	for (auto idx : indices_d) {
		all_indices.insert(idx);
	}

	// Specific slot rotation indices needed for ciphertext-only attack manipulation:
	// Slot shifts for extracting checksum rows at index 16 and 18:
	// row 16: 16*32 = 512, 513, 514
	// row 18: 18*32 = 576, 577, 578
	std::vector<std::int32_t> attack_indices = {1,   -1,   32,  -32,  64,  -64,  480, -480, 512, -512,
	                                            513, -513, 514, -514, 576, -576, 577, -577, 578, -578};
	for (auto idx : attack_indices) {
		all_indices.insert(idx);
	}

	std::vector<std::int32_t> rot_list(all_indices.begin(), all_indices.end());
	cc->EvalRotateKeyGen(keys.secretKey, rot_list);

	return cc;
}

// ============================================================================
// ATTACK 1: Ciphertext-Only Column Transformation Attack
// ============================================================================
std::pair<AttackFHEResult, AttackFHEResult>
RunFHEColumnTransformationAttack(CryptoContext<DCRTPoly> &cc, KeyPair<DCRTPoly> &keys, std::size_t trials,
                                 std::int64_t P, std::size_t d = 16, std::size_t M = 32, uint32_t N = 16384)
{
	AttackFHEResult datasealRes;
	datasealRes.attackName = "Column transformation";
	datasealRes.verifierName = "Algorithm 4 (DataSeal)";
	datasealRes.inputFamily = "Dense random";
	datasealRes.totalTrials = trials;
	datasealRes.acceptedCount = 0;
	datasealRes.incorrectAndAcceptedCount = 0;
	datasealRes.avgServerTime_s = 0.0;
	datasealRes.avgAttackTime_s = 0.0;
	datasealRes.avgClientVerifyTime_s = 0.0;

	AttackFHEResult catchRes;
	catchRes.attackName = "Column transformation";
	catchRes.verifierName = "CATCH-II (Ours)";
	catchRes.inputFamily = "Dense random";
	catchRes.totalTrials = trials;
	catchRes.acceptedCount = 0;
	catchRes.incorrectAndAcceptedCount = 0;
	catchRes.avgServerTime_s = 0.0;
	catchRes.avgAttackTime_s = 0.0;
	catchRes.avgClientVerifyTime_s = 0.0;

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<std::int64_t> alphaDist(2, P - 2);

	// Pre-build public binary masks for column extraction on M x M canvas
	// Rows 0 ... 16 (useful rows 0..15 + checksum row 16) have cols 0 & 1 swapped
	// Row 17 (golden row) and rows >= 18 are preserved untouched
	std::vector<std::int64_t> maskCol0(M * M, 0);
	std::vector<std::int64_t> maskCol1(M * M, 0);
	std::vector<std::int64_t> maskKeep(M * M, 0);

	for (std::size_t i = 0; i < M; ++i) {
		for (std::size_t j = 0; j < M; ++j) {
			std::size_t idx = i * M + j;
			if (i <= 16 && j == 0) {
				maskCol0[idx] = 1;
			} else if (i <= 16 && j == 1) {
				maskCol1[idx] = 1;
			} else {
				maskKeep[idx] = 1;
			}
		}
	}
	auto ptMaskCol0 = cc->MakePackedPlaintext(maskCol0);
	auto ptMaskCol1 = cc->MakePackedPlaintext(maskCol1);
	auto ptMaskKeep = cc->MakePackedPlaintext(maskKeep);

	for (std::size_t t = 0; t < trials; ++t) {
		auto A = GenerateUniformFieldMatrix(d, d, P);
		auto B = GenerateUniformFieldMatrix(d, d, P);

		// Compute true plaintext product for ground-truth correctness check
		LinearAlgebra::Matrix C_true(d, LinearAlgebra::Vector(d, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				std::int64_t sum = 0;
				for (std::size_t k = 0; k < d; ++k) {
					sum = (sum + (A[i][k] % P) * (B[k][j] % P)) % P;
				}
				C_true[i][j] = ModP(sum, P);
			}
		}

		// Client setup: secret key k in [0, P-1], scalar alpha
		auto k = GenerateUniformFieldVector(d, P);
		std::int64_t alpha = alphaDist(gen);

		// Augmented matrix A_tilde (M x M canvas):
		// Rows 0..d-1: A
		// Row d (16): h_A = k^T * A
		// Row d+1 (17): g_A = alpha * h_A
		LinearAlgebra::Matrix A_tilde(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				A_tilde[i][j] = A[i][j];
			}
		}
		// Row 16: h_A = k^T * A
		LinearAlgebra::Vector h_A(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			std::int64_t sum = 0;
			for (std::size_t i = 0; i < d; ++i) {
				sum = (sum + (k[i] % P) * (A[i][j] % P)) % P;
			}
			h_A[j] = ModP(sum, P);
			A_tilde[d][j] = h_A[j];
		}
		// Row 17: g_A = alpha * h_A
		LinearAlgebra::Vector g_golden(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			g_golden[j] = ModP(alpha * h_A[j], P);
			A_tilde[d + 1][j] = g_golden[j];
		}

		// Matrix B padded into M x M canvas
		LinearAlgebra::Matrix B_padded(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				B_padded[i][j] = B[i][j];
			}
		}

		// Encrypt augmented inputs
		auto ctA_tilde =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(A_tilde)));
		auto ctB =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B_padded)));

		// --------------------------------------------------------------------
		// 1. Server Computes Augmented Product ctC_tilde = A_tilde * B in FHE
		// --------------------------------------------------------------------
		auto t_srv_start = high_resolution_clock::now();
		auto ctC_tilde = fhe::OptimalJiangMatrixMultiplication(cc, ctA_tilde, ctB, M, N);
		auto t_srv_stop = high_resolution_clock::now();
		double srv_matmul_time = duration<double>(t_srv_stop - t_srv_start).count();

		// --------------------------------------------------------------------
		// 2. Server Executes CIPHERTEXT-ONLY Column Swap Attack
		// Uses only public masks, EvalMult, and EvalRotate. Zero plaintext access!
		// --------------------------------------------------------------------
		auto t_atk_start = high_resolution_clock::now();
		// Isolate columns 0 and 1
		auto ctCol0 = cc->EvalMult(ctC_tilde, ptMaskCol0);
		auto ctCol1 = cc->EvalMult(ctC_tilde, ptMaskCol1);
		auto ctKeep = cc->EvalMult(ctC_tilde, ptMaskKeep);

		// Shift col 0 to col 1 (shift right by 1, rotate index -1)
		auto ctCol0_to_1 = cc->EvalRotate(ctCol0, -1);
		// Shift col 1 to col 0 (shift left by 1, rotate index +1)
		auto ctCol1_to_0 = cc->EvalRotate(ctCol1, 1);

		// Reassemble tampered ciphertext
		auto ctC_tampered = cc->EvalAdd(ctKeep, cc->EvalAdd(ctCol0_to_1, ctCol1_to_0));
		auto t_atk_stop = high_resolution_clock::now();
		double srv_attack_time = duration<double>(t_atk_stop - t_atk_start).count();

		// --------------------------------------------------------------------
		// 3. Client Decrypts Returned Ciphertext (Including Checksum Rows)
		// --------------------------------------------------------------------
		auto t_ds_verify_start = high_resolution_clock::now();
		auto decC_tilde = DecryptSquareMatrix(cc, keys, ctC_tampered, M, P);

		// Extract returned useful matrix, checksum row h_ret, and golden row g_ret
		LinearAlgebra::Matrix C_ret(d, LinearAlgebra::Vector(d, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				C_ret[i][j] = decC_tilde[i][j];
			}
		}
		LinearAlgebra::Vector h_ret(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			h_ret[j] = decC_tilde[d][j];
		}
		LinearAlgebra::Vector g_ret(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			g_ret[j] = decC_tilde[d + 1][j];
		}

		// Check if returned computation is incorrect
		bool is_incorrect = false;
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				if (C_ret[i][j] != C_true[i][j]) {
					is_incorrect = true;
					break;
				}
			}
			if (is_incorrect) {
				break;
			}
		}

		// DataSeal Algorithm 4 Checks:
		// Check 1: Returned golden row matches client's private golden reference
		// Note: g_ret = g_A * B = (alpha * h_A) * B.
		// Since row 17 was untouched by attack, g_ret equals client's reference!
		// Client computes expected golden vector g_expected = alpha * (k^T * C_true)
		// but client verifies against returned g_ret!
		bool ds_golden_ok = true;
		for (std::size_t j = 0; j < d; ++j) {
			// In Algorithm 4, client checks returned g_ret against precomputed g_golden_B
			// where g_golden_B = g_A * B
			std::int64_t expected_g = 0;
			for (std::size_t i = 0; i < d; ++i) {
				expected_g = (expected_g + (A_tilde[d + 1][i] % P) * (B[i][j] % P)) % P;
			}
			expected_g = ModP(expected_g, P);
			if (g_ret[j] != expected_g) {
				ds_golden_ok = false;
				break;
			}
		}

		// Check 2: k^T * C_ret == h_ret
		bool ds_checksum_ok = true;
		for (std::size_t j = 0; j < d; ++j) {
			std::int64_t sum = 0;
			for (std::size_t i = 0; i < d; ++i) {
				sum = (sum + (k[i] % P) * (C_ret[i][j] % P)) % P;
			}
			sum = ModP(sum, P);
			if (sum != h_ret[j]) {
				ds_checksum_ok = false;
				break;
			}
		}
		auto t_ds_verify_stop = high_resolution_clock::now();
		double ds_verify_time = duration<double>(t_ds_verify_stop - t_ds_verify_start).count();

		bool ds_accepted = ds_golden_ok && ds_checksum_ok;
		if (ds_accepted) {
			datasealRes.acceptedCount++;
		}
		if (ds_accepted && is_incorrect) {
			datasealRes.incorrectAndAcceptedCount++;
		}

		datasealRes.avgServerTime_s += srv_matmul_time;
		datasealRes.avgAttackTime_s += srv_attack_time;
		datasealRes.avgClientVerifyTime_s += ds_verify_time;

		// --------------------------------------------------------------------
		// 4. CATCH-II Verification on the IDENTICAL Corrupted Matrix C_ret
		// --------------------------------------------------------------------
		auto t_catch_verify_start = high_resolution_clock::now();

		// Client draws s = 8 FRESH random challenges in [0, P-1] (128-bit consistency soundness)
		std::size_t s_challenges = 8;
		std::vector<LinearAlgebra::Vector> challenges(s_challenges);
		LinearAlgebra::Matrix P_rect(s_challenges, LinearAlgebra::Vector(d, 0));

		for (std::size_t l = 0; l < s_challenges; ++l) {
			challenges[l] = GenerateUniformFieldVector(d, P);
			auto vA = LinearAlgebra::MultiplyVectorAndMatrix(challenges[l], A);
			for (std::size_t j = 0; j < d; ++j) {
				P_rect[l][j] = ModP(vA[j], P);
			}
		}

		// Pack challenges into d x d tiled matrix for Rectangular Jiang MatMul
		auto P_tiled = LinearAlgebra::TileMatrixVertically(P_rect, d);
		auto ctP =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_tiled)));

		// B matrix formatted as d x d
		auto ctB_d = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B)));

		// Server evaluates reference verification MatMul under FHE: ctV = P * B
		auto t_catch_srv_start = high_resolution_clock::now();
		auto ctV = fhe::RectangularJiangMatrixMultiplication(cc, ctP, ctB_d, d, s_challenges, N);
		auto t_catch_srv_stop = high_resolution_clock::now();
		double catch_srv_time = duration<double>(t_catch_srv_stop - t_catch_srv_start).count();

		// Client decrypts ctV and verifies against returned C_ret
		auto V_mat = DecryptSquareMatrix(cc, keys, ctV, d, P);

		bool catch_accepted = true;
		for (std::size_t l = 0; l < s_challenges; ++l) {
			auto rC = LinearAlgebra::MultiplyVectorAndMatrix(challenges[l], C_ret);
			for (std::size_t j = 0; j < d; ++j) {
				if (ModP(V_mat[l][j], P) != ModP(rC[j], P)) {
					catch_accepted = false;
					break;
				}
			}
			if (!catch_accepted) {
				break;
			}
		}
		auto t_catch_verify_stop = high_resolution_clock::now();
		// CATCH client verify time excludes server reference multiplication
		double catch_verify_time =
		    duration<double>(t_catch_verify_stop - t_catch_verify_start).count() - catch_srv_time;

		if (catch_accepted) {
			catchRes.acceptedCount++;
		}
		if (catch_accepted && is_incorrect) {
			catchRes.incorrectAndAcceptedCount++;
		}

		catchRes.avgServerTime_s += (srv_matmul_time + catch_srv_time);
		catchRes.avgAttackTime_s += srv_attack_time;
		catchRes.avgClientVerifyTime_s += catch_verify_time;
	}

	datasealRes.acceptanceRate = (double)datasealRes.acceptedCount / datasealRes.totalTrials;
	datasealRes.forgerySuccessRate = (double)datasealRes.incorrectAndAcceptedCount / datasealRes.totalTrials;
	datasealRes.avgServerTime_s /= datasealRes.totalTrials;
	datasealRes.avgAttackTime_s /= datasealRes.totalTrials;
	datasealRes.avgClientVerifyTime_s /= datasealRes.totalTrials;

	catchRes.acceptanceRate = (double)catchRes.acceptedCount / catchRes.totalTrials;
	catchRes.forgerySuccessRate = (double)catchRes.incorrectAndAcceptedCount / catchRes.totalTrials;
	catchRes.avgServerTime_s /= catchRes.totalTrials;
	catchRes.avgAttackTime_s /= catchRes.totalTrials;
	catchRes.avgClientVerifyTime_s /= catchRes.totalTrials;

	return {datasealRes, catchRes};
}

// ============================================================================
// ATTACK 2: Ciphertext-Only Bilinear Cancellation Attack (Embedded I_2)
// ============================================================================
std::pair<AttackFHEResult, AttackFHEResult>
RunFHEBilinearCancellationAttack(CryptoContext<DCRTPoly> &cc, KeyPair<DCRTPoly> &keys, std::size_t trials,
                                 std::int64_t P, std::size_t d = 16, std::size_t M = 32, uint32_t N = 16384)
{
	AttackFHEResult datasealRes;
	datasealRes.attackName = "Bilinear cancellation";
	datasealRes.verifierName = "Combined (DataSeal)";
	datasealRes.inputFamily = "Identity (I_2 embedded)";
	datasealRes.totalTrials = trials;
	datasealRes.acceptedCount = 0;
	datasealRes.incorrectAndAcceptedCount = 0;
	datasealRes.avgServerTime_s = 0.0;
	datasealRes.avgAttackTime_s = 0.0;
	datasealRes.avgClientVerifyTime_s = 0.0;

	AttackFHEResult catchRes;
	catchRes.attackName = "Bilinear cancellation";
	catchRes.verifierName = "CATCH-II (Ours)";
	catchRes.inputFamily = "Identity (I_2 embedded)";
	catchRes.totalTrials = trials;
	catchRes.acceptedCount = 0;
	catchRes.incorrectAndAcceptedCount = 0;
	catchRes.avgServerTime_s = 0.0;
	catchRes.avgAttackTime_s = 0.0;
	catchRes.avgClientVerifyTime_s = 0.0;

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<std::int64_t> alphaDist(2, P - 2);

	// Pre-build public extraction masks for slot 512 (h[0]) and slot 513 (h[1])
	std::size_t slot_h0 = d * M + 0; // 16 * 32 = 512
	std::size_t slot_h1 = d * M + 1; // 16 * 32 + 1 = 513

	std::vector<std::int64_t> maskH0(M * M, 0);
	std::vector<std::int64_t> maskH1(M * M, 0);
	maskH0[slot_h0] = 1;
	maskH1[slot_h1] = 1;

	auto ptMaskH0 = cc->MakePackedPlaintext(maskH0);
	auto ptMaskH1 = cc->MakePackedPlaintext(maskH1);

	for (std::size_t t = 0; t < trials; ++t) {
		// A = I_d, B has I_2 embedded in top-left
		LinearAlgebra::Matrix A(d, LinearAlgebra::Vector(d, 0));
		for (std::size_t i = 0; i < d; ++i) {
			A[i][i] = 1;
		}

		LinearAlgebra::Matrix B(d, LinearAlgebra::Vector(d, 0));
		B[0][0] = 1;
		B[1][1] = 1;

		LinearAlgebra::Matrix C_true = B;

		auto k = GenerateUniformFieldVector(d, P);
		std::int64_t alpha = alphaDist(gen);

		// Augmented A_tilde (M x M canvas)
		LinearAlgebra::Matrix A_tilde(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				A_tilde[i][j] = A[i][j];
			}
		}
		// Row 16: h_A = k^T * A = k
		for (std::size_t j = 0; j < d; ++j) {
			A_tilde[d][j] = k[j];
		}
		// Row 17: g_A = alpha * k
		for (std::size_t j = 0; j < d; ++j) {
			A_tilde[d + 1][j] = ModP(alpha * k[j], P);
		}

		// Padded B
		LinearAlgebra::Matrix B_padded(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				B_padded[i][j] = B[i][j];
			}
		}

		auto ctA_tilde =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(A_tilde)));
		auto ctB =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B_padded)));

		// 1. Server Computes Augmented Product
		auto t_srv_start = high_resolution_clock::now();
		auto ctC_tilde = fhe::OptimalJiangMatrixMultiplication(cc, ctA_tilde, ctB, M, N);
		auto t_srv_stop = high_resolution_clock::now();
		double srv_matmul_time = duration<double>(t_srv_stop - t_srv_start).count();

		// 2. Server Executes CIPHERTEXT-ONLY Bilinear Cancellation Attack
		// Extracts encrypted h[0] and h[1], shifts h[1] to (0,0) and -h[0] to (1,0)
		auto t_atk_start = high_resolution_clock::now();
		auto ctH0 = cc->EvalMult(ctC_tilde, ptMaskH0);
		auto ctH1 = cc->EvalMult(ctC_tilde, ptMaskH1);

		// Rotate h[1] from slot 513 to slot 0: EvalRotate(ct, 513)
		auto ctDelta00 = cc->EvalRotate(ctH1, 513);

		// Rotate h[0] from slot 512 to slot 32 (row 1, col 0): 512 - 32 = 480
		// EvalRotate(ct, 480), then negate
		auto ctDelta10 = cc->EvalNegate(cc->EvalRotate(ctH0, 480));

		// Inject cancellation perturbation into ciphertext homomorphically
		auto ctC_tampered = cc->EvalAdd(ctC_tilde, cc->EvalAdd(ctDelta00, ctDelta10));
		auto t_atk_stop = high_resolution_clock::now();
		double srv_attack_time = duration<double>(t_atk_stop - t_atk_start).count();

		// 3. Client Decrypts Returned Ciphertext
		auto t_ds_verify_start = high_resolution_clock::now();
		auto decC_tilde = DecryptSquareMatrix(cc, keys, ctC_tampered, M, P);

		LinearAlgebra::Matrix C_ret(d, LinearAlgebra::Vector(d, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				C_ret[i][j] = decC_tilde[i][j];
			}
		}
		LinearAlgebra::Vector h_ret(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			h_ret[j] = decC_tilde[d][j];
		}
		LinearAlgebra::Vector g_ret(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			g_ret[j] = decC_tilde[d + 1][j];
		}

		// Expected golden vector g_expected = alpha * h_ret
		LinearAlgebra::Vector g_golden(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			std::int64_t expected_g = 0;
			for (std::size_t i = 0; i < d; ++i) {
				expected_g = (expected_g + (A_tilde[d + 1][i] % P) * (B[i][j] % P)) % P;
			}
			g_golden[j] = ModP(expected_g, P);
		}

		bool is_incorrect = false;
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				if (C_ret[i][j] != C_true[i][j]) {
					is_incorrect = true;
					break;
				}
			}
			if (is_incorrect) {
				break;
			}
		}

		// Combined Verifier Checks:
		// Check 1: g_ret == g_golden (Client's reference)
		bool check1 = true;
		for (std::size_t j = 0; j < d; ++j) {
			if (g_ret[j] != g_golden[j]) {
				check1 = false;
				break;
			}
		}

		// Check 2: k^T * C_ret == h_ret
		bool check2 = true;
		for (std::size_t j = 0; j < d; ++j) {
			std::int64_t sum = 0;
			for (std::size_t i = 0; i < d; ++i) {
				sum = (sum + (k[i] % P) * (C_ret[i][j] % P)) % P;
			}
			sum = ModP(sum, P);
			if (sum != h_ret[j]) {
				check2 = false;
				break;
			}
		}

		// Check 3: Link Check g_ret == alpha * h_ret
		bool check3 = true;
		for (std::size_t j = 0; j < d; ++j) {
			if (g_ret[j] != ModP(alpha * h_ret[j], P)) {
				check3 = false;
				break;
			}
		}
		auto t_ds_verify_stop = high_resolution_clock::now();
		double ds_verify_time = duration<double>(t_ds_verify_stop - t_ds_verify_start).count();

		bool ds_accepted = check1 && check2 && check3;
		if (ds_accepted) {
			datasealRes.acceptedCount++;
		}
		if (ds_accepted && is_incorrect) {
			datasealRes.incorrectAndAcceptedCount++;
		}

		datasealRes.avgServerTime_s += srv_matmul_time;
		datasealRes.avgAttackTime_s += srv_attack_time;
		datasealRes.avgClientVerifyTime_s += ds_verify_time;

		// 4. CATCH-II Verification on the IDENTICAL Corrupted Matrix C_ret
		auto t_catch_verify_start = high_resolution_clock::now();
		std::size_t s_challenges = 8;
		std::vector<LinearAlgebra::Vector> challenges(s_challenges);
		LinearAlgebra::Matrix P_rect(s_challenges, LinearAlgebra::Vector(d, 0));

		for (std::size_t l = 0; l < s_challenges; ++l) {
			challenges[l] = GenerateUniformFieldVector(d, P);
			auto vA = LinearAlgebra::MultiplyVectorAndMatrix(challenges[l], A);
			for (std::size_t j = 0; j < d; ++j) {
				P_rect[l][j] = ModP(vA[j], P);
			}
		}

		auto P_tiled = LinearAlgebra::TileMatrixVertically(P_rect, d);
		auto ctP =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_tiled)));
		auto ctB_d = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B)));

		auto t_catch_srv_start = high_resolution_clock::now();
		auto ctV = fhe::RectangularJiangMatrixMultiplication(cc, ctP, ctB_d, d, s_challenges, N);
		auto t_catch_srv_stop = high_resolution_clock::now();
		double catch_srv_time = duration<double>(t_catch_srv_stop - t_catch_srv_start).count();

		auto V_mat = DecryptSquareMatrix(cc, keys, ctV, d, P);

		bool catch_accepted = true;
		for (std::size_t l = 0; l < s_challenges; ++l) {
			auto rC = LinearAlgebra::MultiplyVectorAndMatrix(challenges[l], C_ret);
			for (std::size_t j = 0; j < d; ++j) {
				if (ModP(V_mat[l][j], P) != ModP(rC[j], P)) {
					catch_accepted = false;
					break;
				}
			}
			if (!catch_accepted) {
				break;
			}
		}
		auto t_catch_verify_stop = high_resolution_clock::now();
		double catch_verify_time =
		    duration<double>(t_catch_verify_stop - t_catch_verify_start).count() - catch_srv_time;

		if (catch_accepted) {
			catchRes.acceptedCount++;
		}
		if (catch_accepted && is_incorrect) {
			catchRes.incorrectAndAcceptedCount++;
		}

		catchRes.avgServerTime_s += (srv_matmul_time + catch_srv_time);
		catchRes.avgAttackTime_s += srv_attack_time;
		catchRes.avgClientVerifyTime_s += catch_verify_time;
	}

	datasealRes.acceptanceRate = (double)datasealRes.acceptedCount / datasealRes.totalTrials;
	datasealRes.forgerySuccessRate = (double)datasealRes.incorrectAndAcceptedCount / datasealRes.totalTrials;
	datasealRes.avgServerTime_s /= datasealRes.totalTrials;
	datasealRes.avgAttackTime_s /= datasealRes.totalTrials;
	datasealRes.avgClientVerifyTime_s /= datasealRes.totalTrials;

	catchRes.acceptanceRate = (double)catchRes.acceptedCount / catchRes.totalTrials;
	catchRes.forgerySuccessRate = (double)catchRes.incorrectAndAcceptedCount / catchRes.totalTrials;
	catchRes.avgServerTime_s /= catchRes.totalTrials;
	catchRes.avgAttackTime_s /= catchRes.totalTrials;
	catchRes.avgClientVerifyTime_s /= catchRes.totalTrials;

	return {datasealRes, catchRes};
}

// ============================================================================
// ATTACK 3: Ciphertext-Only Two-Key Cancellation Attack (3D Cross-Product)
// ============================================================================
std::pair<AttackFHEResult, AttackFHEResult> RunFHETwoKeyCancellationAttack(CryptoContext<DCRTPoly> &cc,
                                                                           KeyPair<DCRTPoly> &keys, std::size_t trials,
                                                                           std::int64_t P, std::size_t d = 16,
                                                                           std::size_t M = 32, uint32_t N = 16384)
{
	AttackFHEResult datasealRes;
	datasealRes.attackName = "Two-key cancellation";
	datasealRes.verifierName = "Two-key combined (DataSeal)";
	datasealRes.inputFamily = "Rank-3 identity (I_3)";
	datasealRes.totalTrials = trials;
	datasealRes.acceptedCount = 0;
	datasealRes.incorrectAndAcceptedCount = 0;
	datasealRes.avgServerTime_s = 0.0;
	datasealRes.avgAttackTime_s = 0.0;
	datasealRes.avgClientVerifyTime_s = 0.0;

	AttackFHEResult catchRes;
	catchRes.attackName = "Two-key cancellation";
	catchRes.verifierName = "CATCH-II (Ours)";
	catchRes.inputFamily = "Rank-3 identity (I_3)";
	catchRes.totalTrials = trials;
	catchRes.acceptedCount = 0;
	catchRes.incorrectAndAcceptedCount = 0;
	catchRes.avgServerTime_s = 0.0;
	catchRes.avgAttackTime_s = 0.0;
	catchRes.avgClientVerifyTime_s = 0.0;

	std::random_device rd;
	std::mt19937_64 gen(rd());
	std::uniform_int_distribution<std::int64_t> alphaDist(2, P - 2);

	// Pre-build public extraction masks for:
	// Row 16 (h1): slot 512 (h1[0]), slot 513 (h1[1]), slot 514 (h1[2])
	// Row 18 (h2): slot 576 (h2[0]), slot 577 (h2[1]), slot 578 (h2[2])
	std::vector<std::int64_t> maskH1_0(M * M, 0), maskH1_1(M * M, 0), maskH1_2(M * M, 0);
	std::vector<std::int64_t> maskH2_0(M * M, 0), maskH2_1(M * M, 0), maskH2_2(M * M, 0);

	maskH1_0[16 * M + 0] = 1;
	maskH1_1[16 * M + 1] = 1;
	maskH1_2[16 * M + 2] = 1;
	maskH2_0[18 * M + 0] = 1;
	maskH2_1[18 * M + 1] = 1;
	maskH2_2[18 * M + 2] = 1;

	auto ptH1_0 = cc->MakePackedPlaintext(maskH1_0);
	auto ptH1_1 = cc->MakePackedPlaintext(maskH1_1);
	auto ptH1_2 = cc->MakePackedPlaintext(maskH1_2);

	auto ptH2_0 = cc->MakePackedPlaintext(maskH2_0);
	auto ptH2_1 = cc->MakePackedPlaintext(maskH2_1);
	auto ptH2_2 = cc->MakePackedPlaintext(maskH2_2);

	for (std::size_t t = 0; t < trials; ++t) {
		LinearAlgebra::Matrix A(d, LinearAlgebra::Vector(d, 0));
		for (std::size_t i = 0; i < d; ++i) {
			A[i][i] = 1;
		}

		LinearAlgebra::Matrix B(d, LinearAlgebra::Vector(d, 0));
		B[0][0] = 1;
		B[1][1] = 1;
		B[2][2] = 1;

		LinearAlgebra::Matrix C_true = B;

		auto k1 = GenerateUniformFieldVector(d, P);
		auto k2 = GenerateUniformFieldVector(d, P);
		std::int64_t alpha1 = alphaDist(gen);
		std::int64_t alpha2 = alphaDist(gen);

		// Augmented A_tilde (M x M canvas):
		// Row 16: h1 = k1
		// Row 17: g1 = alpha1 * k1
		// Row 18: h2 = k2
		// Row 19: g2 = alpha2 * k2
		LinearAlgebra::Matrix A_tilde(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				A_tilde[i][j] = A[i][j];
			}
		}
		for (std::size_t j = 0; j < d; ++j) {
			A_tilde[16][j] = k1[j];
			A_tilde[17][j] = ModP(alpha1 * k1[j], P);
			A_tilde[18][j] = k2[j];
			A_tilde[19][j] = ModP(alpha2 * k2[j], P);
		}

		LinearAlgebra::Matrix B_padded(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				B_padded[i][j] = B[i][j];
			}
		}

		auto ctA_tilde =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(A_tilde)));
		auto ctB =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B_padded)));

		// 1. Server Computes Augmented Product
		auto t_srv_start = high_resolution_clock::now();
		auto ctC_tilde = fhe::OptimalJiangMatrixMultiplication(cc, ctA_tilde, ctB, M, N);
		auto t_srv_stop = high_resolution_clock::now();
		double srv_matmul_time = duration<double>(t_srv_stop - t_srv_start).count();

		// 2. Server Executes CIPHERTEXT-ONLY Two-Key Cross-Product Attack
		// Evaluates e0, e1, e2 via homomorphic ciphertext-ciphertext multiplications (EvalMult)
		auto t_atk_start = high_resolution_clock::now();
		auto ct_h1_0 = cc->EvalMult(ctC_tilde, ptH1_0);
		auto ct_h1_1 = cc->EvalMult(ctC_tilde, ptH1_1);
		auto ct_h1_2 = cc->EvalMult(ctC_tilde, ptH1_2);

		auto ct_h2_0 = cc->EvalMult(ctC_tilde, ptH2_0);
		auto ct_h2_1 = cc->EvalMult(ctC_tilde, ptH2_1);
		auto ct_h2_2 = cc->EvalMult(ctC_tilde, ptH2_2);

		// Compute e0 = h1[1]*h2[2] - h1[2]*h2[1] at slot 0:
		// Move h1[1] (513) to 0, h2[2] (578) to 0 -> EvalMult
		auto ct_a1 = cc->EvalRotate(ct_h1_1, 513);
		auto ct_b1 = cc->EvalRotate(ct_h2_2, 578);
		auto term1_e0 = cc->EvalMult(ct_a1, ct_b1);

		auto ct_c1 = cc->EvalRotate(ct_h1_2, 514);
		auto ct_d1 = cc->EvalRotate(ct_h2_1, 577);
		auto term2_e0 = cc->EvalMult(ct_c1, ct_d1);
		auto ct_e0 = cc->EvalSub(term1_e0, term2_e0); // e0 at slot 0 (which is slot (0,0))

		// Compute e1 = h1[2]*h2[0] - h1[0]*h2[2] at slot 0:
		auto ct_a2 = cc->EvalRotate(ct_h1_2, 514);
		auto ct_b2 = cc->EvalRotate(ct_h2_0, 576);
		auto term1_e1 = cc->EvalMult(ct_a2, ct_b2);

		auto ct_c2 = cc->EvalRotate(ct_h1_0, 512);
		auto ct_d2 = cc->EvalRotate(ct_h2_2, 578);
		auto term2_e1 = cc->EvalMult(ct_c2, ct_d2);
		auto e1_at_0 = cc->EvalSub(term1_e1, term2_e1);
		// Move e1 from slot 0 to slot (1, 0) = slot 32: EvalRotate(ct, -32)
		auto ct_e1 = cc->EvalRotate(e1_at_0, -32);

		// Compute e2 = h1[0]*h2[1] - h1[1]*h2[0] at slot 0:
		auto ct_a3 = cc->EvalRotate(ct_h1_0, 512);
		auto ct_b3 = cc->EvalRotate(ct_h2_1, 577);
		auto term1_e2 = cc->EvalMult(ct_a3, ct_b3);

		auto ct_c3 = cc->EvalRotate(ct_h1_1, 513);
		auto ct_d3 = cc->EvalRotate(ct_h2_0, 576);
		auto term2_e2 = cc->EvalMult(ct_c3, ct_d3);
		auto e2_at_0 = cc->EvalSub(term1_e2, term2_e2);
		// Move e2 from slot 0 to slot (2, 0) = slot 64: EvalRotate(ct, -64)
		auto ct_e2 = cc->EvalRotate(e2_at_0, -64);

		// Inject cross-product perturbation into ciphertext
		if (ct_e0->GetLevel() > ctC_tilde->GetLevel()) {
			cc->LevelReduceInPlace(ctC_tilde, nullptr, ct_e0->GetLevel() - ctC_tilde->GetLevel());
		}
		auto ctC_tampered = cc->EvalAdd(ctC_tilde, cc->EvalAdd(ct_e0, cc->EvalAdd(ct_e1, ct_e2)));
		auto t_atk_stop = high_resolution_clock::now();
		double srv_attack_time = duration<double>(t_atk_stop - t_atk_start).count();

		// 3. Client Decrypts Returned Ciphertext
		auto t_ds_verify_start = high_resolution_clock::now();
		auto decC_tilde = DecryptSquareMatrix(cc, keys, ctC_tampered, M, P);

		LinearAlgebra::Matrix C_ret(d, LinearAlgebra::Vector(d, 0));
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				C_ret[i][j] = decC_tilde[i][j];
			}
		}
		LinearAlgebra::Vector h1_ret(d, 0), g1_ret(d, 0), h2_ret(d, 0), g2_ret(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			h1_ret[j] = decC_tilde[16][j];
			g1_ret[j] = decC_tilde[17][j];
			h2_ret[j] = decC_tilde[18][j];
			g2_ret[j] = decC_tilde[19][j];
		}

		// Client expected golden rows
		LinearAlgebra::Vector g1_golden(d, 0), g2_golden(d, 0);
		for (std::size_t j = 0; j < d; ++j) {
			std::int64_t exp_g1 = 0, exp_g2 = 0;
			for (std::size_t i = 0; i < d; ++i) {
				exp_g1 = (exp_g1 + (A_tilde[17][i] % P) * (B[i][j] % P)) % P;
				exp_g2 = (exp_g2 + (A_tilde[19][i] % P) * (B[i][j] % P)) % P;
			}
			g1_golden[j] = ModP(exp_g1, P);
			g2_golden[j] = ModP(exp_g2, P);
		}

		bool is_incorrect = false;
		for (std::size_t i = 0; i < d; ++i) {
			for (std::size_t j = 0; j < d; ++j) {
				if (C_ret[i][j] != C_true[i][j]) {
					is_incorrect = true;
					break;
				}
			}
			if (is_incorrect) {
				break;
			}
		}

		// Two-Key Combined Verifier Checks:
		bool k1_ok = true;
		for (std::size_t j = 0; j < d; ++j) {
			std::int64_t sum1 = 0;
			for (std::size_t i = 0; i < d; ++i) {
				sum1 = (sum1 + (k1[i] % P) * (C_ret[i][j] % P)) % P;
			}
			if (ModP(sum1, P) != h1_ret[j]) {
				k1_ok = false;
				break;
			}
			if (g1_ret[j] != g1_golden[j]) {
				k1_ok = false;
				break;
			}
			if (g1_ret[j] != ModP(alpha1 * h1_ret[j], P)) {
				k1_ok = false;
				break;
			}
		}

		bool k2_ok = true;
		for (std::size_t j = 0; j < d; ++j) {
			std::int64_t sum2 = 0;
			for (std::size_t i = 0; i < d; ++i) {
				sum2 = (sum2 + (k2[i] % P) * (C_ret[i][j] % P)) % P;
			}
			if (ModP(sum2, P) != h2_ret[j]) {
				k2_ok = false;
				break;
			}
			if (g2_ret[j] != g2_golden[j]) {
				k2_ok = false;
				break;
			}
			if (g2_ret[j] != ModP(alpha2 * h2_ret[j], P)) {
				k2_ok = false;
				break;
			}
		}
		auto t_ds_verify_stop = high_resolution_clock::now();
		double ds_verify_time = duration<double>(t_ds_verify_stop - t_ds_verify_start).count();

		bool ds_accepted = k1_ok && k2_ok;
		if (ds_accepted) {
			datasealRes.acceptedCount++;
		}
		if (ds_accepted && is_incorrect) {
			datasealRes.incorrectAndAcceptedCount++;
		}

		datasealRes.avgServerTime_s += srv_matmul_time;
		datasealRes.avgAttackTime_s += srv_attack_time;
		datasealRes.avgClientVerifyTime_s += ds_verify_time;

		// 4. CATCH-II Verification on the IDENTICAL Corrupted Matrix C_ret
		auto t_catch_verify_start = high_resolution_clock::now();
		std::size_t s_challenges = 8;
		std::vector<LinearAlgebra::Vector> challenges(s_challenges);
		LinearAlgebra::Matrix P_rect(s_challenges, LinearAlgebra::Vector(d, 0));

		for (std::size_t l = 0; l < s_challenges; ++l) {
			challenges[l] = GenerateUniformFieldVector(d, P);
			auto vA = LinearAlgebra::MultiplyVectorAndMatrix(challenges[l], A);
			for (std::size_t j = 0; j < d; ++j) {
				P_rect[l][j] = ModP(vA[j], P);
			}
		}

		auto P_tiled = LinearAlgebra::TileMatrixVertically(P_rect, d);
		auto ctP =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_tiled)));
		auto ctB_d = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B)));

		auto t_catch_srv_start = high_resolution_clock::now();
		auto ctV = fhe::RectangularJiangMatrixMultiplication(cc, ctP, ctB_d, d, s_challenges, N);
		auto t_catch_srv_stop = high_resolution_clock::now();
		double catch_srv_time = duration<double>(t_catch_srv_stop - t_catch_srv_start).count();

		auto V_mat = DecryptSquareMatrix(cc, keys, ctV, d, P);

		bool catch_accepted = true;
		for (std::size_t l = 0; l < s_challenges; ++l) {
			auto rC = LinearAlgebra::MultiplyVectorAndMatrix(challenges[l], C_ret);
			for (std::size_t j = 0; j < d; ++j) {
				if (ModP(V_mat[l][j], P) != ModP(rC[j], P)) {
					catch_accepted = false;
					break;
				}
			}
			if (!catch_accepted) {
				break;
			}
		}
		auto t_catch_verify_stop = high_resolution_clock::now();
		double catch_verify_time =
		    duration<double>(t_catch_verify_stop - t_catch_verify_start).count() - catch_srv_time;

		if (catch_accepted) {
			catchRes.acceptedCount++;
		}
		if (catch_accepted && is_incorrect) {
			catchRes.incorrectAndAcceptedCount++;
		}

		catchRes.avgServerTime_s += (srv_matmul_time + catch_srv_time);
		catchRes.avgAttackTime_s += srv_attack_time;
		catchRes.avgClientVerifyTime_s += catch_verify_time;
	}

	datasealRes.acceptanceRate = (double)datasealRes.acceptedCount / datasealRes.totalTrials;
	datasealRes.forgerySuccessRate = (double)datasealRes.incorrectAndAcceptedCount / datasealRes.totalTrials;
	datasealRes.avgServerTime_s /= datasealRes.totalTrials;
	datasealRes.avgAttackTime_s /= datasealRes.totalTrials;
	datasealRes.avgClientVerifyTime_s /= datasealRes.totalTrials;

	catchRes.acceptanceRate = (double)catchRes.acceptedCount / catchRes.totalTrials;
	catchRes.forgerySuccessRate = (double)catchRes.incorrectAndAcceptedCount / catchRes.totalTrials;
	catchRes.avgServerTime_s /= catchRes.totalTrials;
	catchRes.avgAttackTime_s /= catchRes.totalTrials;
	catchRes.avgClientVerifyTime_s /= catchRes.totalTrials;

	return {datasealRes, catchRes};
}

void PrintCsvHeader()
{
	std::cout << "Attack,Verifier,Input family,Trials,Accepted,Incorrect and accepted,Acceptance rate,Forgery "
	             "success rate,AvgServerTime_s,AvgAttackTime_s,AvgClientVerifyTime_s\n";
}

void PrintCsvResult(const AttackFHEResult &r)
{
	std::cout << "\"" << r.attackName << "\",\"" << r.verifierName << "\",\"" << r.inputFamily << "\","
	          << r.totalTrials << "," << r.acceptedCount << "," << r.incorrectAndAcceptedCount << "," << std::fixed
	          << std::setprecision(4) << r.acceptanceRate << "," << std::fixed << std::setprecision(4)
	          << r.forgerySuccessRate << "," << std::setprecision(6) << r.avgServerTime_s << ","
	          << r.avgAttackTime_s << "," << r.avgClientVerifyTime_s << "\n";
}

int main(int argc, char **argv)
{
	std::size_t trials = 20;
	std::int64_t P = 65537;
	std::size_t dim = 16;
	std::string format = "csv";

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--trials" && i + 1 < argc) {
			trials = std::stoul(argv[++i]);
		} else if (arg == "--modulus" && i + 1 < argc) {
			P = std::stoll(argv[++i]);
		} else if (arg == "--dim" && i + 1 < argc) {
			dim = std::stoul(argv[++i]);
		} else if (arg == "--csv") {
			format = "csv";
		} else if (arg == "--text") {
			format = "text";
		} else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: bench_attack_validation_fhe [--trials <N>] [--modulus <P>] [--dim <d>] "
			             "[--csv|--text]\n";
			return 0;
		}
	}

	uint32_t N = 16384;
	std::size_t canvas_M = 32;

	if (format != "csv") {
		std::cout << "Initializing OpenFHE BGV CryptoContext (P=" << P << ", N=" << N << ", Canvas=" << canvas_M
		          << "x" << canvas_M << ")...\n";
	}

	KeyPair<DCRTPoly> keys;
	CryptoContext<DCRTPoly> cc = SetupAttackFHEContext(P, N, keys, canvas_M, dim);

	std::vector<AttackFHEResult> results;

	if (format != "csv") {
		std::cout << "--> Running Attack 1: Ciphertext-Only Column Transformation...\n";
	}
	auto res1 = RunFHEColumnTransformationAttack(cc, keys, trials, P, dim, canvas_M, N);
	results.push_back(res1.first);
	results.push_back(res1.second);

	if (format != "csv") {
		std::cout << "--> Running Attack 2: Ciphertext-Only Bilinear Cancellation...\n";
	}
	auto res2 = RunFHEBilinearCancellationAttack(cc, keys, trials, P, dim, canvas_M, N);
	results.push_back(res2.first);
	results.push_back(res2.second);

	if (format != "csv") {
		std::cout << "--> Running Attack 3: Ciphertext-Only Two-Key Cancellation...\n";
	}
	auto res3 = RunFHETwoKeyCancellationAttack(cc, keys, trials, P, dim, canvas_M, N);
	results.push_back(res3.first);
	results.push_back(res3.second);

	if (format == "csv") {
		PrintCsvHeader();
		for (const auto &r : results) {
			PrintCsvResult(r);
		}
	} else {
		std::cout << "\n======================================================================================="
		             "==========================================\n";
		std::cout << "OPENFHE HOMOMORPHIC ATTACK-VALIDATION EXPERIMENTAL SUITE (128-BIT SECURITY)\n";
		std::cout << "Ciphertext-Only Attacks (Masks, Rotations, Ciphertext Mults); CATCH-II Verified on "
		             "IDENTICAL Corrupted Outputs\n";
		std::cout << "========================================================================================="
		             "========================================\n";
		std::cout << std::left << std::setw(25) << "Attack" << std::setw(28) << "Verifier" << std::setw(24)
		          << "Input family" << std::setw(8) << "Trials" << std::setw(12) << "Accepted" << std::setw(24)
		          << "Incorrect & Accepted" << std::setw(12) << "Server (s)" << std::setw(12) << "Attack (s)"
		          << "\n";
		std::cout << std::string(145, '-') << "\n";

		for (const auto &r : results) {
			std::string acc_str = std::to_string(r.acceptedCount) + " (" +
			                      std::to_string((int)(r.acceptanceRate * 100)) + "%)";
			std::string inc_str = std::to_string(r.incorrectAndAcceptedCount) + " (" +
			                      std::to_string((int)(r.forgerySuccessRate * 100)) + "%)";
			std::cout << std::left << std::setw(25) << r.attackName << std::setw(28) << r.verifierName
			          << std::setw(24) << r.inputFamily << std::setw(8) << r.totalTrials << std::setw(12)
			          << acc_str << std::setw(24) << inc_str << std::setw(12) << std::fixed
			          << std::setprecision(4) << r.avgServerTime_s << std::setw(12) << r.avgAttackTime_s
			          << "\n";
		}
		std::cout << "========================================================================================="
		             "========================================\n\n";
	}

	return 0;
}
