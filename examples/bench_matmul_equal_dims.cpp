#include "fhe_context.hpp"
#include "fhe_mat_mul.hpp"
#include "linear_algebra.hpp"

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace lbcrypto;
using namespace std::chrono;

// ============================================================================
// DATA STRUCTURE FOR EQUAL-USEFUL-DIMENSION BENCHMARK
// ============================================================================
struct EqualDimResult {
	std::string scheme;
	std::size_t usefulDim;         // Strictly equal across all compared schemes (e.g. 16, 32, 64)
	std::size_t paddedDim;         // The actual FHE ring matrix size required (e.g. 16 vs 32 vs 64)
	std::size_t ell;               // Number of keys / challenges
	std::string soundnessSecurity; // Soundness description

	// Timings (seconds)
	double clientSetupTime_s;
	double serverComputeTime_s;
	double clientVerifyTime_s;
	double totalClientTime_s;
	double totalServerTime_s;
	double totalWallTime_s;

	// Communication (bytes)
	std::size_t clientToServerBytes;
	std::size_t serverToClientBytes;
	std::size_t totalBytes;
	std::size_t clientToServerCts;
	std::size_t serverToClientCts;

	bool verificationPassed;
};

std::size_t NextPowerOfTwo(std::size_t n)
{
	std::size_t p = 1;
	while (p < n) {
		p <<= 1;
	}
	return p;
}

std::size_t MeasureCiphertextSize(const Ciphertext<DCRTPoly> &ct)
{
	std::stringstream ss;
	Serial::Serialize(ct, ss, SerType::BINARY);
	return ss.str().size();
}

LinearAlgebra::Matrix DecryptMatrix(CryptoContext<DCRTPoly> &cc, KeyPair<DCRTPoly> &keys, Ciphertext<DCRTPoly> &m,
                                    std::size_t n)
{
	Plaintext p;
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

CryptoContext<DCRTPoly> InitCryptoContext(std::int64_t blockSize, fhe::FHEParameters &params)
{
	params = fhe::SetFHEParameters(blockSize);
	if (params.N < 16384) {
		params.N = 16384;
		params.slot_count = params.N / 2;
		std::int64_t k = 1;
		while (true) {
			params.P = k * 2 * params.N + 1;
			if (LinearAlgebra::IsPrime(params.P)) {
				break;
			}
			k++;
		}
	}

	CCParams<CryptoContextBGVRNS> parameters;
	parameters.SetMultiplicativeDepth(3);
	parameters.SetPlaintextModulus(params.P);
	parameters.SetRingDim(params.N);
	parameters.SetSecurityLevel(HEStd_128_classic);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);
	cc->Enable(FHE);

	return cc;
}

// ============================================================================
// 1. DATASEAL SINGLE-KEY (Equal useful dimension: n x n)
// Adds 2 check rows => (n + 2) rows => Forces padding to next power of 2!
// ============================================================================
EqualDimResult BenchmarkDataSealSingleKey(std::size_t usefulDim, bool verbose = false)
{
	EqualDimResult res;
	res.scheme = "DataSeal-SingleKey";
	res.usefulDim = usefulDim;
	res.ell = 1;
	res.soundnessSecurity = "0-bit D, log2(t)-bit add";

	// Adding 2 rows to an already power-of-two matrix forces the grid to double!
	std::size_t taggedRows = usefulDim + 2;
	std::size_t M_padded = NextPowerOfTwo(taggedRows);
	res.paddedDim = M_padded;

	if (M_padded > 128) {
		res.soundnessSecurity = "OOM: Exceeds Ring Limit (N=131072 required)";
		res.clientSetupTime_s = 0.0;
		res.serverComputeTime_s = 0.0;
		res.clientVerifyTime_s = 0.0;
		res.totalClientTime_s = 0.0;
		res.totalServerTime_s = 0.0;
		res.totalWallTime_s = 0.0;
		res.clientToServerBytes = 0;
		res.serverToClientBytes = 0;
		res.totalBytes = 0;
		res.clientToServerCts = 0;
		res.serverToClientCts = 0;
		res.verificationPassed = false;
		return res;
	}

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(M_padded, params);

	auto t_wall_start = high_resolution_clock::now();

	// Client Setup
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(params.d));

	auto A = LinearAlgebra::GenerateSquareMatrixWithRandomValues(usefulDim);
	auto B = LinearAlgebra::GenerateSquareMatrixWithRandomValues(usefulDim);

	auto s_key = GenerateUniformFieldVector(usefulDim, params.P);
	auto hashOne = LinearAlgebra::MultiplyVectorAndMatrix(s_key, A);
	auto hashTwo = LinearAlgebra::MultiplyScalarAndVector(10.0, hashOne);
	auto goldenOutput = LinearAlgebra::MultiplyVectorAndMatrix(hashTwo, B);

	auto lastA = LinearAlgebra::AppendVectorToMatrixAsRow(hashOne, A);
	lastA = LinearAlgebra::AppendVectorToMatrixAsRow(hashTwo, lastA);

	auto paddedA = LinearAlgebra::PadMatrix(lastA, M_padded);
	auto paddedB = LinearAlgebra::PadMatrix(B, M_padded);

	for (auto &row : paddedA) {
		for (auto &v : row) {
			v %= params.P;
			if (v < 0) {
				v += params.P;
			}
		}
	}
	for (auto &row : paddedB) {
		for (auto &v : row) {
			v %= params.P;
			if (v < 0) {
				v += params.P;
			}
		}
	}

	auto ctA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(paddedA)));
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(paddedB)));
	auto t_setup_stop = high_resolution_clock::now();
	res.clientSetupTime_s = duration<double>(t_setup_stop - t_setup_start).count();

	std::size_t sizeA = MeasureCiphertextSize(ctA);
	std::size_t sizeB = MeasureCiphertextSize(ctB);
	res.clientToServerBytes = sizeA + sizeB;
	res.clientToServerCts = 2;

	// Server Computation
	auto t_server_start = high_resolution_clock::now();
	auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, params.d, params.N);
	auto t_server_stop = high_resolution_clock::now();
	res.serverComputeTime_s = duration<double>(t_server_stop - t_server_start).count();
	res.totalServerTime_s = res.serverComputeTime_s;

	std::size_t sizeC = MeasureCiphertextSize(ctC);
	res.serverToClientBytes = sizeC;
	res.serverToClientCts = 1;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;

	// Client Verification
	auto t_verify_start = high_resolution_clock::now();
	auto decC = DecryptMatrix(cc, keys, ctC, M_padded);

	auto modP = [P = params.P](std::int64_t v) {
		std::int64_t r = v % P;
		if (r < 0) {
			r += P;
		}
		if (r > P / 2) {
			r -= P;
		}
		return r;
	};

	bool pass = true;
	for (std::size_t j = 0; j < usefulDim; ++j) {
		const auto h_ret = modP(decC[usefulDim][j]);
		const auto g_ret = modP(decC[usefulDim + 1][j]);

		// Check the returned golden row against the client's reference.
		if (g_ret != modP(goldenOutput[j])) {
			pass = false;
			break;
		}

		// Check k^T C' = h'.
		std::int64_t expected_h = 0;
		for (std::size_t i = 0; i < usefulDim; ++i) {
			expected_h = modP(expected_h + modP(s_key[i]) * modP(decC[i][j]));
		}
		if (expected_h != h_ret) {
			pass = false;
			break;
		}

		// Combined-verifier link check: g' = alpha h'.
		// This benchmark constructs the golden row using alpha = 10.
		if (g_ret != modP(10LL * h_ret)) {
			pass = false;
			break;
		}
	}
	auto t_verify_stop = high_resolution_clock::now();
	res.clientVerifyTime_s = duration<double>(t_verify_stop - t_verify_start).count();
	res.totalClientTime_s = res.clientSetupTime_s + res.clientVerifyTime_s;

	auto t_wall_stop = high_resolution_clock::now();
	res.totalWallTime_s = duration<double>(t_wall_stop - t_wall_start).count();
	res.verificationPassed = pass;

	if (verbose) {
		std::cout << "[DataSeal-SingleKey] Useful: " << usefulDim << "x" << usefulDim
		          << " (Padded: " << M_padded << "x" << M_padded << ") Server: " << res.serverComputeTime_s
		          << "s\n";
	}
	return res;
}

// ============================================================================
// 2. DATASEAL MULTI-KEY (Equal useful dimension: n x n)
// Adds 2*lambda check rows => (n + 2*lambda) rows => Forces padding!
// ============================================================================
EqualDimResult BenchmarkDataSealMultiKey(std::size_t usefulDim, std::size_t lambda, bool verbose = false)
{
	EqualDimResult res;
	res.scheme = "DataSeal-MultiKey";
	res.usefulDim = usefulDim;
	res.ell = lambda;
	res.soundnessSecurity = "0-bit D, 128-bit add";

	std::size_t taggedRows = usefulDim + 2 * lambda;
	std::size_t M_padded = NextPowerOfTwo(taggedRows);
	res.paddedDim = M_padded;

	if (M_padded > 128) {
		res.soundnessSecurity = "OOM: Exceeds Ring Limit (N=131072 required)";
		res.clientSetupTime_s = 0.0;
		res.serverComputeTime_s = 0.0;
		res.clientVerifyTime_s = 0.0;
		res.totalClientTime_s = 0.0;
		res.totalServerTime_s = 0.0;
		res.totalWallTime_s = 0.0;
		res.clientToServerBytes = 0;
		res.serverToClientBytes = 0;
		res.totalBytes = 0;
		res.clientToServerCts = 0;
		res.serverToClientCts = 0;
		res.verificationPassed = false;
		return res;
	}

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(M_padded, params);

	auto t_wall_start = high_resolution_clock::now();

	// Client Setup
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(params.d));

	auto A = LinearAlgebra::GenerateSquareMatrixWithRandomValues(usefulDim);
	auto B = LinearAlgebra::GenerateSquareMatrixWithRandomValues(usefulDim);

	std::vector<LinearAlgebra::Vector> s_keys(lambda);
	std::vector<LinearAlgebra::Vector> hashOnes(lambda);
	std::vector<LinearAlgebra::Vector> hashTwos(lambda);
	std::vector<LinearAlgebra::Vector> goldenOutputs(lambda);

	auto lastA = A;
	for (std::size_t i = 0; i < lambda; ++i) {
		s_keys[i] = GenerateUniformFieldVector(usefulDim, params.P);
		hashOnes[i] = LinearAlgebra::MultiplyVectorAndMatrix(s_keys[i], A);
		hashTwos[i] = LinearAlgebra::MultiplyScalarAndVector(10.0 + i, hashOnes[i]);
		goldenOutputs[i] = LinearAlgebra::MultiplyVectorAndMatrix(hashTwos[i], B);

		lastA = LinearAlgebra::AppendVectorToMatrixAsRow(hashOnes[i], lastA);
		lastA = LinearAlgebra::AppendVectorToMatrixAsRow(hashTwos[i], lastA);
	}

	std::size_t ell = lastA.size();
	bool canRunRect = (M_padded % ell == 0) && (ell < M_padded);

	Ciphertext<DCRTPoly> ctA;
	if (canRunRect) {
		auto paddedA_rect = LinearAlgebra::PadMatrix(lastA, ell, M_padded);
		for (auto &row : paddedA_rect) {
			for (auto &v : row) {
				v %= params.P;
				if (v < 0) {
					v += params.P;
				}
			}
		}
		auto tiledA = LinearAlgebra::TileMatrixVertically(paddedA_rect, M_padded);
		ctA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(tiledA)));
	} else {
		auto paddedA = LinearAlgebra::PadMatrix(lastA, M_padded);
		for (auto &row : paddedA) {
			for (auto &v : row) {
				v %= params.P;
				if (v < 0) {
					v += params.P;
				}
			}
		}
		ctA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(paddedA)));
	}

	auto paddedB = LinearAlgebra::PadMatrix(B, M_padded);
	for (auto &row : paddedB) {
		for (auto &v : row) {
			v %= params.P;
			if (v < 0) {
				v += params.P;
			}
		}
	}
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(paddedB)));
	auto t_setup_stop = high_resolution_clock::now();
	res.clientSetupTime_s = duration<double>(t_setup_stop - t_setup_start).count();

	std::size_t sizeA = MeasureCiphertextSize(ctA);
	std::size_t sizeB = MeasureCiphertextSize(ctB);
	res.clientToServerBytes = sizeA + sizeB;
	res.clientToServerCts = 2;

	// Server Computation
	auto t_server_start = high_resolution_clock::now();
	Ciphertext<DCRTPoly> ctC;
	if (canRunRect) {
		ctC = fhe::RectangularJiangMatrixMultiplication(cc, ctA, ctB, params.d, ell, params.N);
	} else {
		ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, params.d, params.N);
	}
	auto t_server_stop = high_resolution_clock::now();
	res.serverComputeTime_s = duration<double>(t_server_stop - t_server_start).count();
	res.totalServerTime_s = res.serverComputeTime_s;

	std::size_t sizeC = MeasureCiphertextSize(ctC);
	res.serverToClientBytes = sizeC;
	res.serverToClientCts = 1;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;

	// Client Verification
	auto t_verify_start = high_resolution_clock::now();
	auto decC = DecryptMatrix(cc, keys, ctC, M_padded);

	auto modP = [P = params.P](std::int64_t v) {
		std::int64_t r = v % P;
		if (r < 0) {
			r += P;
		}
		if (r > P / 2) {
			r -= P;
		}
		return r;
	};

	bool pass = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		const std::size_t h_row = usefulDim + 2 * k;
		const std::size_t g_row = h_row + 1;

		// Matches the scalar used when constructing this golden row.
		const std::int64_t alpha = 10 + static_cast<std::int64_t>(k);

		for (std::size_t j = 0; j < usefulDim; ++j) {
			const auto h_ret = modP(decC[h_row][j]);
			const auto g_ret = modP(decC[g_row][j]);

			// Check the returned golden row against the client's reference.
			if (g_ret != modP(goldenOutputs[k][j])) {
				pass = false;
				break;
			}

			// Check k_k^T C' = h'_k.
			std::int64_t expected_h = 0;
			for (std::size_t i = 0; i < usefulDim; ++i) {
				expected_h = modP(expected_h + modP(s_keys[k][i]) * modP(decC[i][j]));
			}
			if (expected_h != h_ret) {
				pass = false;
				break;
			}

			// Combined-verifier link check: g'_k = alpha_k h'_k.
			if (g_ret != modP(alpha * h_ret)) {
				pass = false;
				break;
			}
		}

		if (!pass) {
			break;
		}
	}

	auto t_verify_stop = high_resolution_clock::now();
	res.clientVerifyTime_s = duration<double>(t_verify_stop - t_verify_start).count();
	res.totalClientTime_s = res.clientSetupTime_s + res.clientVerifyTime_s;

	auto t_wall_stop = high_resolution_clock::now();
	res.totalWallTime_s = duration<double>(t_wall_stop - t_wall_start).count();
	res.verificationPassed = pass;

	if (verbose) {
		std::cout << "[DataSeal-MultiKey] Useful: " << usefulDim << "x" << usefulDim << " (Padded: " << M_padded
		          << "x" << M_padded << ") Server: " << res.serverComputeTime_s << "s\n";
	}
	return res;
}

// ============================================================================
// 3. OURS: SEQUENTIAL SETTING II (Equal useful dimension: n x n)
// Exact n x n grid. No padding blowup.
// ============================================================================
EqualDimResult BenchmarkCATCHSequential(std::size_t usefulDim, std::size_t lambda, bool verbose = false)
{
	EqualDimResult res;
	res.scheme = "CATCH-SettingII-Sequential";
	res.usefulDim = usefulDim;
	res.paddedDim = usefulDim; // Exact fit!
	res.ell = lambda;
	res.soundnessSecurity = "128-bit consistency soundness";

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(usefulDim, params);

	auto t_wall_start = high_resolution_clock::now();

	// Client Setup
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(params.d));

	std::size_t M = usefulDim;
	auto A = LinearAlgebra::GenerateSquareMatrixWithRandomValues(M);
	auto B = LinearAlgebra::GenerateSquareMatrixWithRandomValues(M);

	for (auto &row : A) {
		for (auto &v : row) {
			v %= params.P;
			if (v < 0) {
				v += params.P;
			}
		}
	}
	for (auto &row : B) {
		for (auto &v : row) {
			v %= params.P;
			if (v < 0) {
				v += params.P;
			}
		}
	}

	auto ctA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(A)));
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B)));
	auto t_setup_stop = high_resolution_clock::now();
	res.clientSetupTime_s = duration<double>(t_setup_stop - t_setup_start).count();

	std::size_t ct_size = MeasureCiphertextSize(ctA);
	res.clientToServerBytes = 2 * ct_size;
	res.clientToServerCts = 2;

	// Round 1 Server MatMul
	auto t_s1_start = high_resolution_clock::now();
	auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, params.d, params.N);
	auto t_s1_stop = high_resolution_clock::now();
	double s1_time = duration<double>(t_s1_stop - t_s1_start).count();
	res.serverToClientBytes = ct_size;
	res.serverToClientCts = 1;

	// Sequential Verification Rounds
	auto t_v_start = high_resolution_clock::now();
	auto C_mat = DecryptMatrix(cc, keys, ctC, M);
	auto t_v_mid = high_resolution_clock::now();
	double decrypt_c_time = duration<double>(t_v_mid - t_v_start).count();

	auto modP = [P = params.P](std::int64_t v) {
		std::int64_t r = v % P;
		if (r < 0) {
			r += P;
		}
		if (r > P / 2) {
			r -= P;
		}
		return r;
	};

	double total_server_v_time = 0.0;
	double total_client_v_time = decrypt_c_time;
	bool allPass = true;

	for (std::size_t k = 0; k < lambda; ++k) {
		auto t_ck_prep_start = high_resolution_clock::now();
		auto r = GenerateUniformFieldVector(M, params.P);
		auto vA = LinearAlgebra::MultiplyVectorAndMatrix(r, A);
		for (auto &x : vA) {
			x %= params.P;
			if (x < 0) {
				x += params.P;
			}
		}

		LinearAlgebra::Matrix P_mat(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t i = 0; i < M; ++i) {
			for (std::size_t j = 0; j < M; ++j) {
				P_mat[i][j] = vA[j];
			}
		}
		auto ctP_k =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_mat)));
		auto t_ck_prep_stop = high_resolution_clock::now();
		total_client_v_time += duration<double>(t_ck_prep_stop - t_ck_prep_start).count();

		res.clientToServerBytes += ct_size;
		res.clientToServerCts += 1;

		auto t_sk_start = high_resolution_clock::now();
		auto ctV_k = fhe::OptimalJiangMatrixMultiplication(cc, ctP_k, ctB, params.d, params.N);
		auto t_sk_stop = high_resolution_clock::now();
		total_server_v_time += duration<double>(t_sk_stop - t_sk_start).count();

		res.serverToClientBytes += ct_size;
		res.serverToClientCts += 1;

		auto t_ck_check_start = high_resolution_clock::now();
		auto V_mat = DecryptMatrix(cc, keys, ctV_k, M);
		auto rC = LinearAlgebra::MultiplyVectorAndMatrix(r, C_mat);
		for (std::size_t j = 0; j < M; ++j) {
			if (modP(V_mat[0][j]) != modP(rC[j])) {
				allPass = false;
				break;
			}
		}
		auto t_ck_check_stop = high_resolution_clock::now();
		total_client_v_time += duration<double>(t_ck_check_stop - t_ck_check_start).count();
	}

	res.serverComputeTime_s = s1_time + total_server_v_time;
	res.totalServerTime_s = res.serverComputeTime_s;
	res.clientVerifyTime_s = total_client_v_time;
	res.totalClientTime_s = res.clientSetupTime_s + res.clientVerifyTime_s;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;

	auto t_wall_stop = high_resolution_clock::now();
	res.totalWallTime_s = duration<double>(t_wall_stop - t_wall_start).count();
	res.verificationPassed = allPass;

	if (verbose) {
		std::cout << "[CATCH-Sequential] Useful: " << usefulDim << "x" << usefulDim << " (Padded: " << M << "x"
		          << M << ") Server: " << res.serverComputeTime_s << "s\n";
	}
	return res;
}

// ============================================================================
// CATCH: BATCHED SETTING II WITH RECTANGULAR MATMUL (Equal useful dimension: n x n)
// Exact n x n grid. Exactly 2 trips. Only lambda homomorphic mults in verification!
// ============================================================================
EqualDimResult BenchmarkCATCHBatchedRectangular(std::size_t usefulDim, std::size_t lambda, bool verbose = false)
{
	EqualDimResult res;
	res.scheme = "CATCH-SettingII-Batched";
	res.usefulDim = usefulDim;
	res.paddedDim = usefulDim; // Exact fit!
	res.ell = lambda;
	res.soundnessSecurity = "128-bit consistency soundness (Batched 2-Round, Rectangular MatMul)";

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(usefulDim, params);

	auto t_wall_start = high_resolution_clock::now();

	// Client Setup
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(params.d));

	std::size_t M = usefulDim;
	auto A = LinearAlgebra::GenerateSquareMatrixWithRandomValues(M);
	auto B = LinearAlgebra::GenerateSquareMatrixWithRandomValues(M);

	for (auto &row : A) {
		for (auto &v : row) {
			v %= params.P;
			if (v < 0) {
				v += params.P;
			}
		}
	}
	for (auto &row : B) {
		for (auto &v : row) {
			v %= params.P;
			if (v < 0) {
				v += params.P;
			}
		}
	}

	auto ctA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(A)));
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B)));
	auto t_setup_stop = high_resolution_clock::now();
	res.clientSetupTime_s = duration<double>(t_setup_stop - t_setup_start).count();

	std::size_t ct_size = MeasureCiphertextSize(ctA);
	res.clientToServerBytes = 2 * ct_size;
	res.clientToServerCts = 2;

	// Round 1 Server MatMul: C = A * B
	auto t_s1_start = high_resolution_clock::now();
	auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, params.d, params.N);
	auto t_s1_stop = high_resolution_clock::now();
	double s1_time = duration<double>(t_s1_stop - t_s1_start).count();
	res.serverToClientBytes = ct_size;
	res.serverToClientCts = 1;

	// Round 2 Client: Draw all lambda challenges, pack into P_rect (lambda x M)
	auto t_vprep_start = high_resolution_clock::now();
	auto C_mat = DecryptMatrix(cc, keys, ctC, M);

	std::vector<LinearAlgebra::Vector> challenges(lambda);
	LinearAlgebra::Matrix P_rect(lambda, LinearAlgebra::Vector(M, 0));

	for (std::size_t k = 0; k < lambda; ++k) {
		challenges[k] = GenerateUniformFieldVector(M, params.P);
		auto vA = LinearAlgebra::MultiplyVectorAndMatrix(challenges[k], A);
		for (std::size_t j = 0; j < M; ++j) {
			int64_t val = vA[j] % params.P;
			if (val < 0) {
				val += params.P;
			}
			P_rect[k][j] = val;
		}
	}

	// Tile P vertically (M / lambda) times to form full M x M matrix
	auto P_tiled = LinearAlgebra::TileMatrixVertically(P_rect, M);
	auto ctP = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_tiled)));
	auto t_vprep_stop = high_resolution_clock::now();
	double vprep_time = duration<double>(t_vprep_stop - t_vprep_start).count();

	res.clientToServerBytes += ct_size;
	res.clientToServerCts += 1;

	// Round 2 Server MatMul: V = P * B (Rectangular MatMul: ONLY lambda mults!)
	auto t_s2_start = high_resolution_clock::now();
	auto ctV = fhe::RectangularJiangMatrixMultiplication(cc, ctP, ctB, params.d, lambda, params.N);
	auto t_s2_stop = high_resolution_clock::now();
	double s2_time = duration<double>(t_s2_stop - t_s2_start).count();

	res.serverToClientBytes += ct_size;
	res.serverToClientCts += 1;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;

	// Round 2 Client: Decrypt and check all lambda challenges
	auto t_vcheck_start = high_resolution_clock::now();
	auto V_mat = DecryptMatrix(cc, keys, ctV, M);

	auto modP = [P = params.P](std::int64_t v) {
		std::int64_t r = v % P;
		if (r < 0) {
			r += P;
		}
		if (r > P / 2) {
			r -= P;
		}
		return r;
	};

	bool allPass = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		auto rC = LinearAlgebra::MultiplyVectorAndMatrix(challenges[k], C_mat);
		for (std::size_t j = 0; j < M; ++j) {
			if (modP(V_mat[k][j]) != modP(rC[j])) {
				allPass = false;
				break;
			}
		}
		if (!allPass) {
			break;
		}
	}
	auto t_vcheck_stop = high_resolution_clock::now();
	double vcheck_time = duration<double>(t_vcheck_stop - t_vcheck_start).count();

	res.serverComputeTime_s = s1_time + s2_time;
	res.totalServerTime_s = res.serverComputeTime_s;
	res.clientVerifyTime_s = vprep_time + vcheck_time;
	res.totalClientTime_s = res.clientSetupTime_s + res.clientVerifyTime_s;

	auto t_wall_stop = high_resolution_clock::now();
	res.totalWallTime_s = duration<double>(t_wall_stop - t_wall_start).count();
	res.verificationPassed = allPass;

	if (verbose) {
		std::cout << "[CATCH-Batched-Rect] Useful: " << usefulDim << "x" << usefulDim << " (Padded: " << M
		          << "x" << M << ") Server: " << res.serverComputeTime_s << "s (Round 2 Verify: " << s2_time
		          << "s)\n";
	}
	return res;
}

void PrintCsvHeader()
{
	std::cout << "scheme,usefulDim,paddedDim,lambda,soundnessSecurity,"
	          << "clientSetupTime_s,serverComputeTime_s,clientVerifyTime_s,"
	          << "totalClientTime_s,totalServerTime_s,totalWallTime_s,"
	          << "clientToServerBytes,serverToClientBytes,totalBytes,"
	          << "clientToServerCts,serverToClientCts,verificationPassed\n";
}

void PrintCsvResult(const EqualDimResult &r)
{
	std::cout << r.scheme << "," << r.usefulDim << "," << r.paddedDim << "," << r.ell << ",\""
	          << r.soundnessSecurity << "\"," << r.clientSetupTime_s << "," << r.serverComputeTime_s << ","
	          << r.clientVerifyTime_s << "," << r.totalClientTime_s << "," << r.totalServerTime_s << ","
	          << r.totalWallTime_s << "," << r.clientToServerBytes << "," << r.serverToClientBytes << ","
	          << r.totalBytes << "," << r.clientToServerCts << "," << r.serverToClientCts << ","
	          << (r.verificationPassed ? "true" : "false") << "\n";
}

int main(int argc, char **argv)
{
	std::size_t usefulDim = 16;
	std::size_t lambda = 8;
	std::string format = "csv";
	std::string schemeFilter = "all";

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if ((arg == "--useful" || arg == "--size") && i + 1 < argc) {
			usefulDim = std::stoul(argv[++i]);
		} else if (arg == "--lambda" && i + 1 < argc) {
			lambda = std::stoul(argv[++i]);
		} else if (arg == "--csv") {
			format = "csv";
		} else if (arg == "--text") {
			format = "text";
		} else if (arg == "--scheme" && i + 1 < argc) {
			schemeFilter = argv[++i];
		} else if (arg == "--help" || arg == "-h") {
			std::cout
			    << "Usage: bench_matmul_equal_dims [options]\n"
			    << "Options:\n"
			    << "  --useful <N>     Exact useful matrix dimension N (16, 32, 64, 128) [default: 16]\n"
			    << "  --lambda <L>     Number of keys/challenges [default: 8]\n"
			    << "  --scheme <name>  Filter scheme (all, dataseal_single, dataseal_multi, seq, batched)\n"
			    << "  --csv            Output machine-readable CSV (default)\n"
			    << "  --text           Output human-readable formatted summary\n";
			return 0;
		}
	}

	std::vector<EqualDimResult> results;

	if (schemeFilter == "all" || schemeFilter == "dataseal_single") {
		std::cout << "Running DataSeal Single Key\n";
		results.push_back(BenchmarkDataSealSingleKey(usefulDim, format == "text"));
	}
	if (schemeFilter == "all" || schemeFilter == "dataseal_multi") {
		std::cout << "Running DataSeal Multi Key\n";
		results.push_back(BenchmarkDataSealMultiKey(usefulDim, lambda, format == "text"));
	}
	if (schemeFilter == "all" || schemeFilter == "seq" || schemeFilter == "setting2_seq") {
		std::cout << "Running Sequential CATCH\n";
		results.push_back(BenchmarkCATCHSequential(usefulDim, lambda, format == "text"));
	}
	if (schemeFilter == "all" || schemeFilter == "batched" || schemeFilter == "setting2_batched") {
		std::cout << "Running Batched CATCH\n";
		results.push_back(BenchmarkCATCHBatchedRectangular(usefulDim, lambda, format == "text"));
	}

	if (format == "csv") {
		PrintCsvHeader();
		for (const auto &r : results) {
			PrintCsvResult(r);
		}
	} else {
		std::cout
		    << "\n=========================================================================================\n";
		std::cout << "STRICT EQUAL-USEFUL-DIMENSION BENCHMARK (Useful Dim: " << usefulDim << "x" << usefulDim
		          << ", lambda=" << lambda << ")\n";
		std::cout
		    << "=========================================================================================\n";
		std::cout << std::left << std::setw(28) << "Scheme" << std::setw(14) << "Useful Dim" << std::setw(14)
		          << "Padded Dim" << std::setw(14) << "Server (s)" << std::setw(14) << "Client (s)"
		          << std::setw(14) << "Wall (s)" << std::setw(14) << "Comm (KB)" << std::setw(8) << "Pass?"
		          << "\n";
		std::cout << std::string(118, '-') << "\n";
		for (const auto &r : results) {
			std::cout << std::left << std::setw(28) << r.scheme << std::setw(14)
			          << (std::to_string(r.usefulDim) + "x" + std::to_string(r.usefulDim)) << std::setw(14)
			          << (std::to_string(r.paddedDim) + "x" + std::to_string(r.paddedDim)) << std::setw(14)
			          << std::fixed << std::setprecision(4) << r.serverComputeTime_s << std::setw(14)
			          << std::fixed << std::setprecision(4) << r.totalClientTime_s << std::setw(14)
			          << std::fixed << std::setprecision(4) << r.totalWallTime_s << std::setw(14)
			          << std::fixed << std::setprecision(1) << (r.totalBytes / 1024.0) << std::setw(8)
			          << (r.verificationPassed ? "YES" : "NO") << "\n";
		}
		std::cout
		    << "=========================================================================================\n\n";
	}

	return 0;
}
