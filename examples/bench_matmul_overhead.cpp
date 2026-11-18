#include "fhe_context.hpp"
#include "fhe_mat_mul.hpp"
#include "linear_algebra.hpp"

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"

#include <chrono>
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
// DATA STRUCTURE FOR BENCHMARK MEASUREMENTS
// ============================================================================
struct BenchmarkResult {
	std::string scheme;
	std::size_t matrixSize;        // Padded dimension M
	std::size_t usefulDim;         // Useful dimension d
	std::size_t lambda;            // Number of keys (DataSeal) or challenges (Setting II)
	std::string securitySoundness; // Soundness description

	// Timings (in seconds)
	double clientSetupTime;   // KeyGen, encoding, encryption
	double serverComputeTime; // Matrix multiplication(s) on server
	double clientVerifyTime;  // Decryption, checks
	double totalClientTime;   // Setup + Verify
	double totalServerTime;   // Server compute
	double totalWallTime;     // End-to-end elapsed time

	// Communication (in bytes)
	std::size_t clientToServerBytes;
	std::size_t serverToClientBytes;
	std::size_t totalBytes;

	// Number of ciphertexts exchanged
	std::size_t clientToServerCts;
	std::size_t serverToClientCts;

	bool verificationPassed;
};

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
// SCHEME 1: Single-Key DataSeal (m = 1)
// ============================================================================
BenchmarkResult BenchmarkSingleKeyDataSeal(std::int64_t base, bool verbose = false)
{
	BenchmarkResult res;
	res.scheme = "DataSeal-SingleKey";
	res.matrixSize = base;
	res.lambda = 1;
	res.securitySoundness = "0-bit D, log2(t)-bit add";

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(base, params);

	auto t_wall_start = high_resolution_clock::now();

	// 1. Client Setup
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(params.d));

	std::size_t M = base;
	std::size_t usefulDim = (M > 2) ? (M - 2) : 2;
	res.usefulDim = usefulDim;

	auto A = LinearAlgebra::GenerateSquareMatrixWithRandomValues(usefulDim);
	auto B = LinearAlgebra::GenerateSquareMatrixWithRandomValues(usefulDim);
	auto s = GenerateUniformFieldVector(usefulDim, params.P);
	constexpr double scalar = 10;

	auto hashOne = LinearAlgebra::MultiplyVectorAndMatrix(s, A);
	auto hashTwo = LinearAlgebra::MultiplyScalarAndVector(scalar, hashOne);
	auto goldenOutput = LinearAlgebra::MultiplyVectorAndMatrix(hashTwo, B);

	auto tempA = LinearAlgebra::AppendVectorToMatrixAsRow(hashOne, A);
	auto lastA = LinearAlgebra::AppendVectorToMatrixAsRow(hashTwo, tempA);

	auto paddedA = LinearAlgebra::PadMatrix(lastA, M);
	auto paddedB = LinearAlgebra::PadMatrix(B, M);

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

	auto flatA = LinearAlgebra::FlattenRowMajor(paddedA);
	auto ciphertextA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatA));

	auto flatB = LinearAlgebra::FlattenRowMajor(paddedB);
	auto ciphertextB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatB));
	auto t_setup_stop = high_resolution_clock::now();
	res.clientSetupTime = duration<double>(t_setup_stop - t_setup_start).count();

	// Communication: Client -> Server
	std::size_t sizeA = MeasureCiphertextSize(ciphertextA);
	std::size_t sizeB = MeasureCiphertextSize(ciphertextB);
	res.clientToServerBytes = sizeA + sizeB;
	res.clientToServerCts = 2;

	// 2. Server Computation
	auto t_server_start = high_resolution_clock::now();
	auto ciphertextC = fhe::OptimalJiangMatrixMultiplication(cc, ciphertextA, ciphertextB, params.d, params.N);
	auto t_server_stop = high_resolution_clock::now();
	res.serverComputeTime = duration<double>(t_server_stop - t_server_start).count();
	res.totalServerTime = res.serverComputeTime;

	// Communication: Server -> Client
	std::size_t sizeC = MeasureCiphertextSize(ciphertextC);
	res.serverToClientBytes = sizeC;
	res.serverToClientCts = 1;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;

	// 3. Client Verification
	auto t_verify_start = high_resolution_clock::now();
	auto fheResult = DecryptMatrix(cc, keys, ciphertextC, M);

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

	auto fheMatrixC = fheResult;
	fheMatrixC.resize(usefulDim);

	auto fheExtractedHashOne = fheResult[usefulDim];
	auto fheExtractedHashTwo = fheResult[usefulDim + 1];

	auto fheComputedHashOne = LinearAlgebra::MultiplyVectorAndMatrix(s, fheMatrixC);

	bool check1 = true;
	for (std::size_t j = 0; j < usefulDim; ++j) {
		if (modP(fheComputedHashOne[j]) != modP(fheExtractedHashOne[j])) {
			check1 = false;
			break;
		}
	}

	bool check2 = true;
	for (std::size_t j = 0; j < usefulDim; ++j) {
		if (modP(goldenOutput[j]) != modP(fheExtractedHashTwo[j])) {
			check2 = false;
			break;
		}
	}
	auto t_verify_stop = high_resolution_clock::now();
	res.clientVerifyTime = duration<double>(t_verify_stop - t_verify_start).count();
	res.totalClientTime = res.clientSetupTime + res.clientVerifyTime;

	auto t_wall_stop = high_resolution_clock::now();
	res.totalWallTime = duration<double>(t_wall_stop - t_wall_start).count();
	res.verificationPassed = (check1 && check2);

	if (verbose) {
		std::cout << "[DataSeal-SingleKey " << base << "x" << base << "] Server: " << res.serverComputeTime
		          << "s | Client: " << res.totalClientTime << "s | Comm: " << (res.totalBytes / 1024.0)
		          << " KB\n";
	}

	return res;
}

// ============================================================================
// SCHEME 2: Multi-Key DataSeal (m = lambda keys)
// ============================================================================
BenchmarkResult BenchmarkMultiKeyDataSeal(std::int64_t base, std::size_t lambda, bool verbose = false,
                                          bool useRectangular = false)
{
	BenchmarkResult res;
	res.scheme = "DataSeal-MultiKey";
	res.lambda = lambda;
	res.securitySoundness = "0-bit D, 128-bit add";

	// Determine required padded matrix size to accommodate 2 * lambda rows
	std::size_t M_padded = base;
	std::size_t usefulDim = 0;
	if (base > 2 * lambda) {
		usefulDim = base - 2 * lambda;
	} else {
		// Must pad up to next power-of-two block that can hold usefulDim >= 8
		M_padded = (base <= 16) ? 32 : (base * 2);
		usefulDim = (M_padded > 2 * lambda) ? (M_padded - 2 * lambda) : 16;
	}
	res.matrixSize = M_padded;
	res.usefulDim = usefulDim;

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(M_padded, params);

	auto t_wall_start = high_resolution_clock::now();

	// 1. Client Setup
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
	bool canRunRect = useRectangular && (M_padded % ell == 0) && (ell < M_padded);

	Ciphertext<DCRTPoly> ciphertextA;
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
		ciphertextA =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(tiledA)));
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
		ciphertextA =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(paddedA)));
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

	auto flatB = LinearAlgebra::FlattenRowMajor(paddedB);
	auto ciphertextB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatB));
	auto t_setup_stop = high_resolution_clock::now();
	res.clientSetupTime = duration<double>(t_setup_stop - t_setup_start).count();

	// Communication: Client -> Server
	std::size_t sizeA = MeasureCiphertextSize(ciphertextA);
	std::size_t sizeB = MeasureCiphertextSize(ciphertextB);
	res.clientToServerBytes = sizeA + sizeB;
	res.clientToServerCts = 2;

	// 2. Server Computation
	auto t_server_start = high_resolution_clock::now();
	Ciphertext<DCRTPoly> ciphertextC;
	if (canRunRect) {
		ciphertextC =
		    fhe::RectangularJiangMatrixMultiplication(cc, ciphertextA, ciphertextB, params.d, ell, params.N);
	} else {
		ciphertextC = fhe::OptimalJiangMatrixMultiplication(cc, ciphertextA, ciphertextB, params.d, params.N);
	}
	auto t_server_stop = high_resolution_clock::now();
	res.serverComputeTime = duration<double>(t_server_stop - t_server_start).count();
	res.totalServerTime = res.serverComputeTime;

	// Communication: Server -> Client
	std::size_t sizeC = MeasureCiphertextSize(ciphertextC);
	res.serverToClientBytes = sizeC;
	res.serverToClientCts = 1;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;

	// 3. Client Verification
	auto t_verify_start = high_resolution_clock::now();
	auto fheResult = DecryptMatrix(cc, keys, ciphertextC, M_padded);

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

	auto fheMatrixC = fheResult;
	fheMatrixC.resize(usefulDim);

	bool allPass = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		auto fheExtractedHashOne = fheResult[usefulDim + 2 * k];
		auto fheExtractedHashTwo = fheResult[usefulDim + 2 * k + 1];

		auto fheComputedHashOne = LinearAlgebra::MultiplyVectorAndMatrix(s_keys[k], fheMatrixC);

		for (std::size_t j = 0; j < usefulDim; ++j) {
			if (modP(fheComputedHashOne[j]) != modP(fheExtractedHashOne[j])) {
				allPass = false;
				break;
			}
			if (modP(goldenOutputs[k][j]) != modP(fheExtractedHashTwo[j])) {
				allPass = false;
				break;
			}
		}
		if (!allPass) {
			break;
		}
	}
	auto t_verify_stop = high_resolution_clock::now();
	res.clientVerifyTime = duration<double>(t_verify_stop - t_verify_start).count();
	res.totalClientTime = res.clientSetupTime + res.clientVerifyTime;

	auto t_wall_stop = high_resolution_clock::now();
	res.totalWallTime = duration<double>(t_wall_stop - t_wall_start).count();
	res.verificationPassed = allPass;

	if (verbose) {
		std::cout << "[DataSeal-MultiKey (" << lambda << " keys) " << M_padded << "x" << M_padded
		          << "] Server: " << res.serverComputeTime << "s | Client: " << res.totalClientTime
		          << "s | Comm: " << (res.totalBytes / 1024.0) << " KB\n";
	}

	return res;
}

// ============================================================================
// SCHEME 3: Setting II Freivalds (Sequential, lambda challenges, 1+lambda rounds)
// ============================================================================
BenchmarkResult BenchmarkSetting2Freivalds(std::int64_t base, std::size_t lambda, bool verbose = false)
{
	BenchmarkResult res;
	res.scheme = "SettingII-Sequential";
	res.matrixSize = base;
	res.usefulDim = base; // Freivalds needs NO row sacrifice; full dimension utilized
	res.lambda = lambda;
	res.securitySoundness = "128-bit consistency soundness";

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(base, params);

	auto t_wall_start = high_resolution_clock::now();

	// 1. Client Setup (Round 1)
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(params.d));

	std::size_t M = base;
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

	auto flatA = LinearAlgebra::FlattenRowMajor(A);
	auto ciphertextA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatA));

	auto flatB = LinearAlgebra::FlattenRowMajor(B);
	auto ciphertextB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatB));
	auto t_setup_stop = high_resolution_clock::now();
	double round1_client_setup = duration<double>(t_setup_stop - t_setup_start).count();

	// Communication Round 1: Client -> Server
	std::size_t sizeA = MeasureCiphertextSize(ciphertextA);
	std::size_t sizeB = MeasureCiphertextSize(ciphertextB);
	res.clientToServerBytes = sizeA + sizeB;
	res.clientToServerCts = 2;

	// 2. Server Computation (Round 1: Initial MatMul C = A * B)
	auto t_server1_start = high_resolution_clock::now();
	auto ciphertextC = fhe::OptimalJiangMatrixMultiplication(cc, ciphertextA, ciphertextB, params.d, params.N);
	auto t_server1_stop = high_resolution_clock::now();
	double round1_server_time = duration<double>(t_server1_stop - t_server1_start).count();

	// Communication Round 1: Server -> Client
	std::size_t sizeC = MeasureCiphertextSize(ciphertextC);
	res.serverToClientBytes = sizeC;
	res.serverToClientCts = 1;

	// 3. Client Verification Preparation (Round 2: Draw challenges after receiving C)
	auto t_vsetup_start = high_resolution_clock::now();
	// Client decrypts C to commit/prepare comparison
	auto C_mat = DecryptMatrix(cc, keys, ciphertextC, M);

	std::vector<LinearAlgebra::Vector> challenges(lambda);
	std::vector<Ciphertext<DCRTPoly>> ciphertextsP(lambda);
	std::size_t sizeP_total = 0;

	for (std::size_t k = 0; k < lambda; ++k) {
		challenges[k] = GenerateUniformFieldVector(M, params.P);
		auto vA = LinearAlgebra::MultiplyVectorAndMatrix(challenges[k], A);
		for (auto &x : vA) {
			x %= params.P;
			if (x < 0) {
				x += params.P;
			}
		}

		// Replicate row vector vA across all rows of P_k
		LinearAlgebra::Matrix P_mat(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t i = 0; i < M; ++i) {
			for (std::size_t j = 0; j < M; ++j) {
				P_mat[i][j] = vA[j];
			}
		}
		auto flatP = LinearAlgebra::FlattenRowMajor(P_mat);
		ciphertextsP[k] = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatP));
		sizeP_total += MeasureCiphertextSize(ciphertextsP[k]);
	}
	auto t_vsetup_stop = high_resolution_clock::now();
	double round2_client_prep = duration<double>(t_vsetup_stop - t_vsetup_start).count();

	// Communication Round 2: Client -> Server (Challenge vectors)
	res.clientToServerBytes += sizeP_total;
	res.clientToServerCts += lambda;

	// 4. Server Verification Computation (Round 2: V_k = P_k * B for each challenge)
	auto t_server2_start = high_resolution_clock::now();
	std::vector<Ciphertext<DCRTPoly>> ciphertextsV(lambda);
	std::size_t sizeV_total = 0;

	for (std::size_t k = 0; k < lambda; ++k) {
		ciphertextsV[k] =
		    fhe::OptimalJiangMatrixMultiplication(cc, ciphertextsP[k], ciphertextB, params.d, params.N);
		sizeV_total += MeasureCiphertextSize(ciphertextsV[k]);
	}
	auto t_server2_stop = high_resolution_clock::now();
	double round2_server_time = duration<double>(t_server2_stop - t_server2_start).count();

	// Communication Round 2: Server -> Client (Verification outputs)
	res.serverToClientBytes += sizeV_total;
	res.serverToClientCts += lambda;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;

	// 5. Client Verification Checks
	auto t_vcheck_start = high_resolution_clock::now();
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

	bool allChecksPass = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		auto V_mat = DecryptMatrix(cc, keys, ciphertextsV[k], M);
		auto vAB = V_mat[0]; // Extracted row product (r^T * A) * B
		auto rC = LinearAlgebra::MultiplyVectorAndMatrix(challenges[k], C_mat);

		for (std::size_t j = 0; j < M; ++j) {
			if (modP(vAB[j]) != modP(rC[j])) {
				allChecksPass = false;
				break;
			}
		}
		if (!allChecksPass) {
			break;
		}
	}
	auto t_vcheck_stop = high_resolution_clock::now();
	double round2_client_check = duration<double>(t_vcheck_stop - t_vcheck_start).count();

	auto t_wall_stop = high_resolution_clock::now();

	res.clientSetupTime = round1_client_setup;
	res.clientVerifyTime = round2_client_prep + round2_client_check;
	res.totalClientTime = res.clientSetupTime + res.clientVerifyTime;

	res.serverComputeTime = round1_server_time + round2_server_time;
	res.totalServerTime = res.serverComputeTime;
	res.totalWallTime = duration<double>(t_wall_stop - t_wall_start).count();
	res.verificationPassed = allChecksPass;

	if (verbose) {
		std::cout << "[SettingII-Sequential (" << lambda << " challenges) " << base << "x" << base
		          << "] Server: " << res.serverComputeTime << "s | Client: " << res.totalClientTime
		          << "s | Comm: " << (res.totalBytes / 1024.0) << " KB\n";
	}

	return res;
}

// ============================================================================
// SCHEME 4: Setting II Batched Freivalds (Ours, lambda challenges packed, 2-Round)
// ============================================================================
BenchmarkResult BenchmarkBatchedSetting2Freivalds(std::int64_t base, std::size_t lambda, bool verbose = false,
                                                  bool useRectangular = false)
{
	BenchmarkResult res;
	res.scheme = "Ours-SettingII-Batched";
	res.matrixSize = base;
	res.usefulDim = base;
	res.lambda = lambda;
	res.securitySoundness = useRectangular ? "128-bit consistency soundness (Batched 2-Round, Rectangular MatMul)"
	                                       : "128-bit consistency soundness (Batched 2-Round)";

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(base, params);

	auto t_wall_start = high_resolution_clock::now();

	// Client Setup (Round 1)
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(params.d));

	std::size_t M = base;
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

	auto flatA = LinearAlgebra::FlattenRowMajor(A);
	auto ciphertextA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatA));

	auto flatB = LinearAlgebra::FlattenRowMajor(B);
	auto ciphertextB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatB));
	auto t_setup_stop = high_resolution_clock::now();
	double round1_client_setup = duration<double>(t_setup_stop - t_setup_start).count();

	// Communication Round 1: Client -> Server
	std::size_t sizeA = MeasureCiphertextSize(ciphertextA);
	std::size_t sizeB = MeasureCiphertextSize(ciphertextB);
	res.clientToServerBytes = sizeA + sizeB;
	res.clientToServerCts = 2;

	// Server Computation (Round 1: Initial MatMul C = A * B)
	auto t_server1_start = high_resolution_clock::now();
	auto ciphertextC = fhe::OptimalJiangMatrixMultiplication(cc, ciphertextA, ciphertextB, params.d, params.N);
	auto t_server1_stop = high_resolution_clock::now();
	double round1_server_time = duration<double>(t_server1_stop - t_server1_start).count();

	// Communication Round 1: Server -> Client
	std::size_t sizeC = MeasureCiphertextSize(ciphertextC);
	res.serverToClientBytes = sizeC;
	res.serverToClientCts = 1;

	// Client Verification Preparation (Round 2: Draw ALL lambda challenges at once and pack into P)
	auto t_vsetup_start = high_resolution_clock::now();
	auto C_mat = DecryptMatrix(cc, keys, ciphertextC, M);

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
			P_rect[k][j] = val; // Pack challenge k into row k
		}
	}

	bool canRunRect = useRectangular && (M % lambda == 0) && (lambda <= M);
	Ciphertext<DCRTPoly> ciphertextP;
	if (canRunRect) {
		auto P_tiled = LinearAlgebra::TileMatrixVertically(P_rect, M);
		ciphertextP =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_tiled)));
	} else {
		LinearAlgebra::Matrix P_mat(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t k = 0; k < lambda; ++k) {
			P_mat[k] = P_rect[k];
		}
		ciphertextP =
		    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_mat)));
	}
	auto t_vsetup_stop = high_resolution_clock::now();
	double round2_client_prep = duration<double>(t_vsetup_stop - t_vsetup_start).count();

	// Communication Round 2: Client -> Server (Only 1 packed matrix ciphertext!)
	std::size_t sizeP = MeasureCiphertextSize(ciphertextP);
	res.clientToServerBytes += sizeP;
	res.clientToServerCts += 1;

	// Server Verification Computation (Round 2: V = P * B in ONLY ONE MatMul!)
	auto t_server2_start = high_resolution_clock::now();
	Ciphertext<DCRTPoly> ciphertextV;
	if (canRunRect) {
		ciphertextV =
		    fhe::RectangularJiangMatrixMultiplication(cc, ciphertextP, ciphertextB, params.d, lambda, params.N);
	} else {
		ciphertextV = fhe::OptimalJiangMatrixMultiplication(cc, ciphertextP, ciphertextB, params.d, params.N);
	}
	auto t_server2_stop = high_resolution_clock::now();
	double round2_server_time = duration<double>(t_server2_stop - t_server2_start).count();

	// Communication Round 2: Server -> Client (Only 1 verification product ciphertext!)
	std::size_t sizeV = MeasureCiphertextSize(ciphertextV);
	res.serverToClientBytes += sizeV;
	res.serverToClientCts += 1;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;

	// Client Verification Checks (Decrypt V and verify all lambda challenges simultaneously)
	auto t_vcheck_start = high_resolution_clock::now();
	auto V_mat = DecryptMatrix(cc, keys, ciphertextV, M);

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

	bool allChecksPass = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		auto rC = LinearAlgebra::MultiplyVectorAndMatrix(challenges[k], C_mat);
		for (std::size_t j = 0; j < M; ++j) {
			if (modP(V_mat[k][j]) != modP(rC[j])) {
				allChecksPass = false;
				break;
			}
		}
		if (!allChecksPass) {
			break;
		}
	}
	auto t_vcheck_stop = high_resolution_clock::now();
	double round2_client_check = duration<double>(t_vcheck_stop - t_vcheck_start).count();

	auto t_wall_stop = high_resolution_clock::now();

	res.clientSetupTime = round1_client_setup;
	res.clientVerifyTime = round2_client_prep + round2_client_check;
	res.totalClientTime = res.clientSetupTime + res.clientVerifyTime;

	res.serverComputeTime = round1_server_time + round2_server_time;
	res.totalServerTime = res.serverComputeTime;
	res.totalWallTime = duration<double>(t_wall_stop - t_wall_start).count();
	res.verificationPassed = allChecksPass;

	if (verbose) {
		std::cout << "[SettingII-Batched (" << lambda << " challenges) " << base << "x" << base
		          << "] Server: " << res.serverComputeTime << "s | Client: " << res.totalClientTime
		          << "s | Comm: " << (res.totalBytes / 1024.0) << " KB\n";
	}

	return res;
}

void PrintCsvHeader()
{
	std::cout << "scheme,matrixSize,usefulDim,lambda,securitySoundness,"
	          << "clientSetupTime_s,serverComputeTime_s,clientVerifyTime_s,"
	          << "totalClientTime_s,totalServerTime_s,totalWallTime_s,"
	          << "clientToServerBytes,serverToClientBytes,totalBytes,"
	          << "clientToServerCts,serverToClientCts,verificationPassed\n";
}

void PrintCsvResult(const BenchmarkResult &r)
{
	std::cout << r.scheme << "," << r.matrixSize << "," << r.usefulDim << "," << r.lambda << ",\""
	          << r.securitySoundness << "\"," << r.clientSetupTime << "," << r.serverComputeTime << ","
	          << r.clientVerifyTime << "," << r.totalClientTime << "," << r.totalServerTime << ","
	          << r.totalWallTime << "," << r.clientToServerBytes << "," << r.serverToClientBytes << ","
	          << r.totalBytes << "," << r.clientToServerCts << "," << r.serverToClientCts << ","
	          << (r.verificationPassed ? "true" : "false") << "\n";
}

int main(int argc, char **argv)
{
	std::int64_t base = 16;
	std::size_t lambda = 8;
	std::string format = "text";
	std::string schemeFilter = "all";
	bool useRectangular = false;

	for (std::size_t i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--size" && i + 1 < argc) {
			base = std::stoll(argv[++i]);
		} else if (arg == "--lambda" && i + 1 < argc) {
			lambda = std::stoul(argv[++i]);
		} else if (arg == "--csv" || arg == "--json") {
			format = "csv";
		} else if (arg == "--rectangular") {
			useRectangular = true;
		} else if (arg == "--scheme" && i + 1 < argc) {
			schemeFilter = argv[++i];
		} else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: bench_matmul_overhead [options]\n"
			          << "Options:\n"
			          << "  --size <N>       Matrix size N (e.g. 16, 32, 64) [default: 16]\n"
			          << "  --lambda <L>     Number of keys/challenges [default: 8 (128-bit soundness)]\n"
			          << "  --rectangular    Use Jiang Alg 3 rectangular matmul for verification\n"
			          << "  --scheme <name>  Run specific scheme (dataseal_single, dataseal_multi, "
			             "setting2_sequential, setting2_batched, all)\n"
			          << "  --csv            Output machine-readable CSV\n";
			return 0;
		} else if (i == 1 && arg[0] != '-') {
			base = std::stoll(arg);
		}
	}

	std::vector<BenchmarkResult> results;

	if (schemeFilter == "all" || schemeFilter == "dataseal_single") {
		results.push_back(BenchmarkSingleKeyDataSeal(base, format == "text"));
	}
	if (schemeFilter == "all" || schemeFilter == "dataseal_multi") {
		results.push_back(BenchmarkMultiKeyDataSeal(base, lambda, format == "text", useRectangular));
	}
	if (schemeFilter == "all" || schemeFilter == "setting2_sequential" || schemeFilter == "setting2") {
		results.push_back(BenchmarkSetting2Freivalds(base, lambda, format == "text"));
	}
	if (schemeFilter == "all" || schemeFilter == "setting2_batched" || schemeFilter == "batched") {
		results.push_back(BenchmarkBatchedSetting2Freivalds(base, lambda, format == "text", useRectangular));
	}

	if (format == "csv") {
		PrintCsvHeader();
		for (const auto &r : results) {
			PrintCsvResult(r);
		}
	} else {
		std::cout << "\n"
		          << "========================================================================================="
		             "===========================\n"
		          << "SYSTEMS OVERHEAD SUMMARY: " << base << "x" << base << " (lambda = " << lambda << ")\n"
		          << "========================================================================================="
		             "===========================\n";
		std::cout << std::left << std::setw(26) << "Scheme" << std::setw(12) << "Padded Dim" << std::setw(12)
		          << "Useful Dim" << std::setw(14) << "Server (s)" << std::setw(14) << "Client (s)"
		          << std::setw(14) << "Total (s)" << std::setw(16) << "Comm (KB)" << std::setw(12) << "Verified"
		          << "\n";
		std::cout << std::string(116, '-') << "\n";

		for (const auto &r : results) {
			std::cout << std::left << std::setw(26) << r.scheme << std::setw(12)
			          << (std::to_string(r.matrixSize) + "x" + std::to_string(r.matrixSize))
			          << std::setw(12) << (std::to_string(r.usefulDim) + "x" + std::to_string(r.usefulDim))
			          << std::setw(14) << std::fixed << std::setprecision(4) << r.totalServerTime
			          << std::setw(14) << std::fixed << std::setprecision(4) << r.totalClientTime
			          << std::setw(14) << std::fixed << std::setprecision(4) << r.totalWallTime
			          << std::setw(16) << std::fixed << std::setprecision(2) << (r.totalBytes / 1024.0)
			          << std::setw(12) << (r.verificationPassed ? "YES" : "NO") << "\n";
		}
		std::cout << "========================================================================================="
		             "===========================\n";
	}

	return 0;
}
