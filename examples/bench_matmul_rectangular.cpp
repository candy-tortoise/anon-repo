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

struct RectangularBenchmarkResult {
	std::string testCase;       // "SettingII-Batched" or "DataSeal-MultiKey"
	std::string algorithm;      // "Square" vs "Rectangular"
	std::size_t matrixSize;     // d
	std::size_t rectRows;       // ell (active rows, e.g. 8 or 32)
	std::size_t usefulDim;      // useful dimension
	std::size_t lambda;         // lambda
	double serverComputeTime_s; // Server compute time
	double clientSetupTime_s;   // Client setup time
	double clientVerifyTime_s;  // Client verify time
	double totalClientTime_s;   // Client total time
	double totalWallTime_s;     // Total wall time
	std::size_t totalBytes;     // Communication bytes
	bool verificationPassed;    // Correctness check
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
// 1. SETTING II BATCHED FREIVALDS: Square vs. Rectangular
// ============================================================================
std::pair<RectangularBenchmarkResult, RectangularBenchmarkResult>
BenchmarkSetting2BatchedSquareVsRect(std::size_t M, std::size_t lambda, bool verbose = false)
{
	RectangularBenchmarkResult resSquare;
	resSquare.testCase = "SettingII-Batched";
	resSquare.algorithm = "Square";
	resSquare.matrixSize = M;
	resSquare.rectRows = lambda;
	resSquare.usefulDim = M;
	resSquare.lambda = lambda;

	RectangularBenchmarkResult resRect = resSquare;
	resRect.algorithm = "Rectangular";

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(M, params);

	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(params.d));

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
	double clientSetupTime = duration<double>(t_setup_stop - t_setup_start).count();
	resSquare.clientSetupTime_s = clientSetupTime;
	resRect.clientSetupTime_s = clientSetupTime;

	std::size_t ct_size = MeasureCiphertextSize(ctA);
	std::size_t commCtoS_round1 = 2 * ct_size;

	// Round 1 MatMul: C = A * B
	auto t_s1_start = high_resolution_clock::now();
	auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, params.d, params.N);
	auto t_s1_stop = high_resolution_clock::now();
	double s1_time = duration<double>(t_s1_stop - t_s1_start).count();

	// Client prepares lambda challenges locally in plaintext
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

	// Prepare P for Square (zero-padded to M x M)
	LinearAlgebra::Matrix P_square(M, LinearAlgebra::Vector(M, 0));
	for (std::size_t k = 0; k < lambda; ++k) {
		P_square[k] = P_rect[k];
	}
	auto ctP_square =
	    cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_square)));

	// Prepare P for Rectangular (tiled vertically M / lambda times to M x M)
	auto P_tiled = LinearAlgebra::TileMatrixVertically(P_rect, M);
	auto ctP_tiled = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_tiled)));
	auto t_vprep_stop = high_resolution_clock::now();
	double vprep_time = duration<double>(t_vprep_stop - t_vprep_start).count();

	// --- A. EVALUATE SQUARE VERIFICATION ---
	auto t_s2_square_start = high_resolution_clock::now();
	auto ctV_square = fhe::OptimalJiangMatrixMultiplication(cc, ctP_square, ctB, params.d, params.N);
	auto t_s2_square_stop = high_resolution_clock::now();
	double s2_square_time = duration<double>(t_s2_square_stop - t_s2_square_start).count();

	// --- B. EVALUATE RECTANGULAR VERIFICATION ---
	auto t_s2_rect_start = high_resolution_clock::now();
	auto ctV_rect = fhe::RectangularJiangMatrixMultiplication(cc, ctP_tiled, ctB, params.d, lambda, params.N);
	auto t_s2_rect_stop = high_resolution_clock::now();
	double s2_rect_time = duration<double>(t_s2_rect_stop - t_s2_rect_start).count();

	// Client checks
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

	// Check Square
	auto t_check_sq_start = high_resolution_clock::now();
	auto V_square = DecryptMatrix(cc, keys, ctV_square, M);
	bool passSquare = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		auto rC = LinearAlgebra::MultiplyVectorAndMatrix(challenges[k], C_mat);
		for (std::size_t j = 0; j < M; ++j) {
			if (modP(V_square[k][j]) != modP(rC[j])) {
				passSquare = false;
				break;
			}
		}
		if (!passSquare) {
			break;
		}
	}
	auto t_check_sq_stop = high_resolution_clock::now();
	double check_sq_time = duration<double>(t_check_sq_stop - t_check_sq_start).count();

	// Check Rectangular
	auto t_check_rc_start = high_resolution_clock::now();
	auto V_rect = DecryptMatrix(cc, keys, ctV_rect, M);
	bool passRect = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		auto rC = LinearAlgebra::MultiplyVectorAndMatrix(challenges[k], C_mat);
		for (std::size_t j = 0; j < M; ++j) {
			if (modP(V_rect[k][j]) != modP(rC[j])) {
				passRect = false;
				break;
			}
		}
		if (!passRect) {
			break;
		}
	}
	auto t_check_rc_stop = high_resolution_clock::now();
	double check_rc_time = duration<double>(t_check_rc_stop - t_check_rc_start).count();

	// Finalize metrics
	resSquare.serverComputeTime_s = s1_time + s2_square_time;
	resSquare.clientVerifyTime_s = vprep_time + check_sq_time;
	resSquare.totalClientTime_s = resSquare.clientSetupTime_s + resSquare.clientVerifyTime_s;
	resSquare.totalWallTime_s = resSquare.totalClientTime_s + resSquare.serverComputeTime_s;
	resSquare.totalBytes = commCtoS_round1 + ct_size + ct_size + ct_size; // 5 ciphertexts
	resSquare.verificationPassed = passSquare;

	resRect.serverComputeTime_s = s1_time + s2_rect_time;
	resRect.clientVerifyTime_s = vprep_time + check_rc_time;
	resRect.totalClientTime_s = resRect.clientSetupTime_s + resRect.clientVerifyTime_s;
	resRect.totalWallTime_s = resRect.totalClientTime_s + resRect.serverComputeTime_s;
	resRect.totalBytes = resSquare.totalBytes;
	resRect.verificationPassed = passRect;

	if (verbose) {
		std::cout << "[SettingII " << M << "x" << M << " | lambda=" << lambda << "]\n"
		          << "  Square Server Time     : " << resSquare.serverComputeTime_s
		          << "s (Verify MatMul: " << s2_square_time << "s)\n"
		          << "  Rectangular Server Time: " << resRect.serverComputeTime_s
		          << "s (Verify MatMul: " << s2_rect_time << "s)\n"
		          << "  Verify Speedup         : " << (s2_square_time / s2_rect_time) << "x\n";
	}

	return {resSquare, resRect};
}

// ============================================================================
// 2. DATASEAL MULTI-KEY: Square vs. Rectangular (When ell divides d)
// ============================================================================
std::pair<RectangularBenchmarkResult, RectangularBenchmarkResult>
BenchmarkDataSealMultiKeySquareVsRect(std::size_t base, std::size_t lambda, bool verbose = false)
{
	// For DataSeal Multi-Key:
	// If base == 16: usefulDim = 16, ell = 32, padded = 32 (Square baseline)
	// If base >= 32: usefulDim = 16, ell = 32, padded = 64 (Rectangular: ell=32 divides d=64!)
	std::size_t usefulDim = 16;
	std::size_t ell = usefulDim + 2 * lambda; // 16 + 16 = 32 active rows
	std::size_t M_padded = (base <= 16) ? 32 : 64;

	RectangularBenchmarkResult resSquare;
	resSquare.testCase = "DataSeal-MultiKey";
	resSquare.algorithm = "Square";
	resSquare.matrixSize = M_padded;
	resSquare.rectRows = ell;
	resSquare.usefulDim = usefulDim;
	resSquare.lambda = lambda;

	RectangularBenchmarkResult resRect = resSquare;
	resRect.algorithm = "Rectangular";

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(M_padded, params);

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

	// Square inputs: pad lastA to M_padded x M_padded
	auto paddedA_sq = LinearAlgebra::PadMatrix(lastA, M_padded);
	auto paddedB = LinearAlgebra::PadMatrix(B, M_padded);
	for (auto &row : paddedA_sq) {
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

	auto ctA_sq = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(paddedA_sq)));
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(paddedB)));

	// Rectangular inputs: if ell divides M_padded, tile lastA vertically
	bool canRunRect = (M_padded % ell == 0) && (ell < M_padded);
	Ciphertext<DCRTPoly> ctA_rect;
	if (canRunRect) {
		auto paddedA_rect_base = LinearAlgebra::PadMatrix(lastA, ell, M_padded);
		for (auto &row : paddedA_rect_base) {
			for (auto &v : row) {
				v %= params.P;
				if (v < 0) {
					v += params.P;
				}
			}
		}
		auto tiledA = LinearAlgebra::TileMatrixVertically(paddedA_rect_base, M_padded);
		ctA_rect = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(tiledA)));
	}
	auto t_setup_stop = high_resolution_clock::now();
	double setupTime = duration<double>(t_setup_stop - t_setup_start).count();
	resSquare.clientSetupTime_s = setupTime;
	resRect.clientSetupTime_s = setupTime;

	std::size_t ct_size = MeasureCiphertextSize(ctA_sq);
	resSquare.totalBytes = 3 * ct_size;
	resRect.totalBytes = 3 * ct_size;

	// --- A. EVALUATE SQUARE DATASEAL ---
	auto t_sq_start = high_resolution_clock::now();
	auto ctC_sq = fhe::OptimalJiangMatrixMultiplication(cc, ctA_sq, ctB, params.d, params.N);
	auto t_sq_stop = high_resolution_clock::now();
	double sq_time = duration<double>(t_sq_stop - t_sq_start).count();
	resSquare.serverComputeTime_s = sq_time;

	// --- B. EVALUATE RECTANGULAR DATASEAL (if applicable) ---
	double rc_time = sq_time;
	Ciphertext<DCRTPoly> ctC_rect = ctC_sq;
	if (canRunRect) {
		auto t_rc_start = high_resolution_clock::now();
		ctC_rect = fhe::RectangularJiangMatrixMultiplication(cc, ctA_rect, ctB, params.d, ell, params.N);
		auto t_rc_stop = high_resolution_clock::now();
		rc_time = duration<double>(t_rc_stop - t_rc_start).count();
	}
	resRect.serverComputeTime_s = rc_time;

	// Client verification
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

	auto t_v_sq_start = high_resolution_clock::now();
	auto fheResultSq = DecryptMatrix(cc, keys, ctC_sq, M_padded);
	auto fheMatrixC_sq = fheResultSq;
	fheMatrixC_sq.resize(usefulDim);

	bool passSq = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		auto h1 = fheResultSq[usefulDim + 2 * k];
		auto h2 = fheResultSq[usefulDim + 2 * k + 1];
		auto compH1 = LinearAlgebra::MultiplyVectorAndMatrix(s_keys[k], fheMatrixC_sq);
		for (std::size_t j = 0; j < usefulDim; ++j) {
			if (modP(compH1[j]) != modP(h1[j])) {
				passSq = false;
				break;
			}
			if (modP(goldenOutputs[k][j]) != modP(h2[j])) {
				passSq = false;
				break;
			}
		}
		if (!passSq) {
			break;
		}
	}
	auto t_v_sq_stop = high_resolution_clock::now();
	resSquare.clientVerifyTime_s = duration<double>(t_v_sq_stop - t_v_sq_start).count();
	resSquare.totalClientTime_s = resSquare.clientSetupTime_s + resSquare.clientVerifyTime_s;
	resSquare.totalWallTime_s = resSquare.totalClientTime_s + resSquare.serverComputeTime_s;
	resSquare.verificationPassed = passSq;

	auto t_v_rc_start = high_resolution_clock::now();
	auto fheResultRc = DecryptMatrix(cc, keys, ctC_rect, M_padded);
	auto fheMatrixC_rc = fheResultRc;
	fheMatrixC_rc.resize(usefulDim);

	bool passRc = true;
	for (std::size_t k = 0; k < lambda; ++k) {
		auto h1 = fheResultRc[usefulDim + 2 * k];
		auto h2 = fheResultRc[usefulDim + 2 * k + 1];
		auto compH1 = LinearAlgebra::MultiplyVectorAndMatrix(s_keys[k], fheMatrixC_rc);
		for (std::size_t j = 0; j < usefulDim; ++j) {
			if (modP(compH1[j]) != modP(h1[j])) {
				passRc = false;
				break;
			}
			if (modP(goldenOutputs[k][j]) != modP(h2[j])) {
				passRc = false;
				break;
			}
		}
		if (!passRc) {
			break;
		}
	}
	auto t_v_rc_stop = high_resolution_clock::now();
	resRect.clientVerifyTime_s = duration<double>(t_v_rc_stop - t_v_rc_start).count();
	resRect.totalClientTime_s = resRect.clientSetupTime_s + resRect.clientVerifyTime_s;
	resRect.totalWallTime_s = resRect.totalClientTime_s + resRect.serverComputeTime_s;
	resRect.verificationPassed = passRc;

	return {resSquare, resRect};
}

void PrintCsvHeader()
{
	std::cout << "testCase,algorithm,matrixSize,rectRows,usefulDim,lambda,"
	          << "serverComputeTime_s,clientSetupTime_s,clientVerifyTime_s,"
	          << "totalClientTime_s,totalWallTime_s,totalBytes,verificationPassed\n";
}

void PrintCsvResult(const RectangularBenchmarkResult &r)
{
	std::cout << "\"" << r.testCase << "\",\"" << r.algorithm << "\"," << r.matrixSize << "," << r.rectRows << ","
	          << r.usefulDim << "," << r.lambda << "," << r.serverComputeTime_s << "," << r.clientSetupTime_s << ","
	          << r.clientVerifyTime_s << "," << r.totalClientTime_s << "," << r.totalWallTime_s << ","
	          << r.totalBytes << "," << (r.verificationPassed ? "true" : "false") << "\n";
}

int main(int argc, char **argv)
{
	std::size_t size = 16;
	std::size_t lambda = 8;
	std::string format = "csv";

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--size" && i + 1 < argc) {
			size = std::stoul(argv[++i]);
		} else if (arg == "--lambda" && i + 1 < argc) {
			lambda = std::stoul(argv[++i]);
		} else if (arg == "--format" && i + 1 < argc) {
			format = argv[++i];
		} else if (arg == "--help" || arg == "-h") {
			std::cout
			    << "Usage: bench_matmul_rectangular [--size <N>] [--lambda <L>] [--format <csv|text>]\n";
			return 0;
		}
	}

	std::vector<RectangularBenchmarkResult> all_results;

	// 1. Setting II Batched: Square vs Rectangular
	auto p1 = BenchmarkSetting2BatchedSquareVsRect(size, lambda, format == "text");
	all_results.push_back(p1.first);
	all_results.push_back(p1.second);

	// 2. DataSeal Multi-Key: Square vs Rectangular
	auto p2 = BenchmarkDataSealMultiKeySquareVsRect(size, lambda, format == "text");
	all_results.push_back(p2.first);
	all_results.push_back(p2.second);

	if (format == "csv") {
		PrintCsvHeader();
		for (const auto &r : all_results) {
			PrintCsvResult(r);
		}
	} else {
		std::cout
		    << "\n=========================================================================================\n";
		std::cout << "SQUARE VS. RECTANGULAR MATRIX MULTIPLICATION BENCHMARK (Size " << size << "x" << size
		          << ", lambda=" << lambda << ")\n";
		std::cout
		    << "=========================================================================================\n";
		std::cout << std::left << std::setw(22) << "Scheme" << std::setw(14) << "Algorithm" << std::setw(10)
		          << "Dim" << std::setw(12) << "ActiveRows" << std::setw(16) << "Server (s)" << std::setw(16)
		          << "Client (s)" << std::setw(16) << "Wall (s)" << std::setw(10) << "Pass?" << "\n";
		std::cout << std::string(106, '-') << "\n";
		for (const auto &r : all_results) {
			std::cout << std::left << std::setw(22) << r.testCase << std::setw(14) << r.algorithm
			          << std::setw(10)
			          << (std::to_string(r.matrixSize) + "x" + std::to_string(r.matrixSize))
			          << std::setw(12) << r.rectRows << std::setw(16) << std::fixed << std::setprecision(4)
			          << r.serverComputeTime_s << std::setw(16) << std::fixed << std::setprecision(4)
			          << r.totalClientTime_s << std::setw(16) << std::fixed << std::setprecision(4)
			          << r.totalWallTime_s << std::setw(10) << (r.verificationPassed ? "YES" : "NO")
			          << "\n";
		}
		std::cout
		    << "=========================================================================================\n";
	}

	return 0;
}
