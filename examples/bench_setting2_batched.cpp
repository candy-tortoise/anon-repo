#include "fhe_context.hpp"
#include "fhe_mat_mul.hpp"
#include "linear_algebra.hpp"

#include "ciphertext-ser.h"
#include "cryptocontext-ser.h"
#include "key/key-ser.h"
#include "scheme/bgvrns/bgvrns-ser.h"

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

struct BatchedBenchmarkResult {
	std::string variant; // "SettingII-Sequential" vs "SettingII-Batched-2-round-trip"
	std::size_t matrixSize;
	std::size_t lambda;
	std::size_t roundTrips;

	// Timings (seconds)
	double clientSetupTime;
	double serverComputeTime;
	double clientVerifyTime;
	double totalComputeTime;

	// Projected End-to-End Latency with WAN delays
	double latencyLAN_1ms;   // RTT = 1ms
	double latencyWAN_50ms;  // RTT = 50ms (e.g. cross-country)
	double latencyWAN_150ms; // RTT = 150ms (e.g. intercontinental)

	// Communication (bytes)
	std::size_t clientToServerBytes;
	std::size_t serverToClientBytes;
	std::size_t totalBytes;
	std::size_t totalCiphertexts;

	bool verificationPassed;
};

// Measure exact serialized on-the-wire byte size of a ciphertext
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

// ----------------------------------------------------------------------------
// 1. Sequential Setting II: 1 + lambda sequential round trips
// ----------------------------------------------------------------------------
BatchedBenchmarkResult BenchmarkSequentialSetting2(std::size_t M, std::size_t lambda, bool verbose = false)
{
	BatchedBenchmarkResult res;
	res.variant = "Sequential (1+lambda Round Trips)";
	res.matrixSize = M;
	res.lambda = lambda;
	res.roundTrips = 1 + lambda;

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(M, params);

	// Setup
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
	res.clientSetupTime = duration<double>(t_setup_stop - t_setup_start).count();

	std::size_t ct_size = MeasureCiphertextSize(ctA);
	res.clientToServerBytes = 2 * ct_size;

	// Round 1: Initial MatMul
	auto t_s1_start = high_resolution_clock::now();
	auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, params.d, params.N);
	auto t_s1_stop = high_resolution_clock::now();
	double s1_time = duration<double>(t_s1_stop - t_s1_start).count();
	res.serverToClientBytes = ct_size;

	// Sequential Verification Rounds (k = 1 ... lambda)
	auto t_vsetup_start = high_resolution_clock::now();
	auto C_mat = DecryptMatrix(cc, keys, ctC, M);
	auto t_vsetup_stop = high_resolution_clock::now();
	double v_prep_time = duration<double>(t_vsetup_stop - t_vsetup_start).count();

	double total_server_v_time = 0.0;
	double total_client_check_time = 0.0;

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
		res.clientToServerBytes += ct_size;

		// Server verifies challenge k
		auto t_sk_start = high_resolution_clock::now();
		auto ctV_k = fhe::OptimalJiangMatrixMultiplication(cc, ctP_k, ctB, params.d, params.N);
		auto t_sk_stop = high_resolution_clock::now();
		total_server_v_time += duration<double>(t_sk_stop - t_sk_start).count();
		res.serverToClientBytes += ct_size;

		// Client checks challenge k
		auto t_ck_start = high_resolution_clock::now();
		auto V_mat = DecryptMatrix(cc, keys, ctV_k, M);
		auto rC = LinearAlgebra::MultiplyVectorAndMatrix(r, C_mat);
		for (std::size_t j = 0; j < M; ++j) {
			if (modP(V_mat[0][j]) != modP(rC[j])) {
				allPass = false;
				break;
			}
		}
		auto t_ck_stop = high_resolution_clock::now();
		total_client_check_time += duration<double>(t_ck_stop - t_ck_start).count();
	}

	res.serverComputeTime = s1_time + total_server_v_time;
	res.clientVerifyTime = v_prep_time + total_client_check_time;
	res.totalComputeTime = res.clientSetupTime + res.serverComputeTime + res.clientVerifyTime;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;
	res.totalCiphertexts = (2 + lambda) + (1 + lambda);
	res.verificationPassed = allPass;

	// Projected WAN Latencies
	res.latencyLAN_1ms = res.totalComputeTime + res.roundTrips * 0.001;
	res.latencyWAN_50ms = res.totalComputeTime + res.roundTrips * 0.050;
	res.latencyWAN_150ms = res.totalComputeTime + res.roundTrips * 0.150;

	if (verbose) {
		std::cout << "[Sequential Setting II] Server: " << res.serverComputeTime
		          << "s | WAN-50ms Latency: " << res.latencyWAN_50ms << "s (" << res.roundTrips
		          << " round trips)\n";
	}
	return res;
}

// ----------------------------------------------------------------------------
// 2. Batched Setting II: Exactly 2 Round Trips Total (Matrix-Packed Challenges)
// ----------------------------------------------------------------------------
BatchedBenchmarkResult BenchmarkBatchedSetting2(std::size_t M, std::size_t lambda, bool verbose = false,
                                                bool useRectangular = false)
{
	BatchedBenchmarkResult res;
	res.variant = "Batched-2-Round-Trips (Ours)";
	res.matrixSize = M;
	res.lambda = lambda;
	res.roundTrips = 2;

	fhe::FHEParameters params;
	CryptoContext<DCRTPoly> cc = InitCryptoContext(M, params);

	// Setup (Round 1)
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
	res.clientSetupTime = duration<double>(t_setup_stop - t_setup_start).count();

	std::size_t ct_size = MeasureCiphertextSize(ctA);
	res.clientToServerBytes = 2 * ct_size;

	// Round 1 Server MatMul: C = A * B
	auto t_s1_start = high_resolution_clock::now();
	auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, params.d, params.N);
	auto t_s1_stop = high_resolution_clock::now();
	double s1_time = duration<double>(t_s1_stop - t_s1_start).count();
	res.serverToClientBytes = ct_size;

	// [ROUND TRIP 1 COMPLETE]

	// Round 2 Client: Draw ALL lambda challenges at once, compute P = R * A in plaintext
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
			P_rect[k][j] = val; // Pack challenge k into row k
		}
	}

	bool canRunRect = useRectangular && (M % lambda == 0) && (lambda <= M);
	Ciphertext<DCRTPoly> ctP;
	if (canRunRect) {
		auto P_tiled = LinearAlgebra::TileMatrixVertically(P_rect, M);
		ctP = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_tiled)));
	} else {
		LinearAlgebra::Matrix P_mat(M, LinearAlgebra::Vector(M, 0));
		for (std::size_t k = 0; k < lambda; ++k) {
			P_mat[k] = P_rect[k];
		}
		ctP = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_mat)));
	}
	auto t_vprep_stop = high_resolution_clock::now();
	double v_prep_time = duration<double>(t_vprep_stop - t_vprep_start).count();
	res.clientToServerBytes += ct_size;

	// Round 2 Server MatMul: V = P * B (Only 1 MatMul for all lambda challenges!)
	auto t_s2_start = high_resolution_clock::now();
	Ciphertext<DCRTPoly> ctV;
	if (canRunRect) {
		ctV = fhe::RectangularJiangMatrixMultiplication(cc, ctP, ctB, params.d, lambda, params.N);
	} else {
		ctV = fhe::OptimalJiangMatrixMultiplication(cc, ctP, ctB, params.d, params.N);
	}
	auto t_s2_stop = high_resolution_clock::now();
	double s2_time = duration<double>(t_s2_stop - t_s2_start).count();
	res.serverToClientBytes += ct_size;

	// [ROUND TRIP 2 COMPLETE]

	// Round 2 Client: Decrypt V and verify all lambda challenges simultaneously
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
	double v_check_time = duration<double>(t_vcheck_stop - t_vcheck_start).count();

	res.serverComputeTime = s1_time + s2_time; // Exactly 2 MatMuls!
	res.clientVerifyTime = v_prep_time + v_check_time;
	res.totalComputeTime = res.clientSetupTime + res.serverComputeTime + res.clientVerifyTime;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;
	res.totalCiphertexts = 3 + 2; // 5 ciphertexts total!
	res.verificationPassed = allPass;

	// Projected WAN Latencies
	res.latencyLAN_1ms = res.totalComputeTime + res.roundTrips * 0.001;
	res.latencyWAN_50ms = res.totalComputeTime + res.roundTrips * 0.050;
	res.latencyWAN_150ms = res.totalComputeTime + res.roundTrips * 0.150;

	if (verbose) {
		std::cout << "[Batched-2-Round-Trips Setting II] Server: " << res.serverComputeTime
		          << "s | WAN-50ms Latency: " << res.latencyWAN_50ms << "s (" << res.roundTrips
		          << " Round Trips)\n";
	}
	return res;
}

// Print results in CSV format
void PrintCsvHeader()
{
	std::cout << "variant,matrixSize,lambda,roundTrips,"
	          << "clientSetupTime_s,serverComputeTime_s,clientVerifyTime_s,totalComputeTime_s,"
	          << "latencyLAN_1ms_s,latencyWAN_50ms_s,latencyWAN_150ms_s,"
	          << "clientToServerBytes,serverToClientBytes,totalBytes,"
	          << "totalCiphertexts,verificationPassed\n";
}

void PrintCsvResult(const BatchedBenchmarkResult &r)
{
	std::cout << "\"" << r.variant << "\"," << r.matrixSize << "," << r.lambda << "," << r.roundTrips << ","
	          << r.clientSetupTime << "," << r.serverComputeTime << "," << r.clientVerifyTime << ","
	          << r.totalComputeTime << "," << r.latencyLAN_1ms << "," << r.latencyWAN_50ms << ","
	          << r.latencyWAN_150ms << "," << r.clientToServerBytes << "," << r.serverToClientBytes << ","
	          << r.totalBytes << "," << r.totalCiphertexts << "," << (r.verificationPassed ? "true" : "false")
	          << "\n";
}

int main(int argc, char **argv)
{
	std::size_t M = 16;
	std::size_t lambda = 8;
	std::string format = "text"; // "text" or "csv"
	bool useRectangular = false;

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--size" && i + 1 < argc) {
			M = std::stoul(argv[++i]);
		} else if (arg == "--lambda" && i + 1 < argc) {
			lambda = std::stoul(argv[++i]);
		} else if (arg == "--csv" || arg == "--json") {
			format = "csv";
		} else if (arg == "--rectangular") {
			useRectangular = true;
		} else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: bench_setting2_batched [options]\n"
			          << "Options:\n"
			          << "  --size <N>       Matrix size N (e.g. 16, 32, 64) [default: 16]\n"
			          << "  --lambda <L>     Number of challenges [default: 8 (128-bit soundness)]\n"
			          << "  --rectangular    Use Jiang Alg 3 rectangular matmul for verification\n"
			          << "  --csv            Output machine-readable CSV\n";
			return 0;
		}
	}

	std::vector<BatchedBenchmarkResult> results;
	results.push_back(BenchmarkSequentialSetting2(M, lambda, format == "text"));
	results.push_back(BenchmarkBatchedSetting2(M, lambda, format == "text", useRectangular));

	if (format == "csv") {
		PrintCsvHeader();
		for (const auto &r : results) {
			PrintCsvResult(r);
		}
	} else {
		std::cout << "\n"
		          << "========================================================================================="
		             "===========================\n"
		          << "SETTING II: SEQUENTIAL (1+lambda trips) VS. BATCHED (2-trips) COMPARISON (" << M << "x"
		          << M << ", lambda=" << lambda << ")\n"
		          << "========================================================================================="
		             "===========================\n";
		std::cout << std::left << std::setw(26) << "Protocol Variant" << std::setw(8) << "trips"
		          << std::setw(14) << "Server (s)" << std::setw(14) << "Client (s)" << std::setw(16)
		          << "WAN-50ms (s)" << std::setw(16) << "WAN-150ms (s)" << std::setw(14) << "Comm (KB)"
		          << std::setw(8) << "Pass?"
		          << "\n";
		std::cout << std::string(116, '-') << "\n";

		for (const auto &r : results) {
			std::cout << std::left << std::setw(26) << r.variant << std::setw(8) << r.roundTrips
			          << std::setw(14) << std::fixed << std::setprecision(3) << r.serverComputeTime
			          << std::setw(14) << std::fixed << std::setprecision(3)
			          << (r.clientSetupTime + r.clientVerifyTime) << std::setw(16) << std::fixed
			          << std::setprecision(3) << r.latencyWAN_50ms << std::setw(16) << std::fixed
			          << std::setprecision(3) << r.latencyWAN_150ms << std::setw(14) << std::fixed
			          << std::setprecision(1) << (r.totalBytes / 1024.0) << std::setw(8)
			          << (r.verificationPassed ? "YES" : "NO") << "\n";
		}
		std::cout << "========================================================================================="
		             "===========================\n\n";
	}

	return 0;
}
