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

bool IsPrimeModulus(uint64_t n)
{
	if (n < 2) {
		return false;
	}
	if (n % 2 == 0) {
		return (n == 2);
	}
	for (uint64_t i = 3; i * i <= n; i += 2) {
		if (n % i == 0) {
			return false;
		}
	}
	return true;
}

uint64_t FindPrimeModulus(uint64_t min_val, uint32_t N)
{
	uint64_t step = 2ULL * N;
	uint64_t k = (min_val / step);
	if (k * step + 1 < min_val) {
		k++;
	}
	while (true) {
		uint64_t cand = k * step + 1;
		if (cand >= min_val && IsPrimeModulus(cand)) {
			return cand;
		}
		k++;
	}
}

struct ModulusSweepBenchmarkResult {
	std::string protocol;     // "Sequential" or "Batched-2-Round-Trips"
	std::size_t matrixDim;    // Useful matrix dimension d (e.g. 16, 32)
	int targetBits;           // Nominal target bits (e.g. 16, 24, 30, 32, 34)
	std::string t_str;        // Actual prime t as string
	uint64_t modulusP;        // Actual prime t
	double actualBits;        // log2(P)
	std::size_t challenges_s; // s = ceil(128 / log2(P))
	double achievedSoundness; // s * log2(P)
	std::size_t roundTrips;   // 1+s for Sequential, 2 for Batched

	// FHE parameters
	uint32_t ringDim;        // N (e.g. 16384)
	uint32_t log2Q;          // Total ciphertext modulus bit length
	std::size_t numLimbs;    // Number of RNS limbs
	double ciphertextSizeKB; // Serialized size of 1 ciphertext

	// Latency (seconds)
	double clientSetupTime;  // KeyGen + Encryption
	double clientVerifyTime; // Decryption + check computation
	double clientTotalTime;  // Setup + Verify
	double serverRound1Time; // Initial MatMul C = A * B
	double serverVerifyTime; // Homomorphic verification MatMul(s)
	double serverTotalTime;  // Round 1 + Verification
	double totalComputeTime; // Client + Server

	// Projected Network Latencies (Compute + Round Trips * delay)
	double latencyLAN_1ms;   // Delay = 1ms
	double latencyWAN_50ms;  // Delay = 50ms (cross-country)
	double latencyWAN_150ms; // Delay = 150ms (intercontinental)

	// Communication (bytes)
	std::size_t clientToServerBytes;
	std::size_t serverToClientBytes;
	std::size_t totalBytes;
	std::size_t ciphertextCount;

	// Speedup metrics relative to Sequential baseline
	double serverSpeedup; // T_server(Seq) / T_server(Batch)
	double wan50Speedup;  // T_wan50(Seq) / T_wan50(Batch)
	double commReduction; // Bytes(Seq) / Bytes(Batch)

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

// ----------------------------------------------------------------------------
// 1. Sequential Freivalds: 1 + s sequential round trips
// ----------------------------------------------------------------------------
ModulusSweepBenchmarkResult BenchmarkSequentialFreivalds(std::size_t M, int targetBits, uint64_t P, std::size_t s,
                                                         bool verbose = false)
{
	ModulusSweepBenchmarkResult res;
	res.protocol = "Sequential";
	res.matrixDim = M;
	res.targetBits = targetBits;
	res.t_str = std::to_string(P);
	res.modulusP = P;
	res.actualBits = std::log2(static_cast<double>(P));
	res.challenges_s = s;
	res.achievedSoundness = s * res.actualBits;
	res.roundTrips = 1 + s;
	res.serverSpeedup = 1.0;
	res.wan50Speedup = 1.0;
	res.commReduction = 1.0;

	uint32_t N = 16384;
	std::size_t required_slots = M * M;
	while (N < 2 * required_slots) {
		N *= 2;
	}
	res.ringDim = N;

	CCParams<CryptoContextBGVRNS> parameters;
	parameters.SetMultiplicativeDepth(3);
	parameters.SetPlaintextModulus(P);
	parameters.SetRingDim(N);
	if (verbose) {
		std::cout << "Ring Dimension = " << N << "\n";
		std::cout << "Plaintext Mod  = " << P << "\n";
	}
	parameters.SetSecurityLevel(HEStd_128_classic);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);

	if (verbose) {
		fhe::PrintInfo(cc);
	}

	auto el = cc->GetElementParams();
	res.log2Q = el->GetModulus().GetMSB();
	res.numLimbs = el->GetParams().size();

	// Client Setup: KeyGen + Encrypt(A), Encrypt(B)
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(M));

	auto A = LinearAlgebra::GenerateSquareMatrixWithRandomValues(M);
	auto B = LinearAlgebra::GenerateSquareMatrixWithRandomValues(M);
	for (auto &row : A) {
		for (auto &v : row) {
			v %= P;
			if (v < 0) {
				v += P;
			}
		}
	}
	for (auto &row : B) {
		for (auto &v : row) {
			v %= P;
			if (v < 0) {
				v += P;
			}
		}
	}

	auto ctA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(A)));
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B)));
	auto t_setup_stop = high_resolution_clock::now();
	res.clientSetupTime = duration<double>(t_setup_stop - t_setup_start).count();

	std::size_t ct_size = MeasureCiphertextSize(ctA);
	res.ciphertextSizeKB = static_cast<double>(ct_size) / 1024.0;
	res.clientToServerBytes = 2 * ct_size;

	// Round 1: Server performs C = A * B
	auto t_s1_start = high_resolution_clock::now();
	auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, M, N);
	auto t_s1_stop = high_resolution_clock::now();
	res.serverRound1Time = duration<double>(t_s1_stop - t_s1_start).count();
	res.serverToClientBytes = ct_size;

	// [ROUND TRIP 1 COMPLETE]

	// Sequential Verification: s separate challenges across s rounds
	auto t_vsetup_start = high_resolution_clock::now();
	auto C_mat = DecryptMatrix(cc, keys, ctC, M);
	auto t_vsetup_stop = high_resolution_clock::now();
	double v_prep_time = duration<double>(t_vsetup_stop - t_vsetup_start).count();

	double total_server_v_time = 0.0;
	double total_client_check_time = 0.0;
	bool allPass = true;

	auto modP = [P](std::int64_t v) {
		std::int64_t r = v % static_cast<std::int64_t>(P);
		if (r < 0) {
			r += P;
		}
		if (r > static_cast<std::int64_t>(P) / 2) {
			r -= P;
		}
		return r;
	};

	for (std::size_t k = 0; k < s; ++k) {
		// Client creates challenge k
		auto t_ck_prep_start = high_resolution_clock::now();
		auto r = GenerateUniformFieldVector(M, P);
		auto vA = LinearAlgebra::MultiplyVectorAndMatrix(r, A);
		for (auto &x : vA) {
			x %= P;
			if (x < 0) {
				x += P;
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
		total_client_check_time += duration<double>(t_ck_prep_stop - t_ck_prep_start).count();
		res.clientToServerBytes += ct_size;

		// Server computes challenge k: V_k = P_k * B
		auto t_sk_start = high_resolution_clock::now();
		auto ctV_k = fhe::OptimalJiangMatrixMultiplication(cc, ctP_k, ctB, M, N);
		auto t_sk_stop = high_resolution_clock::now();
		total_server_v_time += duration<double>(t_sk_stop - t_sk_start).count();
		res.serverToClientBytes += ct_size;

		// Client checks challenge k: V_k[0] == r * C
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
		total_client_check_time += duration<double>(t_ck_check_stop - t_ck_check_start).count();
	}

	res.serverVerifyTime = total_server_v_time;
	res.serverTotalTime = res.serverRound1Time + res.serverVerifyTime;
	res.clientVerifyTime = v_prep_time + total_client_check_time;
	res.clientTotalTime = res.clientSetupTime + res.clientVerifyTime;
	res.totalComputeTime = res.clientTotalTime + res.serverTotalTime;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;
	res.ciphertextCount = (2 + s) + (1 + s);
	res.verificationPassed = allPass;

	// Projected Network Latencies
	res.latencyLAN_1ms = res.totalComputeTime + res.roundTrips * 0.001;
	res.latencyWAN_50ms = res.totalComputeTime + res.roundTrips * 0.050;
	res.latencyWAN_150ms = res.totalComputeTime + res.roundTrips * 0.150;

	if (verbose) {
		std::cout << "[Sequential Freivalds | t=" << P << " (" << targetBits << "b), s=" << s
		          << "] Server=" << res.serverTotalTime << "s | WAN-50ms=" << res.latencyWAN_50ms << "s ("
		          << res.roundTrips << " Round Trips)\n";
	}
	return res;
}

// ----------------------------------------------------------------------------
// 2. Batched Freivalds (CATCH-II / Ours): Exactly 2 Round Trips Total
// ----------------------------------------------------------------------------
ModulusSweepBenchmarkResult BenchmarkBatchedFreivalds(std::size_t M, int targetBits, uint64_t P, std::size_t s,
                                                      bool verbose = false)
{
	ModulusSweepBenchmarkResult res;
	res.protocol = "Batched-2RTT";
	res.matrixDim = M;
	res.targetBits = targetBits;
	res.t_str = std::to_string(P);
	res.modulusP = P;
	res.actualBits = std::log2(static_cast<double>(P));
	res.challenges_s = s;
	res.achievedSoundness = s * res.actualBits;
	res.roundTrips = 2;

	uint32_t N = 16384;
	std::size_t required_slots = M * M;
	while (N < 2 * required_slots) {
		N *= 2;
	}
	res.ringDim = N;

	CCParams<CryptoContextBGVRNS> parameters;
	parameters.SetMultiplicativeDepth(3);
	parameters.SetPlaintextModulus(P);
	parameters.SetRingDim(N);
	if (verbose) {
		std::cout << "Ring Dimension = " << N << "\n";
		std::cout << "Plaintext Mod  = " << P << "\n";
	}
	parameters.SetSecurityLevel(HEStd_128_classic);

	CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
	cc->Enable(PKE);
	cc->Enable(KEYSWITCH);
	cc->Enable(LEVELEDSHE);
	cc->Enable(ADVANCEDSHE);

	if (verbose) {
		fhe::PrintInfo(cc);
	}

	auto el = cc->GetElementParams();
	res.log2Q = el->GetModulus().GetMSB();
	res.numLimbs = el->GetParams().size();

	// Client Setup: KeyGen + Encrypt(A), Encrypt(B)
	auto t_setup_start = high_resolution_clock::now();
	KeyPair<DCRTPoly> keys = cc->KeyGen();
	cc->EvalMultKeysGen(keys.secretKey);
	cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(M));

	auto A = LinearAlgebra::GenerateSquareMatrixWithRandomValues(M);
	auto B = LinearAlgebra::GenerateSquareMatrixWithRandomValues(M);
	for (auto &row : A) {
		for (auto &v : row) {
			v %= P;
			if (v < 0) {
				v += P;
			}
		}
	}
	for (auto &row : B) {
		for (auto &v : row) {
			v %= P;
			if (v < 0) {
				v += P;
			}
		}
	}

	auto ctA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(A)));
	auto ctB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(B)));
	auto t_setup_stop = high_resolution_clock::now();
	res.clientSetupTime = duration<double>(t_setup_stop - t_setup_start).count();

	std::size_t ct_size = MeasureCiphertextSize(ctA);
	res.ciphertextSizeKB = static_cast<double>(ct_size) / 1024.0;
	res.clientToServerBytes = 2 * ct_size;

	// Round 1: Server computes C = A * B
	auto t_s1_start = high_resolution_clock::now();
	auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, M, N);
	auto t_s1_stop = high_resolution_clock::now();
	res.serverRound1Time = duration<double>(t_s1_stop - t_s1_start).count();
	res.serverToClientBytes = ct_size;

	// [ROUND TRIP 1 COMPLETE]

	// Round 2 Client: Draw ALL s challenges at once, pack into P_rect
	auto t_vprep_start = high_resolution_clock::now();
	auto C_mat = DecryptMatrix(cc, keys, ctC, M);

	// Determine rectangular packing parameter ell (next power of 2 >= s that divides M)
	std::size_t ell = 1;
	while (ell < s) {
		ell <<= 1;
	}
	bool canRunRect = (ell <= M) && (M % ell == 0);
	if (!canRunRect) {
		ell = M; // Fall back to full M x M square
	}

	std::vector<LinearAlgebra::Vector> challenges(s);
	LinearAlgebra::Matrix P_rect(ell, LinearAlgebra::Vector(M, 0));

	for (std::size_t k = 0; k < s; ++k) {
		challenges[k] = GenerateUniformFieldVector(M, P);
		auto vA = LinearAlgebra::MultiplyVectorAndMatrix(challenges[k], A);
		for (std::size_t j = 0; j < M; ++j) {
			int64_t val = vA[j] % static_cast<int64_t>(P);
			if (val < 0) {
				val += P;
			}
			P_rect[k][j] = val; // Pack challenge k into row k
		}
	}

	Ciphertext<DCRTPoly> ctP;
	if (canRunRect && ell < M) {
		auto P_tiled = LinearAlgebra::TileMatrixVertically(P_rect, M);
		ctP = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_tiled)));
	} else {
		ctP = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(LinearAlgebra::FlattenRowMajor(P_rect)));
	}
	auto t_vprep_stop = high_resolution_clock::now();
	double v_prep_time = duration<double>(t_vprep_stop - t_vprep_start).count();
	res.clientToServerBytes += ct_size;

	// Round 2 Server: Evaluate V = P * B (Only 1 MatMul with ell multiplications for ALL s challenges!)
	auto t_s2_start = high_resolution_clock::now();
	Ciphertext<DCRTPoly> ctV;
	if (canRunRect && ell < M) {
		ctV = fhe::RectangularJiangMatrixMultiplication(cc, ctP, ctB, M, ell, N);
	} else {
		ctV = fhe::OptimalJiangMatrixMultiplication(cc, ctP, ctB, M, N);
	}
	auto t_s2_stop = high_resolution_clock::now();
	res.serverVerifyTime = duration<double>(t_s2_stop - t_s2_start).count();
	res.serverToClientBytes += ct_size;

	// [ROUND TRIP 2 COMPLETE]

	// Round 2 Client: Decrypt V and verify all s challenges simultaneously
	auto t_vcheck_start = high_resolution_clock::now();
	auto V_mat = DecryptMatrix(cc, keys, ctV, M);

	auto modP = [P](std::int64_t v) {
		std::int64_t r = v % static_cast<std::int64_t>(P);
		if (r < 0) {
			r += P;
		}
		if (r > static_cast<std::int64_t>(P) / 2) {
			r -= P;
		}
		return r;
	};

	bool allPass = true;
	for (std::size_t k = 0; k < s; ++k) {
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

	res.serverTotalTime = res.serverRound1Time + res.serverVerifyTime;
	res.clientVerifyTime = v_prep_time + v_check_time;
	res.clientTotalTime = res.clientSetupTime + res.clientVerifyTime;
	res.totalComputeTime = res.clientTotalTime + res.serverTotalTime;
	res.totalBytes = res.clientToServerBytes + res.serverToClientBytes;
	res.ciphertextCount = 3 + 2; // Exactly 5 ciphertexts total!
	res.verificationPassed = allPass;

	// Projected Network Latencies
	res.latencyLAN_1ms = res.totalComputeTime + res.roundTrips * 0.001;
	res.latencyWAN_50ms = res.totalComputeTime + res.roundTrips * 0.050;
	res.latencyWAN_150ms = res.totalComputeTime + res.roundTrips * 0.150;

	if (verbose) {
		std::cout << "[Batched-2-Round Freivalds | t=" << P << " (" << targetBits << "b), s=" << s
		          << "] Server=" << res.serverTotalTime << "s | WAN-50ms=" << res.latencyWAN_50ms << "s ("
		          << res.roundTrips << " Round Trips)\n";
	}
	return res;
}

// Print results in CSV format
void PrintCsvHeader()
{
	std::cout << "matrixDim,targetBits,modulusP,modulusBits,challenges_s,achievedSoundnessBits,"
	          << "protocol,roundTrips,ringDim_N,log2Q,numLimbs,ciphertextSizeKB,"
	          << "clientSetupTime_s,clientVerifyTime_s,clientTotalTime_s,"
	          << "serverRound1Time_s,serverVerifyTime_s,serverTotalTime_s,"
	          << "totalComputeTime_s,latencyLAN_1ms_s,latencyWAN_50ms_s,latencyWAN_150ms_s,"
	          << "clientToServerBytes,serverToClientBytes,totalCommKB,ciphertextCount,"
	          << "serverSpeedup,wan50Speedup,commReduction,verificationPassed\n";
}

void PrintCsvResult(const ModulusSweepBenchmarkResult &r)
{
	double totalCommKB = static_cast<double>(r.totalBytes) / 1024.0;
	std::cout << r.matrixDim << "," << r.targetBits << "," << r.modulusP << "," << std::fixed
	          << std::setprecision(4) << r.actualBits << "," << r.challenges_s << "," << std::fixed
	          << std::setprecision(2) << r.achievedSoundness << ","
	          << "\"" << r.protocol << "\"," << r.roundTrips << "," << r.ringDim << "," << r.log2Q << ","
	          << r.numLimbs << "," << std::fixed << std::setprecision(2) << r.ciphertextSizeKB << ","
	          << std::setprecision(6) << r.clientSetupTime << "," << r.clientVerifyTime << "," << r.clientTotalTime
	          << "," << r.serverRound1Time << "," << r.serverVerifyTime << "," << r.serverTotalTime << ","
	          << r.totalComputeTime << "," << r.latencyLAN_1ms << "," << r.latencyWAN_50ms << ","
	          << r.latencyWAN_150ms << "," << r.clientToServerBytes << "," << r.serverToClientBytes << ","
	          << std::setprecision(2) << totalCommKB << "," << r.ciphertextCount << "," << std::setprecision(2)
	          << r.serverSpeedup << "," << r.wan50Speedup << "," << r.commReduction << ","
	          << (r.verificationPassed ? "true" : "false") << "\n";
}

int main(int argc, char **argv)
{
	std::size_t matrixDim = 16;
	bool emitCsv = false;
	bool verbose = false;
	bool includeConservativeS9 = true;
	std::vector<int> bit_targets = {16, 24, 30, 32, 34};

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--csv") {
			emitCsv = true;
		} else if (arg == "--verbose" || arg == "-v") {
			verbose = true;
		} else if (arg == "--dim" && i + 1 < argc) {
			matrixDim = std::stoul(argv[++i]);
		} else if (arg == "--no-s9") {
			includeConservativeS9 = false;
		} else if (arg == "--bits" && i + 1 < argc) {
			bit_targets.clear();
			std::string list = argv[++i];
			std::stringstream ss(list);
			std::string token;
			while (std::getline(ss, token, ',')) {
				if (!token.empty()) {
					bit_targets.push_back(std::stoi(token));
				}
			}
		} else if (arg == "--help" || arg == "-h") {
			std::cout << "Usage: " << argv[0] << " [options]\n"
			          << "Options:\n"
			          << "  --dim <N>        Matrix dimension (e.g. 16, 32) [default: 16]\n"
			          << "  --bits <list>    Comma-separated bit sizes (default: 16,24,30,32,34)\n"
			          << "  --no-s9          Exclude conservative 16-bit s=9 case\n"
			          << "  --csv            Output machine-readable CSV\n"
			          << "  --verbose, -v    Verbose diagnostic logs\n";
			return 0;
		}
	}

	uint32_t N = 16384;
	std::size_t required_slots = matrixDim * matrixDim;
	while (N < 2 * required_slots) {
		N *= 2;
	}

	struct SweepCase {
		int targetBits;
		uint64_t P;
		std::size_t s;
		std::string note;
	};

	std::vector<SweepCase> cases;

	for (int b : bit_targets) {
		uint64_t min_v = (b <= 63) ? (1ULL << (b - 1)) : 0;
		uint64_t P = FindPrimeModulus(min_v, N);
		double log2p = std::log2(static_cast<double>(P));
		std::size_t s = static_cast<std::size_t>(std::ceil(128.0 / log2p));

		// For 16-bit, if requested, add conservative s=9 first (matching 2^15 < t < 2^16)
		if (b == 16 && includeConservativeS9) {
			cases.push_back({16, P, 9, "2^15 < t < 2^16 (Conservative s=9)"});
		}
		cases.push_back({b, P, s, "Target " + std::to_string(b) + "-bit"});
	}

	if (emitCsv) {
		PrintCsvHeader();
	} else {
		std::cout
		    << "========================================================================================\n"
		    << "FREIVALDS MODULUS-VERSUS-CHALLENGE-COUNT SWEEP BENCHMARK (128-bit Soundness Target)\n"
		    << "Matrix Dimension: " << matrixDim << "x" << matrixDim << " | Ring Dim N: " << N << "\n"
		    << "========================================================================================\n";
	}

	for (const auto &c : cases) {
		if (verbose && !emitCsv) {
			std::cout << "\n--> Running " << c.note << " | P=" << c.P << ", s=" << c.s
			          << " challenges...\n";
		}

		// Run Sequential
		auto seqRes = BenchmarkSequentialFreivalds(matrixDim, c.targetBits, c.P, c.s, verbose && !emitCsv);

		// Run Batched
		auto batchRes = BenchmarkBatchedFreivalds(matrixDim, c.targetBits, c.P, c.s, verbose && !emitCsv);

		// Compute relative speedups (Formula: Sequential_Metric / Batched_Metric)
		double serverSpeedup = seqRes.serverTotalTime / batchRes.serverTotalTime;
		double wan50Speedup = seqRes.latencyWAN_50ms / batchRes.latencyWAN_50ms;
		double commReduction =
		    static_cast<double>(seqRes.totalBytes) / static_cast<double>(batchRes.totalBytes);

		batchRes.serverSpeedup = serverSpeedup;
		batchRes.wan50Speedup = wan50Speedup;
		batchRes.commReduction = commReduction;

		if (emitCsv) {
			PrintCsvResult(seqRes);
			PrintCsvResult(batchRes);
		} else {
			std::cout << "\n[" << c.note << " | P = " << c.P << " | s = " << c.s
			          << " | Soundness = " << std::fixed << std::setprecision(1)
			          << (c.s * std::log2(static_cast<double>(c.P))) << " bits]\n"
			          << "  Sequential: " << seqRes.roundTrips
			          << " Round Trips | Server: " << std::setprecision(4) << seqRes.serverTotalTime
			          << "s | WAN-50ms: " << seqRes.latencyWAN_50ms
			          << "s | Comm: " << (seqRes.totalBytes / 1024.0) << " KB\n"
			          << "  Batched:    " << batchRes.roundTrips
			          << " Round Trips | Server: " << batchRes.serverTotalTime
			          << "s | WAN-50ms: " << batchRes.latencyWAN_50ms
			          << "s | Comm: " << (batchRes.totalBytes / 1024.0) << " KB\n"
			          << "  Speedup:    " << std::setprecision(2) << serverSpeedup
			          << "x Server [Formula: T_server(Seq) / T_server(Batch)] | " << wan50Speedup
			          << "x WAN-50ms [Formula: T_wan(Seq) / T_wan(Batch)] | " << commReduction
			          << "x Less Comm [Formula: Bytes(Seq) / Bytes(Batch)]\n";
		}
	}

	return 0;
}
