#include "fhe_context.hpp"
#include "fhe_mat_mul.hpp"
#include "linear_algebra.hpp"
#include "openfhe.h"

// #include "ciphertext-ser.h"
#include "cryptocontext-ser.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace lbcrypto;
using namespace std::chrono;

// Primality test
bool IsPrime(uint64_t n)
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

// Find next prime t = k * (2 * N) + 1 >= min_val
uint64_t FindNextPrimeModulus(uint64_t min_val, uint32_t N)
{
	uint64_t step = 2 * N;
	uint64_t k = (min_val / step);
	if (k * step + 1 < min_val) {
		k++;
	}
	while (true) {
		uint64_t cand = k * step + 1;
		if (cand < min_val) {
			k++;
			continue;
		}
		if (IsPrime(cand)) {
			return cand;
		}
		k++;
	}
}

// Record for each tested bit-width configuration
struct ModulusSweepResult {
	int targetBits;
	std::string t_str;
	double actualBits;
	std::string status; // "INSTANTIATED", "REFUSED_LIBRARY_LIMIT", "API_TYPE_LIMIT"
	std::string failureReason;

	// FHE Parameter attributes
	uint32_t ringDim;
	uint32_t log2Q;
	std::size_t numLimbs;
	double ciphertextSizeKB;

	// Latency (seconds)
	double keyGenTime_s;
	double encTime_s;
	double matmulTime_s;
	double decTime_s;

	// Security & Correctness
	double additiveSoundnessBits;  // log2(t)
	double linearMapSoundnessBits; // Always 0 bits for DataSeal!
	bool decryptCorrect;
};

// Measure exact serialized on-the-wire byte size of a ciphertext
std::size_t MeasureCiphertextSize(const Ciphertext<DCRTPoly> &ct)
{
	std::stringstream ss;
	Serial::Serialize(ct, ss, SerType::BINARY);
	return ss.str().size();
}

ModulusSweepResult TestBitWidth(int targetBits, uint32_t N, std::size_t matrixDim, bool verbose = false)
{
	ModulusSweepResult res;
	res.targetBits = targetBits;
	res.ringDim = N;
	res.linearMapSoundnessBits = 0.0; // 0 bits against D in DataSeal
	res.ciphertextSizeKB = 0.0;
	res.keyGenTime_s = 0.0;
	res.encTime_s = 0.0;
	res.matmulTime_s = 0.0;
	res.decTime_s = 0.0;
	res.decryptCorrect = false;

	if (targetBits >= 64) {
		res.status = "API_TYPE_LIMIT";
		res.t_str = "N/A (> 64-bit integer)";
		res.actualBits = targetBits;
		res.additiveSoundnessBits = targetBits;
		res.failureReason = "Plaintext modulus exceeds uint64_t NativeInteger. OpenFHE SetPlaintextModulus() "
		                    "takes uint64_t, physically preventing instantiation of t >= 2^64.";
		if (verbose) {
			std::cout << "[Target " << targetBits
			          << " bits] API Limit: t >= 2^64 cannot be represented as uint64_t.\n";
		}
		return res;
	}

	uint64_t min_v = (targetBits <= 63) ? (1ULL << (targetBits - 1)) : 0;
	uint64_t p = FindNextPrimeModulus(min_v, N);
	res.t_str = std::to_string(p);
	res.actualBits = std::log2(static_cast<double>(p));
	res.additiveSoundnessBits = res.actualBits;

	CCParams<CryptoContextBGVRNS> parameters;
	parameters.SetMultiplicativeDepth(3);
	parameters.SetRingDim(N);
	parameters.SetSecurityLevel(HEStd_128_classic);
	parameters.SetPlaintextModulus(p);

	try {
		auto t_kg_start = high_resolution_clock::now();
		CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
		cc->Enable(PKE);
		cc->Enable(KEYSWITCH);
		cc->Enable(LEVELEDSHE);
		cc->Enable(ADVANCEDSHE);

		KeyPair<DCRTPoly> keys = cc->KeyGen();
		cc->EvalMultKeysGen(keys.secretKey);
		cc->EvalRotateKeyGen(keys.secretKey, fhe::ComputeRequiredRotationIndices(matrixDim));
		auto t_kg_stop = high_resolution_clock::now();

		res.status = "INSTANTIATED";
		res.failureReason = "None";
		res.ringDim = cc->GetRingDimension();
		auto modQ = cc->GetModulus();
		res.log2Q = modQ.GetLengthForBase(2);
		res.keyGenTime_s = duration<double>(t_kg_stop - t_kg_start).count();

		// Generate test matrices
		/*
		auto A = LinearAlgebra::GenerateSquareMatrixWithRandomValues(matrixDim);
		auto B = LinearAlgebra::GenerateSquareMatrixWithRandomValues(matrixDim);
		auto expectedC = LinearAlgebra::MultiplyMatrixes(A, B);

		for (auto &row : A) {
		        for (auto &v : row) {
		                v %= p;
		                if (v < 0) {
		                        v += p;
		                }
		        }
		}
		for (auto &row : B) {
		        for (auto &v : row) {
		                v %= p;
		                if (v < 0) {
		                        v += p;
		                }
		        }
		}
		for (auto &row : expectedC) {
		        for (auto &v : row) {
		                v %= p;
		                if (v < 0) {
		                        v += p;
		                }
		        }
		}

		auto flatA = LinearAlgebra::FlattenRowMajor(A);
		auto flatB = LinearAlgebra::FlattenRowMajor(B);

		// Encrypt
		auto t_enc_start = high_resolution_clock::now();
		auto ctA = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatA));
		auto ctB = cc->Encrypt(keys.publicKey, cc->MakePackedPlaintext(flatB));
		auto t_enc_stop = high_resolution_clock::now();
		res.encTime_s = duration<double>(t_enc_stop - t_enc_start).count();

		res.ciphertextSizeKB = MeasureCiphertextSize(ctA) / 1024.0;

		// Server MatMul
		auto t_mm_start = high_resolution_clock::now();
		auto ctC = fhe::OptimalJiangMatrixMultiplication(cc, ctA, ctB, matrixDim, N);
		auto t_mm_stop = high_resolution_clock::now();
		res.matmulTime_s = duration<double>(t_mm_stop - t_mm_start).count();

		// Decrypt & verify
		auto t_dec_start = high_resolution_clock::now();
		Plaintext ptC;
		cc->Decrypt(keys.secretKey, ctC, &ptC);
		ptC->SetLength(matrixDim * matrixDim);
		auto packedC = ptC->GetPackedValue();
		auto t_dec_stop = high_resolution_clock::now();
		res.decTime_s = duration<double>(t_dec_stop - t_dec_start).count();

		bool match = true;
		for (std::size_t i = 0; i < matrixDim; ++i) {
		        for (std::size_t j = 0; j < matrixDim; ++j) {
		                int64_t actual = packedC[i * matrixDim + j] % p;
		                if (actual < 0) {
		                        actual += p;
		                }
		                int64_t exp = expectedC[i][j] % p;
		                if (exp < 0) {
		                        exp += p;
		                }
		                if (actual != exp) {
		                        match = false;
		                        break;
		                }
		        }
		        if (!match) {
		                break;
		        }
		}
		res.decryptCorrect = match;

		if (verbose) {
		        std::cout << "[Target " << targetBits << " bits | t=" << p
		                  << "] INSTANTIATED | log2(Q)=" << res.log2Q << " | CT Size=" << res.ciphertextSizeKB
		                  << " KB | MatMul=" << res.matmulTime_s << "s | Correct=" << (match ? "YES" : "NO")
		                  << "\n";
		}
		*/
	} catch (const std::exception &e) {
		res.status = "REFUSED_LIBRARY_LIMIT";
		res.failureReason = e.what();
		if (verbose) {
			std::cout << "[Target " << targetBits << " bits | t=" << p << "] REFUSED: " << e.what() << "\n";
		}
	}

	return res;
}

// Print results in CSV format
void PrintCsvHeader()
{
	std::cout << "targetBits,t_str,actualBits,status,failureReason,ringDim,log2Q,ciphertextSizeKB,"
	          << "keyGenTime_s,encTime_s,matmulTime_s,decTime_s,"
	          << "additiveSoundnessBits,linearMapSoundnessBits,decryptCorrect\n";
}

void PrintCsvResult(const ModulusSweepResult &r)
{
	std::string cleanReason = r.failureReason;
	for (char &c : cleanReason) {
		if (c == '"' || c == '\n' || c == '\r' || c == ',') {
			c = ' ';
		}
	}
	std::cout << r.targetBits << ",\"" << r.t_str << "\"," << r.actualBits << ",\"" << r.status << "\",\""
	          << cleanReason << "\"," << r.ringDim << "," << r.log2Q << "," << r.ciphertextSizeKB << ","
	          << r.keyGenTime_s << "," << r.encTime_s << "," << r.matmulTime_s << "," << r.decTime_s << ","
	          << r.additiveSoundnessBits << "," << r.linearMapSoundnessBits << ","
	          << (r.decryptCorrect ? "true" : "false") << "\n";
}

int main(int argc, char **argv)
{
	uint32_t N = 16384 * 16;
	std::size_t matrixDim = 16;
	std::string format = "text"; // "text" or "csv"
	std::vector<int> bit_targets = {16, 20, 24, 28, 30, 32, 34, 36, 38, 40, 48, 60, 64, 80, 128};

	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--csv" || arg == "--json") {
			format = "csv";
		} else if (arg == "--dim" && i + 1 < argc) {
			matrixDim = std::stoul(argv[++i]);
		} else if (arg == "--ring_dim" && i + 1 < argc) {
			N = std::stoul(argv[++i]);
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
			std::cout << "Usage: bench_plaintext_modulus_sweep [options]\n"
			          << "Options:\n"
			          << "  --dim <N>        Matrix dimension (e.g. 16) [default: 16]\n"
			          << "  --ring_dim <N>   FHE Ring dimension (e.g. 16384) [default: 16384]\n"
			          << "  --bits <list>    Comma-separated bit sizes (e.g. 16,20,32,40,64,128)\n"
			          << "  --csv            Output machine-readable CSV\n";
			return 0;
		}
	}

	std::vector<ModulusSweepResult> results;
	for (int bits : bit_targets) {
		results.push_back(TestBitWidth(bits, N, matrixDim, format == "text"));
	}

	if (format == "csv") {
		PrintCsvHeader();
		for (const auto &r : results) {
			PrintCsvResult(r);
		}
	} else {
		std::cout << "\n"
		          << "========================================================================================="
		             "==========================\n"
		          << "PLAINTEXT MODULUS (t) SWEEP: PARAMETER BLOWUP & LIMITS ANALYSIS\n"
		          << "========================================================================================="
		             "==========================\n";
		std::cout << std::left << std::setw(8) << "Bits" << std::setw(22) << "Plaintext t" << std::setw(24)
		          << "Status" << std::setw(12) << "log2(Q)" << std::setw(14) << "CT Size (KB)" << std::setw(14)
		          << "KeyGen (s)" << std::setw(14) << "MatMul (s)" << std::setw(12) << "Add. Sound"
		          << std::setw(10) << "D Sound" << "\n";
		std::cout << std::string(129, '-') << "\n";

		for (const auto &r : results) {
			std::string q_str = (r.status == "INSTANTIATED") ? std::to_string(r.log2Q) + " bits" : "---";
			std::string ct_str = (r.status == "INSTANTIATED")
			                         ? std::to_string(static_cast<int>(r.ciphertextSizeKB)) + " KB"
			                         : "---";
			std::string kg_str =
			    (r.status == "INSTANTIATED") ? std::to_string(r.keyGenTime_s).substr(0, 6) + "s" : "---";
			std::string mm_str =
			    (r.status == "INSTANTIATED") ? std::to_string(r.matmulTime_s).substr(0, 6) + "s" : "---";
			std::string add_sound = std::to_string(static_cast<int>(r.additiveSoundnessBits)) + " bits";

			std::cout << std::left << std::setw(8) << r.targetBits << std::setw(22)
			          << (r.t_str.length() > 20 ? r.t_str.substr(0, 18) + ".." : r.t_str) << std::setw(24)
			          << r.status << std::setw(12) << q_str << std::setw(14) << ct_str << std::setw(14)
			          << kg_str << std::setw(14) << mm_str << std::setw(12) << add_sound << std::setw(10)
			          << "0 bits" << "\n";
		}
		std::cout << "========================================================================================="
		             "==========================\n";
	}

	return 0;
}
