#pragma once

#include <openfhe.h>

namespace fhe
{

using lbcrypto::Ciphertext;
using lbcrypto::CryptoContext;
using lbcrypto::DCRTPoly;

struct FHEParameters {
	std::int64_t P;
	std::int64_t N;
	std::int64_t base;
	std::int64_t slot_count;
	std::size_t d;
	std::size_t n;
};

FHEParameters SetFHEParameters(std::int64_t target_base);
void PrintInfo(CryptoContext<DCRTPoly> &cc);

} // namespace fhe
