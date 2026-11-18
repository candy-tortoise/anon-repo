#include "fhe_context.hpp"

#include "linear_algebra.hpp"

namespace fhe
{

FHEParameters SetFHEParameters(std::int64_t target_base)
{
	FHEParameters params;

	params.base = target_base;
	params.d = target_base;
	params.n = target_base;

	std::int64_t required_slots = params.base * params.base;
	params.N = 2;
	while (params.N < 2 * required_slots) {
		params.N *= 2;
	}

	params.slot_count = params.N / 2;

	std::int64_t k = 1;
	while (true) {
		params.P = k * 2 * params.N + 1;
		if (LinearAlgebra::IsPrime(params.P)) {
			break;
		}
		k++;
	}

	return params;
}

void PrintInfo(CryptoContext<DCRTPoly> &cc)
{
	const auto cp = std::dynamic_pointer_cast<lbcrypto::CryptoParametersRNS>(cc->GetCryptoParameters());

	if (!cp) {
		throw std::runtime_error("Expected an RNS crypto context");
	}

	std::cout << "Ring dimension: " << cc->GetRingDimension() << '\n';
	std::cout << "Plaintext modulus: " << cp->GetPlaintextModulus() << '\n';
	std::cout << "Scaling technique: " << cp->GetScalingTechnique() << '\n';
	std::cout << "Key-switching technique: " << cp->GetKeySwitchTechnique() << '\n';

	// Initial ciphertext modulus Q = q[0] * ... * q[L].
	const auto &qParams = cp->GetElementParams()->GetParams();
	std::cout << "Ciphertext modulus chain, in stored order:\n";
	for (size_t i = 0; i < qParams.size(); ++i) {
		const auto &qi = qParams[i]->GetModulus();
		std::cout << "  q[" << i << "] = " << qi << " (" << qi.GetMSB() << " bits)\n";
	}

	if (cp->GetKeySwitchTechnique() == lbcrypto::BV) {
		std::cout << "BV digit size: " << cp->GetDigitSize() << '\n';
	} else if (cp->GetKeySwitchTechnique() == lbcrypto::HYBRID) {
		std::cout << "HYBRID Q partitions: " << cp->GetNumberOfQPartitions() << '\n';
		std::cout << "HYBRID towers per partition: " << cp->GetNumPerPartQ() << '\n';

		// Auxiliary key-switching basis P, reported separately from Q.
		const auto pParams = cp->GetParamsP();
		if (pParams) {
			for (size_t i = 0; i < pParams->GetParams().size(); ++i) {
				const auto &pi = pParams->GetParams()[i]->GetModulus();
				std::cout << "  p[" << i << "] = " << pi << " (" << pi.GetMSB() << " bits)\n";
			}
		}
	}
}

} // namespace fhe
