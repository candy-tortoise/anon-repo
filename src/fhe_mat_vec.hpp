#pragma once

#include "fhe_context.hpp"

namespace fhe
{

std::vector<std::int32_t> ComputeBSGSKeys(std::size_t d);
std::vector<std::int64_t> PlaintextRotate(const std::vector<std::int64_t> &vec, int shift);

// ---------------------------------------------------------
// FAST BSGS (Hardware-wrapped)
// ---------------------------------------------------------
Ciphertext<DCRTPoly> FastRotate(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_x, int shift);
Ciphertext<DCRTPoly> FastBabyStepGiantStep(CryptoContext<DCRTPoly> &cc,
                                           const std::vector<std::vector<std::int64_t>> &A_diagonals,
                                           Ciphertext<DCRTPoly> &encrypted_x, int d);

// ---------------------------------------------------------
// ROBUST BSGS (Explicitly-wrapped)
// ---------------------------------------------------------
Ciphertext<DCRTPoly> RobustRotate(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_x, int shift,
                                  std::size_t d);
Ciphertext<DCRTPoly> RobustBabyStepGiantStep(CryptoContext<DCRTPoly> &cc,
                                             const std::vector<std::vector<std::int64_t>> &A_diagonals,
                                             Ciphertext<DCRTPoly> &encrypted_x, int d);

// ---------------------------------------------------------
// OPTIMAL ROUTER
// ---------------------------------------------------------
Ciphertext<DCRTPoly> OptimalMatrixVectorMult(CryptoContext<DCRTPoly> &cc,
                                             const std::vector<std::vector<std::int64_t>> &A_diagonals,
                                             Ciphertext<DCRTPoly> &encrypted_x, int d, std::size_t N);

} // namespace fhe
