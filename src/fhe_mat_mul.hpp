#pragma once

#include "fhe_context.hpp"
#include "linear_algebra.hpp"

#include <cstdint>
#include <openfhe.h>
#include <vector>

namespace fhe
{

std::vector<std::int32_t> ComputeRequiredRotationIndices(std::size_t d);
Ciphertext<DCRTPoly> RotateCiphertextA(int s, CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                       std::size_t d);
Ciphertext<DCRTPoly> ChangeCipherToMatrixA(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                           std::size_t d);
std::vector<Ciphertext<DCRTPoly>> FlattenAndEncryptMatrix(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
                                                          lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys,
                                                          LinearAlgebra::Matrix &m, bool diagonalize);
LinearAlgebra::Matrix Decrypt(lbcrypto::CryptoContext<lbcrypto::DCRTPoly> &cc,
                              lbcrypto::KeyPair<lbcrypto::DCRTPoly> &keys, lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &m,
                              std::size_t n);

// ---------------------------------------------------------
// FAST ALGORITHMS (Requires d^2 == N/2)
// ---------------------------------------------------------
Ciphertext<DCRTPoly> FastRotateCiphertextB(int i, CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                           std::size_t d);
Ciphertext<DCRTPoly> FastChangeCipherToMatrixB(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                               std::size_t d);
Ciphertext<DCRTPoly> FastJiangMatrixMultiplication(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_A,
                                                   Ciphertext<DCRTPoly> &encrypted_B, std::size_t d);

// ---------------------------------------------------------
// ROBUST ALGORITHMS (Works for any N/2 >= d^2)
// ---------------------------------------------------------
Ciphertext<DCRTPoly> RobustRotateCiphertextB(int i, CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                             std::size_t d);
Ciphertext<DCRTPoly> RobustChangeCipherToMatrixB(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &cipher_matrix,
                                                 std::size_t d);
Ciphertext<DCRTPoly> RobustJiangMatrixMultiplication(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_A,
                                                     Ciphertext<DCRTPoly> &encrypted_B, std::size_t d);

// ---------------------------------------------------------
// OPTIMAL ALGORITHM ROUTER
// ---------------------------------------------------------
Ciphertext<DCRTPoly> OptimalJiangMatrixMultiplication(CryptoContext<DCRTPoly> &cc, Ciphertext<DCRTPoly> &encrypted_A,
                                                      Ciphertext<DCRTPoly> &encrypted_B, std::size_t d, std::size_t N);

// ---------------------------------------------------------
// RECTANGULAR MATRIX MULTIPLICATION (Jiang et al. CCS'18 Algorithm 3)
// ---------------------------------------------------------
Ciphertext<DCRTPoly> RectangularJiangMatrixMultiplication(CryptoContext<DCRTPoly> &cc,
                                                          Ciphertext<DCRTPoly> &encrypted_A_tiled,
                                                          Ciphertext<DCRTPoly> &encrypted_B, std::size_t d,
                                                          std::size_t ell, std::size_t N = 0);

} // namespace fhe
