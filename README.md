# CATCH

Commit-And-Then-CHallenge verification for FHE matrix multiplication and attacks against DataSeal.

## Requirements
* [CMake](https://cmake.org/) >= 3.16
* [OpenFHE](https://github.com/openfheorg/openfhe-development) version 1.6.1
* C++ compiler

## Tested Environment
* OS: WSL 3 Arch Linux
* Compiler: Clang/Clang++ 23.1.5

## Compilation

First, make sure OpenFHE is installed. In the root of the project run:
```bash
cmake -B build
cmake --build build
```

## Usage

The examples can be run as follows after building
```bash
./scripts/benchmark_equal_dims.py               # Table 2
./scripts/benchmark_plaintext_modulus_blowup.py # Table 3
./scripts/benchmark_attacks_fhe.py              # Table 4
./scripts/benchmark_catch_modulus_sweep.py      # Table 5
```

## License

TBD
