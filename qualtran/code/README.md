# Source Code for "How to factor 2048 bit RSA integers with less than a million noisy qubits"

This repository contains code and assets created for the paper
"How to factor 2048 bit RSA integers with less than a million noisy qubits".


# Inventory:

- [assets/](assets/): Assets created for the paper.
- [src/](src/): Source code.
    - [src/facto/](src/facto/): Python code directly related to the paper (e.g. simulating and estimating the cost of quantum factoring).
        - [src/facto/algorithm/](src/facto/algorithm/): Python code for simulating the approximate modular exponentiation.
        - [src/facto/algorithm/](src/facto/operations/): Python code for testing and drawing some of the detailed mock-up figures in the appendix.
        - [src/facto/t_decomp/](src/facto/t_decomp/): C++ and Python code for generating/verifying decompositions of rotations into Clifford+T.
    - [src/scatter_script/](src/scatter_script/): Python library for writing and testing nearly-classical quantum computations.
    - [src/gen/](src/gen/): Python library for produce QEC circuits.
- [testdata/](testdata/): Data files for tests.
- [tools/](tools/): Command line tools.


## Usage Example - Regenerate Plots

```bash
# In a fresh python environment
pip install -r requirements.txt
```

```bash
# Perform a parameter grid scan and store results in a CSV file in the out/ directory.
./step1_collect
```

```bash
# Consume the grid scan data to produce plots.
# (Also produce various other plots.)
./step2_plot
```

## Usage Example - Simulate Algorithm

Write a .ini file describing the problem.
The repo includes several examples in the [`testdata/`](testdata/) directory.
For example, this is the contents of [`testdata/rsa100.ini'](testdata/rsa100.ini):

```ini
; RSA100 challenge number
; 330 bits (100 digits)
; p=37975227936943673922808872755445627854565536638199
; q=40094690950920881030683735292761468389214899724061
; source: https://en.wikipedia.org/wiki/RSA_numbers#RSA-100

modulus = 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654000350692006139
generator = 5
s = 6
num_shots = 7
num_input_qubits = 221

window1 = 5
window3a = 2
window3b = 3
window4 = 5
min_wraparound_gap = 28
len_accumulator = 28
mask_bits = 14
```

From the problem .ini file, the tables of discrete logs and other values used during the quantum computation can be precomputed:

```bash
./tools/factor_precompute \
    --problem_ini testdata/rsa100.ini \
    --out_dir out/rsa100
```

Then simulated runs can be performed (this is relatively slow):

```bash
./tools/factor_simulate_approx_modexp \
    --exec_config_dir out/rsa100
```

The simulation script choose several random cases and outputs the resulting approximate modular exponentiations:

```bash
generator: 0x5
modulus: 0x2c8d59af47c81ab3725b472be417e3bf7ab85439af726ed3dfdf66489d155dc0b771c7a50ef7c5e58fb
 deviation |     approx_result |                                                   exponent
  8.41e-05 |  0xf28f3f0 << 300 |  0x1fd5c1c74323f43dff089e023935349ecaaecd1d2f102c78fb1ee1d
  1.71e-05 | 0x2c3d443c << 300 |  0xef4e33ffe51f0393a3c86e0829fc5ba6793eeb2e16f5b87472bde4f
  3.34e-07 |  0x7eb436c << 300 | 0x19211e05db6ec4fec8902612010697f623d5d84fc7d686ca939f6a05
  1.53e-05 | 0x1e6c2920 << 300 | 0x1d848d3c0a9f688a20f6888f0b25fb7edc3128cec66e1bda0aacf68e
  3.63e-05 | 0x2a76f798 << 300 |  0xc400e9f40e3f462885893e2b5e90d884d25e913d1fe25b034236445
  8.17e-05 |  0x13037bc << 300 | 0x18fb950b10a9269c415a5d3747aa06b4747faf334b6d4dd6cfb63922
  4.53e-05 | 0x1e61c02c << 300 |  0xef40f1112ad96e6affec29c40cc61cc46cff98b64eca108606fe286
  6.67e-05 |  0xaa1a21c << 300 |  0x9d2ceff6d2d021b460a4c5dd998f3737d350738b2eb55c566b328c4
     4e-05 |  0x93c6b74 << 300 |  0x670a4f4a87fb79e12edebb4f54646ca2d8cd55e3f193e148a574583
  5.12e-05 | 0x143a5464 << 300 | 0x19102910eb945406a247be9a9c4338c0e710d24ab6f9c74a1f61dc8b
```

You can then verify that the displayed result matches the most significant bits of the exact computation:

```python
generator = 0x5
modulus = 0x2c8d59af47c81ab3725b472be417e3bf7ab85439af726ed3dfdf66489d155dc0b771c7a50ef7c5e58fb
exponent = 0x1fd5c1c74323f43dff089e023935349ecaaecd1d2f102c78fb1ee1d
print(hex(pow(generator, exponent, modulus)))
# prints            0xf27fe66106980301776678cfca21d392d86f7851e0e11d910c2ffdcd00dcd1baf2ea5cfce0c849e2d5
# which is close to 0xf28f3f0 << 300
```

## Usage Example - Verify Gate Decomposition

The paper includes tables specifying rotation decompositions into Clifford+T.
There is a tool included in the code for checking the infidelity of these decompositions:

```bash
./tools/compute_phase_gradient_gate_sequence_infidelity \
    --seq "+--++-++--+-+++-+----+HY" \
    --target_phase_gradient_qubit_index 3
```

Outputs:

```
5.812672333442671e-07
```
