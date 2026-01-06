# Hardware Mapping in Aetherium

## Overview
Aetherium's hardware mapping enables seamless execution across heterogeneous substrates (CPU, GPU, TPU, QPU) via the `@` pin operator and topology-aware optimization. It abstracts devices while enforcing locality for performance and correctness.

## Key Mechanisms
- **@ Operator**: Pins data/compute to labels like `@host_mem`, `@tensor_cores`, `@qpu_fabric`. Creates refined types (e.g., `Tensor @ GPU`).
- **Topology Awareness**: Runtime queries connections (PCIe, NVLink) via `std::system::get_topology()` to minimize transfers.
- **Data Movement**: Implicit/explicit (`.to(@device)`); compiler verifies no races with borrow-checker rules.
- **Backend Integration**: Codegen targets specific IRs; aeth.toml handles SDK deps (e.g., CUDA for `@tensor_cores.nvidia`).

## Semantics
- Mismatched pins raise `E_Locality_Mismatch` at compile-time.
- In Distributed, use DMA channels for zero-copy across nodes.
- Quantum: `@qpu_fabric` ensures linear resource handling.

## Implementation Notes
- **Runtime Support**: Unified Memory Manager handles paging/sharding.
- **Challenges**: Vendor differences (e.g., IBM vs. Google QPUs)—use provider-agnostic abstractions in aeth.toml.
- **Optimizations**: Schedule based on topology graph; fuse kernels to reduce transfers.

Draws from SYCL/Halide for heterogeneity, but with metamorphic type integration for safety.
