# Compiler Pipeline in Aetherium

## Overview
The Aetherium compiler (`aethc`) is a multi-pass, heterogeneous system that transforms source code into optimized executables for diverse backends. It emphasizes formal verification, automatic differentiation, and hardware-specific lowering.

## Pipeline Stages
1. **Parse & CST**: Lex and parse to Concrete Syntax Tree. Handles mathematical syntax like `∇` and `⊗`.
2. **Meta-Type Inference**: Propagate metamorphic types (e.g., infer Differentiable from `∇` usage). Raise errors for mismatches.
3. **AETH-IR Generation**: Build a graph-based intermediate representation (IR) with hyper-ops (e.g., MatrixMult, Hadamard). Edges carry type, context, and location (@) info.
4. **Verification Pass (Prover)**:
   - Differentiability: Scan for non-diff ops (e.g., floor).
   - Reversibility: Ensure unitarity via symbolic simulation.
   - Use external provers (e.g., Coq plugins) for complex cases.
5. **Differentiation Pass**: Source-to-source AD for Differentiable graphs, generating adjoint nodes.
6. **Optimization & Fusion**: Fuse ops (e.g., Conv + ReLU), schedule data movement, apply quantum gate fusion.
7. **Backend Codegen**:
   - Classical → LLVM → x86/ARM.
   - Differentiable → NVPTX (CUDA) / TPU IR.
   - Reversible → QASM / QIR.
   - Distributed → MPI / NCCL.

## Implementation Notes
- **IR Design**: Hypergraph for higher-order relations; nodes tagged with meta-types.
- **Heterogeneity**: Use MLIR-inspired dialects for backend portability.
- **Error Integration**: Failures in verification trigger semantic errors (see error_philosophy.md).
- **Performance**: Lazy codegen for lazy evaluation in runtime.

Inspired by Mojo (Modular) and Julia's JIT, but with built-in quantum/probabilistic support.
