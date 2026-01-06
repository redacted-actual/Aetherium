# Error Philosophy in Aetherium

## Overview
Aetherium's errors are not mere syntax checks but semantic critiques, akin to a mathematical peer review. The compiler acts as a prover, highlighting flaws in computational proofs (e.g., non-unitary ops in Reversible).

## Design Principles
- **Semantic Focus**: Prioritize mathematical soundness over syntax (e.g., "Discontinuity Error" for non-diff branches).
- **Constructive Feedback**: Errors include hints/notes (e.g., "Did you mean a sigmoid approximation?").
- **Compile-Time Dominance**: Catch bugs early via meta-type verification; runtime errors are minimal.
- **User Experience**: Readable, contextual messages with line/col refs.

## Example Errors
- **E0101: Metamorphic Type Violation**: Irreversible op in Reversible (e.g., Measure inside quantum block).
- **E0102: Discontinuity Error**: Non-differentiable construct in Differentiable.
- **E0201: Contextual Operator Misuse**: `~` outside Probabilistic.
- **E0300: Hardware/Type Mismatch**: Invalid op on pinned type (e.g., matmul on Qubit).
- **E0103: Metamorphic Inference Failed**: `∇` on non-Differentiable function.

## Implementation Notes
- **Error Rendering**: Use AST spans for precise locations; integrate with LSP for IDE hints.
- **Extensibility**: Allow custom meta-types to define new error classes.
- **Inspiration**: Borrow from Rust's helpful diagnostics, but elevate to proof-level critiques.

This philosophy turns errors into learning opportunities, reinforcing code-as-mathematics.
