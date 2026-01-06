# Metamorphic Type System in Aetherium

## Overview
The Metamorphic Type System is the cornerstone of Aetherium's design philosophy: "Computation as Proof. Code as Mathematics." Unlike traditional type systems that focus solely on data shapes (e.g., `Int`, `Float`), metamorphic types encode *computational properties* as first-class citizens. These properties act as effect systems, imposing semantic constraints that the compiler formally verifies.

This system unifies disparate computing paradigms by allowing types to "morph" based on context, ensuring correctness across classical, differentiable, reversible, probabilistic, and distributed computations.

## Key Concepts
- **Meta-Types as Effects**: Each meta-type (e.g., Differentiable, Reversible) is an effect that propagates through the type inference process. For example:
  - `Differentiable` requires all operations to be smooth and traceable, rejecting discontinuities like hard branches.
  - `Reversible` enforces linearity and unitarity, adhering to quantum no-cloning theorems.
- **Type Refinements**: Types can be refined with hardware or context qualifiers, e.g., `Tensor<F32> @ GPU : Differentiable`.
- **Inference and Verification**: The compiler infers meta-types from declarations and verifies them via a prover pass. Undecidable cases may require user annotations or restricted subsets.

## Formal Semantics
Consider a type `T<C>` where `C` is a context (meta-type). The typing judgment is extended as:
\[
\Gamma \vdash e : T<C>
\]
Where \(\Gamma\) includes context constraints. For Reversible:
- No destructive assignments: \(\Gamma \vdash x = y : \bot\) if not uncomputed.
For Differentiable:
- All paths must have defined gradients: Reject `if` on non-constant conditions unless approximated.

## Implementation Notes
- **IR Representation**: Use tagged nodes in AETH-IR to track meta-types.
- **Challenges**: Scalability of proofs—integrate SMT solvers like Z3 for verification.
- **Extensions**: Future meta-types could include Secure (for cryptography) or EnergyEfficient (for edge devices).

This system draws from effect handlers (e.g., Koka) and dependent types (e.g., Idris), but ties them to hardware for practical unification.
