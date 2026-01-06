# aeth.toml Manifest in Aetherium

## Overview
`aeth.toml` is Aetherium's declarative package manifest, extending Cargo's model with hardware-aware dependencies and backend configurations. It enables "morphing" builds that adapt to available substrates.

## Structure
- **[package]**: Name, version, authors, description.
- **[dependencies]**: Aetherium libs (e.g., `aeth-linalg = "1.5.0"`).
- **[backends]**: Declares supported hardware (e.g., `provides = ["@tensor_cores", "@qpu_fabric"]`).
- **[target.*]**: Hierarchical deps per backend/provider (e.g., `[target.@qpu_fabric.ibm]` with SDK versions).
- **[profile.*]**: Optimization flags, including hardware-specific (e.g., `fast-math = true` for GPU).

## Semantics
- **Build Process**: `aet build` scans system, resolves deps (e.g., fallback to simulator if no QPU), links SDKs.
- **Mixed-Language**: Supports Python/C interop via `provider = "pip"`.
- **Extensibility**: Env vars for API keys; profiles for dev/release.

## Example
```toml
[package]
name = "QuantumLeap"
version = "0.2.0"

[dependencies]
aeth-prob = "0.8.1"

[backends]
provides = ["@qpu_fabric"]

[target.@qpu_fabric.virtual]
lib = { name = "aeth-qsim", version = "1.0" }

[profile.release.@qpu_fabric]
fidelity-optimization = true
