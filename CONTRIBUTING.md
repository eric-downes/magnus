# Contributing to MAGNUS

Thank you for your interest in contributing to the MAGNUS project! This document outlines the process for contributing and the standards we follow.

## Code Standards

### Constants Management (CRITICAL)

**All numeric constants MUST be defined in `src/constants.rs`**

This is a strict requirement to prevent bugs. The bitonic sort bug that required significant debugging was caused by hardcoded constants. To prevent similar issues:

#### Rules for Constants

1. **Never hardcode numeric literals** in implementation code
2. **Always define constants** in `src/constants.rs` 
3. **Use descriptive ALL_CAPS names** that clearly indicate purpose
4. **Document each constant** with a comment
5. **Group related constants** under section headers

#### Examples

❌ **BAD**: Hardcoded constant
```rust
if elements.len() >= 10000 {
    use_gpu_acceleration();
}
```

✅ **GOOD**: Named constant
```rust
// In src/constants.rs
/// Threshold for using Metal GPU acceleration (number of elements)
pub const METAL_GPU_THRESHOLD: usize = 10_000;

// In implementation file
use crate::constants::METAL_GPU_THRESHOLD;
if elements.len() >= METAL_GPU_THRESHOLD {
    use_gpu_acceleration();
}
```

#### When to Add Constants

Add to `src/constants.rs` when you have:
- Algorithm selection thresholds
- Buffer sizes or capacity limits
- Architecture-specific parameters
- Floating-point tolerances
- Display/debug limits
- Any "magic number" affecting behavior

## Development Process

1. **Fork the Repository**: Start by forking the repository and cloning it locally.

2. **Create a Feature Branch**: Create a branch for your work from the `master` branch:
   ```
   git checkout -b feature/your-feature-name
   ```

3. **Code Standards**:
   - Follow the Rust style guidelines using `rustfmt`
   - Write comprehensive tests for new features
   - Document your code with appropriate comments and Rustdoc
   - Keep commits focused and with clear commit messages

4. **Testing**:
   - Run tests locally with `cargo test`
   - Run benchmarks with `cargo bench`
   - Ensure no new clippy warnings with `cargo clippy`

5. **Pull Request Process**:
   - Update the README and documentation if needed
   - Fill out the PR template completely
   - Link to any related issues
   - Respond to code review feedback
   - All tests must pass in CI before merging

## Branch Protection Rules

The `master` branch is protected with the following rules:

- Pull requests require at least one approval from code owners
- All CI checks must pass before merging
- Branches must be up to date with `master` before merging
- No force pushes or deletion of the `master` branch

## Continuous Integration

Our CI pipeline runs the following checks:

1. **Build and Test**: Ensures the project builds and all tests pass
2. **Linting**: Runs `rustfmt` and `clippy` to ensure code quality
3. **Benchmarks**: For PRs, runs some basic benchmarks to detect performance regressions

## Getting Help

If you need help, please:
1. Check the documentation
2. Open an issue for substantial discussions
3. Reach out to code owners for guidance

Thank you for contributing to MAGNUS!