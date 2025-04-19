# Contributing to MAGNUS

Thank you for your interest in contributing to the MAGNUS project! This document outlines the process for contributing and the standards we follow.

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