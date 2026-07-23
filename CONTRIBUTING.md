# Contributing to rustcpd

Thanks for your interest in improving rustcpd. Bug reports, feature
requests, and pull requests are all welcome.

## Development setup

The project is a Cargo workspace: the Rust core lives in `rustcpd/`, the
Python bindings (PyO3 + maturin) in `python/`.

```bash
# Rust core
cargo test --release                                # core tests
cargo test --release --features completion          # + shape completion
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all -- --check

# Python bindings
cd python
pip install maturin numpy pytest
maturin develop --release                           # build + install into the current env
python -m pytest tests -q
```

## Before opening a pull request

- `cargo fmt --all` and `cargo clippy --workspace --all-targets` are clean.
- New behavior has tests (Rust in `rustcpd/tests/`, Python in
  `python/tests/`). The numerical core is expected to stay deterministic —
  parallel and serial execution must produce bitwise-identical results.
- Public API changes are noted in `CHANGELOG.md` under an `Unreleased`
  heading, and breaking changes are called out explicitly (the project
  follows [Semantic Versioning](https://semver.org)).
- Parameter or algorithm changes that affect results are validated against
  a reference where one exists (see `benchmarks/`).

## Releasing (maintainers)

Version is set once in `rustcpd/Cargo.toml` and `python/Cargo.toml`; the
Python package reads it dynamically. Tagging a commit `vX.Y.Z` triggers the
`Wheels` workflow, which builds and tests wheels for Linux (x86_64 +
aarch64), macOS (universal2), and Windows (x64), then publishes them to
PyPI via trusted publishing.

## License

By contributing you agree that your contributions are licensed under the
project's [BSD 2-Clause License](LICENSE).
