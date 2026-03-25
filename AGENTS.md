# AGENTS.md

## Cursor Cloud specific instructions

### Overview

ntsc-rs is a Rust-based NTSC/VHS video effect emulator. The workspace contains multiple crates:

| Crate | Path | Description |
|---|---|---|
| `ntsc-rs` | `crates/ntscrs` | Core NTSC/VHS effect processing engine (library) |
| `ntsc-rs-gui` | `crates/gui` | Standalone GUI app, launcher, and CLI (`ntsc-rs-standalone`, `ntsc-rs-launcher`, `ntsc-rs-cli`) |
| `ntsc-rs-openfx-plugin` | `crates/openfx-plugin` | OpenFX plugin for video editors |
| `ntsc-rs-ae-plugin` | `crates/ae-plugin` | After Effects/Premiere plugin (Windows/macOS only, won't build functional code on Linux) |
| `xtask` | `xtask` | Build automation (e.g. `cargo xtask build-ofx-plugin`) |

### System dependencies (Ubuntu/Debian)

These must be installed before building:

```
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libatk1.0-dev libgtk-3-dev libssl-dev pkg-config libclang-dev \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

### Build, lint, and test

```bash
cargo build --workspace       # Build all crates
cargo clippy --workspace      # Lint
cargo test --workspace        # Run tests (29 unit tests in ntscrs crate)
```

### Running the GUI

```bash
cargo run --bin ntsc-rs-standalone   # Standalone GUI app
cargo run --bin ntsc-rs-cli          # CLI tool
```

The GUI requires a display (`DISPLAY` env var). In Cloud Agent VMs, `:1` is typically available.

### Important caveats

- **Rust edition 2024**: Several crates use `edition = "2024"`, requiring Rust 1.85+. The update script ensures the stable toolchain is set as default via `rustup default stable`.
- **Git submodules**: The OpenFX plugin depends on vendored OpenFX SDK headers at `crates/openfx-plugin/vendor/openfx`. Run `git submodule update --init --recursive` after cloning.
- **AE plugin on Linux**: The After Effects plugin crate compiles on Linux but only produces functional code on Windows/macOS (platform-gated via `cfg`). It will build but won't produce a usable plugin.
- **GStreamer runtime plugins**: The dev packages are needed for compilation. The runtime plugin packages (`gstreamer1.0-plugins-*`, `gstreamer1.0-libav`) are needed to actually open/process media files at runtime.
