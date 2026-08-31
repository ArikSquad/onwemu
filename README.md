# onwemu

A Rust/WebAssembly, bring-your-own-ROM arcade for NES and original Game Boy games. ROM data is read with the browser File API, passed directly into WASM, and is never uploaded or persisted. No copyrighted game data is included in the site or build output.

Both emulator cores are owned source in this repository: the clean-room `gbsml` Game Boy implementation and a safe-Rust port of `nesma`'s NROM architecture. No external emulator library is used.

## Run locally

Requirements: a recent Rust toolchain with `wasm32-unknown-unknown`, a WASM linker (`lld`), and Node 20+.

```sh
npm install
npm run dev
```

Open the printed URL, choose NES or Game Boy, then select a ROM you legally own. Controls are arrows, `Z` (A), `X` (B), Enter (Start), and Shift (Select).

## Test

Fast, ROM-free unit tests run everywhere:

```sh
cargo test --all-targets
cargo check --features web
```

The conformance script clones redistributable emulator test ROM repositories into a temporary directory; it never adds them to the app or repository:

```sh
./scripts/test-roms.sh
```

You can also run a legally obtained Blargg-style serial test directly:

```sh
cargo run --release --bin gbsml-headless -- path/to/test.gb
```

CI tests Rust on Linux, macOS, and Windows, builds the production WASM bundle, and runs the public Game Boy CPU ROM suite on Linux. `tests/roms/`, `*.gb`, and `*.nes` remain ignored so accidental ROM commits are difficult.

## Vercel

Import the repository in Vercel with no framework preset. `vercel.json` installs the Amazon Linux Rust/WASM packages, runs `npm run build`, and publishes `dist/`. The deployment contains only HTML/CSS/JS and the WASM module—there is no server function capable of receiving ROMs.

## Scope

- NES: iNES mapper 0 / NROM, matching `nesma`'s current scope (including Super Mario Bros.).
- Game Boy: original monochrome DMG cartridges supported by `gbsml`; Game Boy Color-only ROMs are intentionally rejected/not advertised.
- Save states, battery saves, audio, touch controls, and multiplayer are future work.

Commercial ROM images are copyrighted. Dump cartridges you own and follow the laws where you live.
