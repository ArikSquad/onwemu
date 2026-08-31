# onwemu

A Rust/WebAssembly, bring-your-own-ROM arcade for NES and original Game Boy games. ROM data is read with the browser File API, passed directly into WASM.

Both emulator cores are owned source in this repository: the clean-room `gbsml` Game Boy implementation and a safe-Rust port of `nesma`'s NROM architecture. 

## Run locally

Requirements: a recent Rust toolchain with `wasm32-unknown-unknown`, a WASM linker (`lld`), and Node 20+.

```sh
npm install
npm run dev
```

