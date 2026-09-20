# onwemu

A Rust/WebAssembly, bring-your-own-ROM arcade for NES and original Game Boy games. ROM data is read with the browser File API, passed directly into WASM.

Both emulator cores are owned source in this repository: the clean-room `gbsml` Game Boy implementation and a safe-Rust NES core. NES cartridges support iNES mappers 0, 1, 2, 3, 4, and 7 (NROM, MMC1, UxROM, CNROM, MMC3, and AxROM). Game Boy cartridges support ROM-only, MBC1, MBC2, MBC3 with RTC, and MBC5, including versioned battery saves.

Frontends integrate through the public `core::Core` contract. It covers video formats, normalized digital/analog input, frame execution, and persistent data; a future PSP core can implement that contract and register with `core::create` without adding system-specific branches to each frontend.

## Run locally

Requirements: a recent Rust toolchain with `wasm32-unknown-unknown`, a WASM linker (`lld`), and Node 20+.

```sh
npm install
npm run dev
```
