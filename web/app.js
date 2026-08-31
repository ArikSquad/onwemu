import init, { WebEmulator } from './pkg/onwemu.js';

const SYSTEMS = {
  nes: { extension: '.nes' },
  gb: { extension: '.gb' },
};

const KEY_MAP = {
  arrowup: 'up',
  arrowdown: 'down',
  arrowleft: 'left',
  arrowright: 'right',
  z: 'a',
  x: 'b',
  enter: 'start',
  shift: 'select',
};

const ui = {
  change: document.getElementById('change'),
  drop: document.getElementById('drop'),
  error: document.getElementById('error'),
  filename: document.getElementById('filename'),
  picker: document.getElementById('picker'),
  player: document.getElementById('player'),
  rom: document.getElementById('rom'),
  screen: document.getElementById('screen'),
  systems: document.querySelectorAll('.system'),
};

const screenContext = ui.screen.getContext('2d');
if (!screenContext) {
  throw new Error('Unable to create a canvas rendering context');
}

const state = {
  animationId: 0,
  emulator: null,
  height: 0,
  system: 'nes',
  wasmReady: null,
  width: 0,
};

function setError(message = '') {
  ui.error.textContent = message;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function stopRendering() {
  cancelAnimationFrame(state.animationId);
  state.animationId = 0;
}

function disposeEmulator() {
  stopRendering();
  state.emulator?.free();
  state.emulator = null;
  state.width = 0;
  state.height = 0;
}

function showPicker(message = '') {
  disposeEmulator();
  ui.player.hidden = true;
  ui.picker.hidden = false;
  ui.rom.value = '';
  setError(message);
}

function selectSystem(system) {
  if (!SYSTEMS[system]) return;

  state.system = system;
  ui.systems.forEach((button) => {
    const selected = button.dataset.system === system;
    button.classList.toggle('active', selected);
    button.setAttribute('aria-checked', selected);
  });
  ui.rom.accept = SYSTEMS[system].extension;
  setError();
}

function loadWasm() {
  state.wasmReady ??= init();
  return state.wasmReady;
}

function startEmulator(emulator, filename) {
  disposeEmulator();
  state.emulator = emulator;
  state.width = emulator.width();
  state.height = emulator.height();

  ui.screen.width = state.width;
  ui.screen.height = state.height;
  ui.screen.style.aspectRatio = `${state.width} / ${state.height}`;
  ui.filename.textContent = filename;
  ui.picker.hidden = true;
  ui.player.hidden = false;

  renderFrame();
}

function renderFrame() {
  if (!state.emulator) return;

  try {
    const pixels = state.emulator.run_frame();
    const expectedLength = state.width * state.height * 4;
    if (pixels.length !== expectedLength) {
      throw new Error(`Invalid frame size: expected ${expectedLength} bytes, got ${pixels.length}`);
    }

    const image = new ImageData(
      new Uint8ClampedArray(pixels),
      state.width,
      state.height,
    );
    screenContext.putImageData(image, 0, 0);
    state.animationId = requestAnimationFrame(renderFrame);
  } catch (error) {
    showPicker(errorMessage(error));
  }
}

async function loadRom(file) {
  if (!file) return;

  const { extension } = SYSTEMS[state.system];
  setError();
  if (!file.name.toLowerCase().endsWith(extension)) {
    setError(`That does not look like a ${extension} file.`);
    return;
  }

  try {
    await loadWasm();
    const rom = new Uint8Array(await file.arrayBuffer());
    const emulator = new WebEmulator(state.system, rom);
    startEmulator(emulator, file.name);
  } catch (error) {
    showPicker(errorMessage(error));
  }
}

function handleKey(event, down) {
  const button = KEY_MAP[event.key.toLowerCase()];
  if (!state.emulator || !button) return;

  event.preventDefault();
  state.emulator.set_button(button, down);
}

ui.systems.forEach((button) => {
  button.addEventListener('click', () => selectSystem(button.dataset.system));
});

ui.rom.addEventListener('change', (event) => loadRom(event.target.files[0]));
ui.change.addEventListener('click', () => showPicker());

['dragenter', 'dragover'].forEach((eventName) => {
  ui.drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    ui.drop.classList.add('over');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  ui.drop.addEventListener(eventName, (event) => {
    event.preventDefault();
    ui.drop.classList.remove('over');
  });
});

ui.drop.addEventListener('drop', (event) => loadRom(event.dataTransfer.files[0]));
addEventListener('keydown', (event) => handleKey(event, true));
addEventListener('keyup', (event) => handleKey(event, false));
