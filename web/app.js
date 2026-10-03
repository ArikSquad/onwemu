import init, { WebEmulator } from './pkg/onwemu.js';

const SYSTEMS = {
  nes: { extensions: ['.nes'] },
  gb: { extensions: ['.gb'] },
  psp: { extensions: ['.iso', '.chd', '.pbp', '.elf'] },
};

const KEY_MAP = {
  arrowup: 'up',
  arrowdown: 'down',
  arrowleft: 'left',
  arrowright: 'right',
  z: 'a',
  x: 'b',
  c: 'square',
  v: 'triangle',
  q: 'l',
  e: 'r',
  enter: 'start',
  shift: 'select',
  escape: 'home',
};

const ui = {
  change: document.getElementById('change'),
  drop: document.getElementById('drop'),
  error: document.getElementById('error'),
  filename: document.getElementById('filename'),
  fullscreen: document.getElementById('fullscreen'),
  picker: document.getElementById('picker'),
  player: document.getElementById('player'),
  rom: document.getElementById('rom'),
  screen: document.getElementById('screen'),
  systems: document.querySelectorAll('.system'),
  controls: document.getElementById('controls'),
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
  if (document.fullscreenElement === ui.player) {
    document.exitFullscreen().catch(() => {});
  }
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
  ui.rom.accept = SYSTEMS[system].extensions.join(',');
  ui.controls.innerHTML = system === 'psp'
    ? '<span>Move <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd></span><span>Cross <kbd>Z</kbd></span><span>Circle <kbd>X</kbd></span><span>Square <kbd>C</kbd></span><span>Triangle <kbd>V</kbd></span><span>L/R <kbd>Q</kbd><kbd>E</kbd></span><span>Start <kbd>Enter</kbd></span><span>Select <kbd>Shift</kbd></span><span>Home <kbd>Esc</kbd></span>'
    : '<span>Move <kbd>↑</kbd><kbd>↓</kbd><kbd>←</kbd><kbd>→</kbd></span><span>A <kbd>Z</kbd></span><span>B <kbd>X</kbd></span><span>Start <kbd>Enter</kbd></span><span>Select <kbd>Shift</kbd></span>';
  setError();
}

selectSystem(state.system);

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

  const { extensions } = SYSTEMS[state.system];
  setError();
  if (!extensions.some((extension) => file.name.toLowerCase().endsWith(extension))) {
    setError(`Choose a supported file: ${extensions.join(', ')}.`);
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
  if (event.key.toLowerCase() === 'escape' && document.fullscreenElement === ui.player) return;
  const button = KEY_MAP[event.key.toLowerCase()];
  if (!state.emulator || !button) return;

  event.preventDefault();
  state.emulator.set_button(button, down);
}

ui.systems.forEach((button) => {
  button.addEventListener('click', () => selectSystem(button.dataset.system));
});

async function toggleFullscreen() {
  try {
    if (document.fullscreenElement === ui.player) {
      await document.exitFullscreen();
    } else {
      await ui.player.requestFullscreen();
    }
  } catch (error) {
    ui.fullscreen.title = `Unable to change fullscreen mode: ${errorMessage(error)}`;
  }
}

ui.fullscreen.disabled = typeof ui.player.requestFullscreen !== 'function';
ui.fullscreen.title = ui.fullscreen.disabled
  ? 'Fullscreen is not available in this browser'
  : 'Enter fullscreen mode';
ui.fullscreen.addEventListener('click', toggleFullscreen);
document.addEventListener('fullscreenchange', () => {
  const active = document.fullscreenElement === ui.player;
  ui.fullscreen.textContent = active ? 'Exit fullscreen' : 'Fullscreen';
  ui.fullscreen.setAttribute('aria-pressed', String(active));
  ui.fullscreen.title = active ? 'Exit fullscreen mode' : 'Enter fullscreen mode';
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
