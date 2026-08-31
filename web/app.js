import init, { WebEmulator } from './pkg/onwemu.js';

let system = 'nes', emulator = null, animation = 0, wasmReady;
const $ = id => document.getElementById(id);
const ext = { nes: '.nes', gb: '.gb' };
const keymap = { ArrowUp:'up', ArrowDown:'down', ArrowLeft:'left', ArrowRight:'right', z:'a', Z:'a', x:'b', X:'b', Enter:'start', Shift:'select' };

document.querySelectorAll('.system').forEach(button => button.onclick = () => {
  document.querySelectorAll('.system').forEach(x => { x.classList.toggle('active', x === button); x.setAttribute('aria-checked', x === button); });
  system = button.dataset.system; $('rom').accept = ext[system]; $('error').textContent = '';
});

async function load(file) {
  $('error').textContent = '';
  if (!file.name.toLowerCase().endsWith(ext[system])) return void ($('error').textContent = `That does not look like a ${ext[system]} file.`);
  try {
    await (wasmReady ??= init());
    emulator?.free();
    emulator = new WebEmulator(system, new Uint8Array(await file.arrayBuffer()));
    const canvas = $('screen'); canvas.width = emulator.width; canvas.height = emulator.height;
    canvas.style.aspectRatio = `${canvas.width}/${canvas.height}`;
    $('filename').textContent = file.name; $('picker').hidden = true; $('player').hidden = false;
    cancelAnimationFrame(animation); frame();
  } catch (error) { $('picker').hidden = false; $('player').hidden = true; $('error').textContent = error.message || String(error); }
}
function frame() {
  if (!emulator) return;
  const pixels = emulator.run_frame();
  $('screen').getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(pixels), emulator.width, emulator.height), 0, 0);
  animation = requestAnimationFrame(frame);
}
$('rom').onchange = event => event.target.files[0] && load(event.target.files[0]);
$('change').onclick = () => { cancelAnimationFrame(animation); emulator?.free(); emulator = null; $('player').hidden = true; $('picker').hidden = false; $('rom').value = ''; };
['dragenter','dragover'].forEach(name => $('drop').addEventListener(name, e => { e.preventDefault(); $('drop').classList.add('over'); }));
['dragleave','drop'].forEach(name => $('drop').addEventListener(name, e => { e.preventDefault(); $('drop').classList.remove('over'); }));
$('drop').addEventListener('drop', e => e.dataTransfer.files[0] && load(e.dataTransfer.files[0]));
addEventListener('keydown', e => { if (emulator && keymap[e.key]) { e.preventDefault(); emulator.set_button(keymap[e.key], true); } });
addEventListener('keyup', e => { if (emulator && keymap[e.key]) { e.preventDefault(); emulator.set_button(keymap[e.key], false); } });
