# uno-q-live — the live-demo duplicated for the Arduino Uno Q

**Status: code + docs only. Nothing has been flashed or run on the board yet.
No camera, no LED, no latency number has been verified on hardware.** The code
has been smoke-tested on a Linux x86-64 host (see below); treat every claim
about on-board behavior as unverified until you run it.

This folder is a port of `../../live-demo/live_explain.py` (the host-machine
demo) to the Uno Q's Linux MPU. The CLI, output format and both explanation
paths are unchanged; the inference model was swapped to FANE ResNet18:

```
camera frame (USB webcam / IP camera / still image) — user frames a
    close-up face, filled to the frame like the training images
    -> FANE ResNet18 inference (~11.2M params, 128x128 RGB, ImageNet norm)
    -> printed "explanation + suggested response" to the console
```

The model does **not** detect or crop faces. It was trained on face-centric
portrait images (face fills the frame; median ~94x123 px, aspect ~0.77) that
the default Haar cascade only detects on ~18% of samples — not enough to
build the demo on. **Frame the face to fill the view** (sit close to the
camera), like the training images. For a clearly-visible framed face, this
full-frame path is exactly the reference runner `../micheal chen/AI_Hackathon.py`.

Unlike a verbatim copy, the Uno Q constraints are respected: inference uses
the bundled ONNX/TorchScript assets (no repo `training/` import), the default
source is webcam `0`, everything degrades gracefully when a display, camera or
cloud service is missing, and an optional Bridge RPC blinks the onboard LED
via the MCU companion sketch (like `../uno-q-face/`).

Two explanation paths, same as the host demo:

1. **Fixed templates (default; always works offline)** — per-class hedged
   sentences plus a suggested-response line, identical wording to the host
   live-demo and `../uno-q-face/`.
2. **Tinker social coach (optional, cloud-side prototype)** — if Tinker is
   reachable and configured, each prediction gets a structured,
   confidence-hedged plain-language read (how they likely feel / what you
   might notice / what you can do). **It needs the network, a `TINKER_API_KEY`,
   and the `tinker` package on the machine that runs the
   script.** See "The Tinker social coach on the Uno Q" below.

The Tinker coach is a **cloud-side prototype** and is not flashed to the Uno
Q. Only the CNN and the Python app would ever live on the board.

## Layout

```
uno-q-live/
├── app.yaml                 App Lab app manifest
├── requirements.txt         Linux-MPU Python deps (onnxruntime, opencv-headless, numpy, pillow)
├── python/live_explain.py   the duplicated demo (full-frame -> ResNet18 -> explanation)
├── sketch/sketch.ino        optional MCU companion (STM32U585): LED blink per emotion
├── uno-complete/            ported michael chen model (lexfm0.pt + AI_Hackathon.py runner)
└── assets/
    ├── fane_resnet18_float32.onnx   default backend (input "input", [1,3,128,128])
    ├── fane_resnet18_float32.pt     TorchScript fallback (--backend torch)
    └── lexfm0.pt                  native ResNet18 checkpoint (--backend pytorch)
```

Models are fp32 and **large for this board** (~45 MB each; ~11.2M params —
roughly 45x the TinyEmoNet this app previously used). The ONNX and TorchScript
artifacts were exported from `../micheal chen/lexfm0.pt` by
`../../export/export_resnet18_fane.py` and verified on host: checkpoint vs
TorchScript logits agree exactly (max |diff| = 0.0), checkpoint vs ONNX ≈
5.7e-06 (< 1e-4 tolerance).

The `uno-complete/` folder is the full port of the michael chen model from
`../micheal chen/`: the original `lexfm0.pt` checkpoint and the
`AI_Hackathon.py` runner, running side by side in `uno-q-live/uno-complete/`.
The `lexfm0.pt` checkpoint is also bundled in `assets/` and can be run
natively in PyTorch via `--backend pytorch` (see below). No training code or
dataset is needed on the board.

## Run it — option A: Arduino App Lab

1. Copy this whole `uno-q-live/` folder into your App Lab apps directory
   (or import it as an app).
2. Let App Lab install `requirements.txt` for the Linux side.
3. Optionally compile/flash `sketch/` as the MCU companion (LED blinks).
   The Python app works without it — Bridge calls are best-effort.
4. Press Run. Sit close enough that your face fills the frame. The console
   prints timestamped lines and the explanation, e.g. (templates path):

   ```
   2026-11-07 19:36:16  angry 98%
       On-device style template: they looks frustrated or annoyed.
       Suggested response (fixed template): lower the stakes: give space, speak calmly, and ask an open question
   ```

App Lab runs `python/live_explain.py` with no arguments: webcam `0`,
continuous mode, templates (or auto-detected Tinker if you configured it in
the environment), Bridge signals on.

## Run it — option B: plain shell (on the board, or host smoke test)

```bash
cd uno-q-live
pip install -r requirements.txt        # host: use a venv

python3 python/live_explain.py --source 0                     # default webcam, continuous
python3 python/live_explain.py --source 0 --interval 3        # every 3s
python3 python/live_explain.py --source face.jpg --one-shot   # still image, single pass
python3 python/live_explain.py --source rtsp://user:pass@cam/stream --one-shot
python3 python/live_explain.py --backend torch --source face.jpg --one-shot   # TorchScript
python3 python/live_explain.py --backend pytorch --source face.jpg --one-shot   # native PyTorch (lexfm0.pt, uno-complete/)
python3 python/live_explain.py --no-tinker --source face.jpg --one-shot       # templates only
python3 python/live_explain.py --no-bridge --source face.jpg --one-shot       # never touch the Bridge
```

`--source` accepts a webcam index (`0`), an `rtsp://`/`http://` stream URL,
or a still image path (OpenCV `VideoCapture` handles all three). Still-image
files are EXIF-transposed before inference (camera frames carry no EXIF
orientation). Paths are resolved relative to the current working directory.
See `--help` for everything (`--max-frames`, `--show`, `--tinker-checkpoint`, ...).

Verified on host (x86-64 Linux, CPU), 2026-11-07, all one-shot:

```
$ python/live_explain.py --source ../../data/raw/fane_data/angry/angry1.jpg --one-shot --no-tinker --no-bridge
[uno-q-live] backend: onnx (fane_resnet18_float32.onnx)
[uno-q-live] --no-tinker set; using fixed templates
2026-11-07 19:36:16  angry 98%
    On-device style template: they looks frustrated or annoyed.
    Suggested response (fixed template): lower the stakes: give space, speak calmly, and ask an open question

$ python/live_explain.py --source ../../data/raw/fane_data/happy/happy1000.jpg --one-shot --no-tinker --no-bridge
[uno-q-live] backend: onnx (fane_resnet18_float32.onnx)
2026-11-07 19:36:17  happy 97%

$ python/live_explain.py --backend torch --source ../../data/raw/fane_data/neutral/neutral1.jpg --one-shot --no-tinker --no-bridge
[uno-q-live] backend: torch (fane_resnet18_float32.pt)
2026-11-07 19:36:20  neutral 96%
```

`--show` needs GUI OpenCV and a display; headless OpenCV degrades to a
one-time warning.

## The Tinker social coach on the Uno Q

The script keeps the host demo's optional Tinker path, but on the board it is
**off by default and usually unavailable**, by design:

- The checkpoint auto-detect (`--tinker-runs-root`, default: repo root when
  run inside a full checkout) looks for `tinker-verbalizer/runs/*/checkpoints.jsonl`.
  A bare app folder copied to the board has no such directory, so the script
  prints a clear note and falls back to templates — no failure.
- If you point `--tinker-checkpoint tinker://...` / `--tinker-runs-root` at a
  real run, you still need: network access, `TINKER_API_KEY` in the
  environment, and `pip install tinker` on the machine running
  the script. The package plus its dependency tree is heavy for the
  MPU (compare: torch is already a near-miss just for TorchScript inference).
- Nothing from the Tinker coach — weights, tokenizer, prompts — is flashed to
  the Uno Q. The cloud call, if enabled, sends the current sample plus the
  previous sample so the coach can track how the person's expression changed
  between samples: `{"label": ..., "confidence": ..., "at": ...}` for the
  previous frame and `{"label": ..., "confidence": ...}` for the current frame.

Recommended on-device configuration: `--no-tinker` (or simply leave Tinker
uninstalled). The templates give the same short, hedged, plain-language output
contract with zero network dependency.

## Relationship to `../uno-q-face/`

Two apps, same on-device core, different emphasis:

- `uno-q-face/` (`python/main.py`): minimal, strictly-offline app. No Tinker
  code path at all. Best fit for the board.
- `uno-q-live/` (this folder, `python/live_explain.py`): faithful duplicate of
  the host `live-demo/live_explain.py` — same CLI, same banners, same output
  wording — with only the changes listed below. Choose it if you want the live
  demo's exact interface on the board, with the Tinker path available as an
  explicit opt-in.

Differences vs. the host `live-demo/live_explain.py`:

| Area | Host live-demo | This duplicate |
|---|---|---|
| Inference | `training/model.py` `TinyEmoNet` + `export/tinyemonet_best.pt` (torch) | bundled ONNX `assets/fane_resnet18_float32.onnx` (ORT, default) or bundled TorchScript `.pt` (`--backend torch`) |
| Imports | imports the repo (`training.common` for CLASSES) | fully self-contained; `CLASSES` embedded |
| Default `--source` | IP-camera URL-ish / `0` | `0` (USB webcam on the board) |
| Bridge/MCU LED | none | optional `Bridge` RPC to `sketch/sketch.ino`; `--no-bridge` to disable |
| Headless OpenCV | tolerates missing `--show` | `--show` degrades to a warning instead of crashing (no display on the MPU) |
| Tinker auto-detect | `<repo>/tinker-verbalizer/runs` | same, plus `--tinker-runs-root` override; bare app folder → templates |
| Face Haar / crop | yes | removed — full frame scored (FANE ResNet18 was trained full-frame) |
| Preprocessing | 48×48 RGB, [0,1], (x−0.5)/0.5 | 128×128 RGB, EXIF transpose, aspect-preserving gray pad, ImageNet norm |
| `TEMPLATES` / softmax / output wording | — | unchanged |

## Honesty notes / not done yet

- **Model quality:** the bundled FANE ResNet18 reports **57.8% held-out test
  accuracy / macro F1 0.565 (N=1778)** in its model card (validation 58.4% /
  macro F1 0.569, selected epoch 20; seed 42; see `../../notes/provenance.md`
  and `../micheal chen/README.md`). Random = 1/9 ≈ 0.111. Predictions are not
  calibrated probabilities. This is a working prototype, not a reliable
  emotion reader.
- **Nothing measured on the Uno Q:** no on-board latency, memory, or accuracy
  numbers exist. A ~45 MB fp32 ResNet18 is much heavier than the TinyEmoNet
  this app used to bundle; the QRB2210's CPU and 4 GB RAM budget are unknowns
  until timed on the board. Do not quote any numbers. Time it on the board,
  then update this file.
- **`sketch/sketch.ino` has not been compiled** with Arduino_RouterBridge.h.
  It blinks the onboard LED: solid=happy, fast blink=surprise, slow blink=
  angry/disgust/fear/sad, off=everything else. Verify with App Lab before relying on it.
- **Audio output is not implemented.** The templated/Tinker sentences go to
  the console only.
- No face detection means framing matters: profiles, occlusions, low light and
  loose framing will degrade results, and the app cannot tell you when the
  frame is bad — it will confidently classify whatever is in view.
- Do not use for safety- or affect-critical decisions. See `../../notes/limitations.md`.
