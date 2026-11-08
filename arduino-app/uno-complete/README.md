# uno-complete — ported michael chen image-recognition model into uno-q-live

This folder is the complete port of the **michael chen** image-recognition model
(from `../micheal chen/`) into the `uno-q-live/` app, kept self-contained so the
model and its runner live together under `uno-q-live/uno-complete/`.

## Source of the model

The model comes from `arduino/micheal chen/` (the original reference runner),
which is the ResNet18 checkpoint `lexfm0.pt` trained by Michael Chen on the
FANE (Facial Expressions & Emotion) 9-class expression dataset.

Everything from that folder that is relevant to running the model has been
ported here:

| Ported from `../micheal chen/` | Where it lives now |
|---|---|
| `lexfm0.pt` (the ResNet18 checkpoint, 11,181,129 params) | `lexfm0.pt` |
| `AI_Hackathon.py` (prediction runner: architecture + load + preprocess + predict) | `AI_Hackathon.py` |
| `requirements.txt` (torch, numpy, pillow) | `requirements.txt` |
| `README.md` (model card + usage) | `README.md` (this file, with full context) |

## What the model does

The final ResNet18 model predicts **nine labels**:

`angry, confused, disgust, fear, happy, neutral, sad, shy, surprise`

It is a full-frame classifier — it does **not** detect or crop faces. It was
trained on face-centric portrait images where the face fills the frame
(median ~94x123 px, aspect ~0.77). Frame the face to fill the view (sit close
to the camera), like the training images.

## Preprocessing (must match training)

Automatically handles, in order:

1. EXIF orientation correction
2. Convert to RGB
3. Aspect-preserving gray `(128, 128, 128)` pad to `128x128` (LANCZOS)
4. `float32 / 255`
5. ImageNet normalization (mean `0.485/0.456/0.406`, std `0.229/0.224/0.225`)

This matches `../micheal chen/AI_Hackathon.py` exactly and is bit-exact with
the reference runner on sample images.

## Model card (from the michael chen README)

- **Final reported held-out test results:** 1,778 images; accuracy **57.8%**;
  macro F1 **0.565**.
- **Validation:** accuracy 58.4%; macro F1 0.569; selected epoch 20.
- Random baseline = 1/9 ≈ 0.111.
- Predictions are **probabilities from the model, not calibrated confidence**
  or reliable measurements of someone's internal emotional state.

## How to run

```bash
cd uno-complete
source ../../../.venv/bin/activate   # or create your own venv
python -m pip install -r requirements.txt

python AI_Hackathon.py predict "/path/to/image.jpg"
```

Multiple images can be listed after `predict`. CPU is the default; add
`--device mps` to use the Mac GPU. The checkpoint (`lexfm0.pt`) is selected
automatically. No dataset or internet connection is needed to run predictions.

If rebuilding the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Relationship to the sibling apps in `arduino/`

- `uno-q-live/` (parent): duplicates the host `live-demo/live_explain.py` on
  the Uno Q. It uses the bundled ONNX/TorchScript FANE ResNet18 assets in
  `uno-q-live/assets/` (exported from this same `lexfm0.pt` by
  `../../export/export_resnet18_fane.py`) and adds a Tinker cloud-social-coach
  opt-in plus optional MCU Bridge LED signaling.
- `uno-q-face/`: minimal, strictly-offline Uno Q app using the same bundled
  FANE ResNet18 assets.

This `uno-complete/` folder instead keeps the **original PyTorch checkpoint**
`lexfm0.pt` and the original `AI_Hackathon.py` runner from `micheal chen/`,
so you can run the model in its original form right next to the Uno Q app.

## Honesty notes

- Nothing has been flashed or measured on a real Arduino Uno Q. On-device
  latency, memory, and accuracy are unverified.
- This is a working prototype, not a reliable emotion reader. Do not use for
  safety- or affect-critical decisions.
- No face detection means framing matters: profiles, occlusions, low light and
  loose framing will degrade results; the app will confidently classify
  whatever is in view.
