#!/usr/bin/env python3
"""uno-q-live — live facial-expression demo, duplicated for the Arduino Uno Q.

This is a port of ../../live-demo/live_explain.py (the host-machine demo) to
the Arduino Uno Q's Linux MPU (Dragonwing QRB2210, Debian via App Lab). The
CLI, output format, and both explanation paths are unchanged; the inference
model was swapped from TinyEmoNet to FANE ResNet18:

    camera frame / still image (user frames a clear, close-up face)
        -> FANE ResNet18 inference (~11.2M params, 128x128 RGB, ImageNet norm)
        -> printed "explanation + suggested response" to the console
           (Tinker cloud social coach if available, else fixed templates)

The model does NOT detect or crop faces: it was trained on face-centric
portrait images (face fills the frame; median ~94x123 px, aspect ~0.77) that
the default Haar cascade only detects on ~18% of samples. Frame the face to
fill the view like the training images (sit close to the camera); for a
clearly-visible framed face this full-frame path is exactly the reference
runner ../micheal chen/AI_Hackathon.py.

What changed vs. the host live-demo (and why):

  * Inference runs on a bundled ONNX model (assets/fane_resnet18_float32.onnx,
    default) or the bundled TorchScript asset (--backend torch) — the board
    has no repo checkout, and torch + training code are far too heavy for it.
    The host demo instead loads export/tinyemonet_best.pt through
    training/model.py (TinyEmoNet in Python); this app uses the FANE ResNet18
    export. Class order is unchanged (0-8).
  * CLASSES / TEMPLATES are embedded; nothing imports from ../../training/.
  * --source defaults to 0 (USB webcam on the board) instead of an IP camera.
  * Optional Bridge signal to the MCU companion sketch (onboard LED), like
    ../uno-q-face. Never required; use --no-bridge to disable.
  * opencv-python-headless is the expected OpenCV (no display on the MPU);
    --show needs a GUI build of OpenCV and a display.
  * The Tinker social-coach auto-detect also searches <repo>/tinker-verbalizer/
    runs when the folder sits inside a full checkout, and takes an explicit
    --tinker-runs-root / --tinker-checkpoint. On the board by itself there is
    no runs directory, so it cleanly falls back to fixed templates.

The Tinker social coach is a CLOUD-SIDE PROTOTYPE. It is NOT the on-device
model and nothing from it is flashed to the Uno Q. If Tinker is unavailable
(no API key, no completed run, or --no-tinker), the script falls back to
per-class fixed sentences plus a template-based "suggested response" line,
and says so clearly.

Two ways to run:

  * Arduino App Lab: add this folder as an app (see ../README.md). App Lab
    runs this file with no arguments; it opens the default webcam (source 0),
    prints to the app console and, if the companion sketch is loaded, signals
    the MCU over the Bridge.
  * Plain shell (on the board, or on any host for a smoke test):

        python3 python/live_explain.py --source 0                 # webcam, continuous
        python3 python/live_explain.py --source face.jpg --one-shot   # still image
        python3 python/live_explain.py --help

Preprocessing matches the lexfm0 reference runner / model card exactly
(see ../../notes/provenance.md): EXIF orientation fix -> RGB ->
aspect-preserving gray (128,128,128) pad to 128x128 -> float32 /255 ->
ImageNet normalization. Still images passed via --source are EXIF-transposed
before use (camera frames have no EXIF orientation).

Honest performance note: the bundled ResNet18 reports 57.8% held-out test
accuracy / macro F1 0.565 (N=1778) in its model card (validation 58.4% /
0.569 @ epoch 20; see ../../notes/provenance.md). Predictions are not
calibrated probabilities. Treat this as a working prototype, not a product.

Notes:
- Nothing has been flashed or measured on a real Uno Q yet. On-board latency,
  memory and accuracy are UNVERIFIED.
- TINKER_API_KEY must be set in the environment for the cloud social coach;
  it stays on your machine and is sent only to the Tinker service. The coach
  needs `pip install tinker`, which is heavy on the MPU; tinker-cookbook is
  not required.
  the default (no key / not installed / --no-tinker) is the template path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("opencv-python-headless is required: pip install opencv-python-headless")

APP_DIR = Path(__file__).resolve().parent.parent
ASSETS = APP_DIR / "assets"
ONNX_MODEL = ASSETS / "fane_resnet18_float32.onnx"  # input: "input"  [1,3,128,128]
TORCH_MODEL = ASSETS / "fane_resnet18_float32.pt"   # TorchScript, CPU
PYTORCH_MODEL = ASSETS / "lexfm0.pt"                # native ResNet18 checkpoint (michael chen port, uno-complete/)
# Full repo checkout ( …/unoq-fane-micro-llm/arduino/uno-q-live/python/x.py ):
REPO_ROOT = Path(__file__).resolve().parents[3]
IMG_SIZE = 128

# ImageNet normalization (must match the checkpoint's 'normalization: imagenet').
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]

# Canonical FANE class order (must match the lexfm0 checkpoint 'labels').
CLASSES = ["angry", "confused", "disgust", "fear", "happy",
           "neutral", "sad", "shy", "surprise"]

# Template fallback sentences + suggested responses, used when Tinker is
# unavailable or --no-tinker is passed. Identical wording to the host
# live-demo. The on-device path is templates only; the cloud social coach
# adds confidence-aware hedging per call.
TEMPLATES = {
    "angry":    ("looks frustrated or annoyed",
                 "lower the stakes: give space, speak calmly, and ask an open question"),
    "confused": ("seems to be struggling to follow",
                 "slow down, re-explain step by step, or ask what part is unclear"),
    "disgust":  ("finds something off-putting or unpleasant",
                 "identify what bothers them and remove or address it; don't push"),
    "fear":     ("may be anxious or unsafe",
                 "stay calm, don't startle them, acknowledge the worry, and reassure"),
    "happy":    ("looks genuinely positive or engaged",
                 "keep the current approach; mirror the energy and invite them to continue"),
    "neutral":  ("shows no strong emotional signal",
                 "no action needed; a neutral face is a perfectly fine state"),
    "sad":      ("may feel low, hurt, or discouraged",
                 "acknowledge the feeling, listen without fixing, and offer a break or support"),
    "shy":      ("seems hesitant or withdrawing",
                 "reduce pressure: offer written/alternative ways to respond, be patient"),
    "surprise": ("looks startled or taken aback",
                 "give them a moment; confirm the unexpected thing and check their reaction"),
}

# --- optional Bridge link to the MCU (sketch/sketch.ino), on-device only ------
try:
    from arduino.app_utils import Bridge  # type: ignore
except Exception:  # not under App Lab (e.g. host smoke test)
    Bridge = None


def bridge_set_emotion(class_index: int) -> None:
    """Best-effort 'blink the onboard LED' signal to the MCU. Never fatal."""
    if Bridge is None:
        return
    try:
        Bridge.call("set_emotion", int(class_index))
    except Exception:
        pass  # no sketch loaded / bridge unavailable: carry on silently


# --- native ResNet18 architecture (ported from ../micheal chen/AI_Hackathon.py) ---
# This is the original model definition from the michael chen image-recognition
# model, kept so the uno-complete/ checkpoint (lexfm0.pt) can run natively in
# PyTorch alongside the ONNX / TorchScript exports used by the other backends.
# Defined lazily behind a torch availability check so the ONNX path keeps
# working on machines without torch installed.


try:  # torch is optional; only the pytorch backend needs it
    import torch as _torch
    from torch import nn as _nn
except ImportError:  # pragma: no cover
    _torch = None
    _nn = None

if _nn is not None:

    class ResidualBlock(_nn.Module):
        """Basic ResNet block with names compatible with the official weights."""

        def __init__(self, inputs, outputs, stride=1):
            super().__init__()
            self.conv1 = _nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
            self.bn1 = _nn.BatchNorm2d(outputs)
            self.relu = _nn.ReLU(inplace=True)
            self.conv2 = _nn.Conv2d(outputs, outputs, 3, padding=1, bias=False)
            self.bn2 = _nn.BatchNorm2d(outputs)
            self.downsample = None
            if stride != 1 or inputs != outputs:
                self.downsample = _nn.Sequential(
                    _nn.Conv2d(inputs, outputs, 1, stride, bias=False),
                    _nn.BatchNorm2d(outputs))

        def forward(self, x):
            identity = x if self.downsample is None else self.downsample(x)
            x = self.relu(self.bn1(self.conv1(x)))
            return self.relu(self.bn2(self.conv2(x)) + identity)


    class ResNet18(_nn.Module):
        """ResNet18 architecture for loading the selected local checkpoint."""

        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = _nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = _nn.BatchNorm2d(64)
            self.relu = _nn.ReLU(inplace=True)
            self.maxpool = _nn.MaxPool2d(3, 2, 1)
            self.layer1 = _nn.Sequential(ResidualBlock(64, 64), ResidualBlock(64, 64))
            self.layer2 = _nn.Sequential(ResidualBlock(64, 128, 2), ResidualBlock(128, 128))
            self.layer3 = _nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256))
            self.layer4 = _nn.Sequential(ResidualBlock(256, 512, 2), ResidualBlock(512, 512))
            self.avgpool = _nn.AdaptiveAvgPool2d((1, 1))
            self.fc = _nn.Linear(512, 1000)
            self.fc = _nn.Linear(512, num_classes)

        def forward(self, x):
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
            return self.fc(_torch.flatten(self.avgpool(x), 1))


# --- inference backends (no training code needed on the board) -----------------
class OnnxBackend:
    def __init__(self, path: Path):
        import onnxruntime as ort
        self._sess = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"])
        self.input_name = self._sess.get_inputs()[0].name

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return self._sess.run(None, {self.input_name: batch})[0]


class TorchBackend:
    def __init__(self, path: Path):
        import torch
        self._torch = torch
        self._model = torch.jit.load(str(path), map_location="cpu")
        self._model.eval()

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with self._torch.no_grad():
            return self._model(self._torch.from_numpy(batch)).numpy()


class PyTorchBackend:
    """Native PyTorch ResNet18 running the uno-complete/ checkpoint (lexfm0.pt).

    Loads the original michael chen checkpoint through the ported ResNet18
    architecture and runs inference directly in PyTorch (no ONNX, no
    TorchScript). This is the backend that exercises the uno-complete model
    in its native form, right next to the ONNX / TorchScript exports.
    """

    def __init__(self, path: Path):
        if _torch is None or ResNet18 is None:  # pragma: no cover
            sys.exit("backend 'pytorch' selected but torch is not installed: "
                     "pip install torch")
        checkpoint = _torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != "resnet18":
            sys.exit(f"unexpected architecture: {checkpoint.get('architecture')}")
        labels = checkpoint["labels"]
        if list(labels) != CLASSES:
            sys.exit(f"label order mismatch: {labels} != {CLASSES}")
        model = ResNet18(len(labels))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self._torch = _torch
        self._model = model.to("cpu").eval()
        self._labels = labels

    def predict(self, batch: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            return self._model(self._torch.from_numpy(batch)).numpy()


def load_backend(name: str):
    if name in ("auto", "onnx"):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            if name == "onnx":
                sys.exit("backend 'onnx' selected but onnxruntime is not installed: "
                         "pip install onnxruntime")
        else:
            if not ONNX_MODEL.exists():
                sys.exit(f"bundled ONNX model missing: {ONNX_MODEL}")
            print(f"[uno-q-live] backend: onnx ({ONNX_MODEL.name})")
            return OnnxBackend(ONNX_MODEL)
    if name in ("auto", "torch"):
        try:
            import torch  # noqa: F401
        except ImportError:
            if name == "torch":
                sys.exit("backend 'torch' selected but torch is not installed: "
                         "pip install torch")
            sys.exit("no inference backend available: pip install onnxruntime "
                     "(preferred, small) or torch")
        if not TORCH_MODEL.exists():
            sys.exit(f"bundled TorchScript model missing: {TORCH_MODEL}")
        print(f"[uno-q-live] backend: torch ({TORCH_MODEL.name})")
        return TorchBackend(TORCH_MODEL)
    if name in ("auto", "pytorch"):
        if _torch is None or ResNet18 is None:
            if name == "pytorch":
                sys.exit("backend 'pytorch' selected but torch is not installed: "
                         "pip install torch")
            sys.exit("no inference backend available: pip install onnxruntime "
                     "(preferred, small) or torch")
        if not PYTORCH_MODEL.exists():
            sys.exit(f"bundled PyTorch model missing: {PYTORCH_MODEL}")
        print(f"[uno-q-live] backend: pytorch ({PYTORCH_MODEL.name})")
        return PyTorchBackend(PYTORCH_MODEL)
    sys.exit(f"unknown backend: {name} (expected auto|onnx|torch|pytorch)")


# --- vision pipeline -----------------------------------------------------------
def preprocess(rgb) -> np.ndarray:
    """Training-parity preprocessing: aspect-preserving gray (128,128,128) pad
    to 128x128 -> float32 /255 -> ImageNet normalization. Returns [1,3,128,128]."""
    from PIL import Image, ImageOps

    image = Image.fromarray(rgb)
    image = ImageOps.pad(image, (IMG_SIZE, IMG_SIZE),
                         method=Image.Resampling.LANCZOS,
                         color=(128, 128, 128))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    tensor = arr.transpose(2, 0, 1)  # HWC -> CHW
    normalized = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(normalized[None, ...].astype(np.float32))


def classify(backend, rgb):
    """Score the full frame with FANE ResNet18 (the training distribution)."""
    batch = preprocess(rgb)
    logits = backend.predict(batch)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    idx = int(probs[0].argmax())
    return CLASSES[idx], float(probs[0][idx]), idx


def find_latest_tinker_checkpoint(runs_root: Path):
    """Auto-detect the newest tinker:// sampler_path from a verbalizer run."""
    runs = runs_root / "tinker-verbalizer" / "runs"
    if not runs.is_dir():
        return None, None
    candidates = sorted(runs.glob("*/checkpoints.jsonl"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        except OSError:
            continue
        for line in reversed(lines):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("sampler_path"):
                return rec["sampler_path"], path.parent.name
    return None, None


class TinkerVerbalizer:
    """Cloud-side Tinker LoRA social coach (prototype, not on-device).

    Requires only the ``tinker`` Python package (``tinker-cookbook`` is not
    needed). The prompt is built as plain text and sent to the Tinker service
    without a chat-template tokenizer.
    """

    def __init__(self, sampler_path: str, base_model: str = "Qwen/Qwen3.5-4B",
                 max_tokens: int = 220, temperature: float = 0.7, top_p: float = 0.9):
        import os
        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError("TINKER_API_KEY is not set")
        import tinker

        service = tinker.ServiceClient()
        self.client = service.create_sampling_client(model_path=sampler_path).result()
        self.base_model = base_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _build_model_input(self, prompt: str):
        """Build a tinker ModelInput from a plain-text prompt.

        Tries text-based constructors first, then falls back to a simple
        character-level integer encoding so no external tokenizer is needed.
        """
        import tinker

        # Prefer a text-based constructor if the tinker package exposes one.
        for attr in ("from_text", "from_str", "from_prompt"):
            ctor = getattr(tinker.types.ModelInput, attr, None)
            if ctor is not None:
                try:
                    return ctor(prompt)
                except Exception:
                    continue

        # Fallback: simple character-level integer encoding.
        try:
            ints = [ord(ch) for ch in prompt]
            return tinker.types.ModelInput.from_ints(ints)
        except Exception:
            # Last resort: empty input (service may still respond).
            return tinker.types.ModelInput.from_ints([])

    def explain(self, label: str, confidence: float,
                previous: dict | None = None) -> str:
        import tinker
        current = ('The camera sees now: {"label": "%s", "confidence": %.2f}. '
                   % (label, confidence))
        if previous is not None and previous.get("label") is not None:
            previous_part = ('Previously the camera saw: '
                             '{"label": "%s", "confidence": %.2f, "at": "%s"}. '
                             % (previous["label"], previous["confidence"],
                                previous["stamp"]))
            prompt = (previous_part + current +
                      ' The person is on camera across these two samples. '
                      'What are they feeling now, how has their expression '
                      'changed since the previous sample, and what should I do?')
        else:
            prompt = current + 'What are they feeling, and what should I do?'

        model_input = self._build_model_input(prompt)
        params = tinker.types.SamplingParams(
            max_tokens=self.max_tokens, temperature=self.temperature, top_p=self.top_p)
        resp = self.client.sample(model_input, num_samples=1, sampling_params=params).result()

        # Decode the response: try token-based decode first, then raw text.
        text = ""
        try:
            tokens = getattr(resp.sequences[0], "tokens", None)
            if tokens is not None:
                text = self._decode_tokens(tokens)
        except Exception:
            text = ""
        if not text:
            try:
                text = getattr(resp.sequences[0], "text", "") or ""
            except Exception:
                text = ""
        text = text.strip()
        # The fine-tuned model tends to open with an empty </think> marker; drop it.
        if text.startswith("</think>"):
            text = text[len("</think>"):].strip()
        return text

    @staticmethod
    def _decode_tokens(tokens) -> str:
        """Best-effort decode of token ids to text without a tokenizer."""
        chars = []
        for tok in tokens:
            try:
                t = int(tok)
                if 32 <= t <= 126:  # printable ASCII
                    chars.append(chr(t))
                elif t == 10:
                    chars.append("\n")
                elif t == 13:
                    chars.append("\r")
                elif t == 9:
                    chars.append("\t")
                else:
                    chars.append(" ")
            except (TypeError, ValueError):
                chars.append(" ")
        return "".join(chars)


def load_app_config() -> dict:
    """Load user configuration for the Arduino App Lab app.

    Priority (highest wins):
      1. CLI arguments (parsed by argparse in main()).
      2. Environment variables (UNO_COMPLETE_<NAME>, uppercase).
      3. App Lab config JSON file (ARDUINO_APP_CONFIG env var, or
         <APP_DIR>/app.config.json).
      4. Hard-coded argparse defaults.

    App Lab injects user-configured values via environment variables and/or a
    config JSON file; this function merges them so the app is fully configurable
    without editing code.
    """
    config = {}

    # 3. App Lab config JSON file (lowest priority, above argparse defaults).
    config_path = os.environ.get("ARDUINO_APP_CONFIG")
    if not config_path:
        config_path = APP_DIR / "app.config.json"
    else:
        config_path = Path(config_path)
    if config_path and Path(config_path).is_file():
        try:
            with open(config_path, encoding="utf-8") as fh:
                file_cfg = json.load(fh)
            if isinstance(file_cfg, dict):
                config.update(file_cfg)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[uno-complete] could not read app config {config_path}: {exc}")

    # 2. Environment variables (override the config file).
    #    UNO_COMPLETE_<NAME> → config key <name> (hyphens, matching CLI args).
    env_prefix = "UNO_COMPLETE_"
    for key, value in os.environ.items():
        if key.startswith(env_prefix):
            name = key[len(env_prefix):].lower().replace("_", "-")
            # Booleans come as "true"/"false" strings; convert them.
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            config[name] = value

    # 1. CLI arguments are applied in main() by setting defaults from this
    #    config dict before parsing, so explicit CLI flags still win.
    return config


def decode_source(raw: str):
    return int(raw) if raw.isdigit() else raw


def is_still_path(raw: str) -> bool:
    """True for a still-image file path (not a webcam index or stream URL)."""
    if raw.isdigit():
        return False
    lowered = raw.lower()
    if lowered.startswith(("rtsp://", "http://", "https://", "udp://", "tcp://")):
        return False
    return Path(raw).is_file()


def load_still(path: str):
    """Load a still image as RGB, applying EXIF orientation like the reference runner."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        return np.asarray(ImageOps.exif_transpose(source).convert("RGB"))


def explain_frame(rgb, backend, verbalizer, args, stamp: str,
                  previous: dict | None = None):
    """Score one frame, print the explanation, return (label, conf, idx).

    If ``previous`` is supplied (the result of the prior sample), it is handed
    to the Tinker social coach so the model can track how the person's
    expression changed between samples and react accordingly.
    """
    label, conf, class_index = classify(backend, rgb)
    print(f"{stamp}  {label} {conf:.0%}")
    if verbalizer is not None:
        try:
            coach_text = verbalizer.explain(label, conf, previous=previous)
            coach_lines = coach_text.splitlines() or [coach_text]
            print("    Tinker social coach (cloud prototype):")
            for coach_line in coach_lines:
                print(f"        {coach_line}")
        except Exception as exc:  # noqa: BLE001
            print(f"    [Tinker error: {exc}]")
            meaning, advice = TEMPLATES[label]
            print(f"    (fallback) On-device style template: they {meaning}.")
            print(f"    (fallback) Suggested response (fixed template): {advice}")
    else:
        meaning, advice = TEMPLATES[label]
        print(f"    On-device style template: they {meaning}.")
        print(f"    Suggested response (fixed template): {advice}")
    if not args.no_bridge:
        bridge_set_emotion(class_index)
    return label, conf, class_index, {
        "label": label, "confidence": conf, "stamp": stamp,
    }


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    # Load App Lab / environment configuration and use it as argparse defaults
    # so explicit CLI flags still win.
    app_config = load_app_config()

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=app_config.get("source", "0"),
                    help="webcam index (default: 0), IP camera URL "
                         "(rtsp://.../http://...), or a still image path")
    ap.add_argument("--backend", default=app_config.get("backend", "auto"),
                    choices=["auto", "onnx", "torch", "pytorch"],
                    help="inference backend (default: auto = onnx if available, "
                         "else torch; pytorch runs the uno-complete/ lexfm0.pt "
                         "checkpoint natively in PyTorch)")
    ap.add_argument("--tinker-checkpoint",
                    default=app_config.get("tinker-checkpoint") or None,
                    help="tinker:// sampler_path of the LoRA social coach; "
                         "default: auto-detect newest tinker-verbalizer/runs/*/checkpoints.jsonl "
                         "(needs a full repo checkout; otherwise templates are used)")
    ap.add_argument("--tinker-runs-root",
                    default=app_config.get("tinker-runs-root") or REPO_ROOT,
                    type=Path,
                    help="directory that contains tinker-verbalizer/runs "
                         "(default: repo root when inside a checkout, else the app folder)")
    ap.add_argument("--no-tinker",
                    default=_as_bool(app_config.get("no-tinker", False)),
                    action="store_true",
                    help="skip the cloud social coach; use built-in fixed templates")
    ap.add_argument("--interval", type=float,
                    default=float(app_config.get("interval", 10.0)),
                    help="seconds between classifications (default: 10.0 — "
                         "one sample every ten seconds so frames do not pile up)")
    ap.add_argument("--save-frames",
                    default=app_config.get("save-frames") or None,
                    type=Path,
                    help="folder to save each analyzed frame into (created if "
                         "missing); frames are written with a timestamp name "
                         "(disabled by default)")
    ap.add_argument("--max-frames", type=int,
                    default=int(app_config.get("max-frames", -1)),
                    help="stop after this many classified frames (-1 = run forever)")
    ap.add_argument("--one-shot",
                    default=_as_bool(app_config.get("one-shot", False)),
                    action="store_true",
                    help="classify a single frame and exit")
    ap.add_argument("--show",
                    default=_as_bool(app_config.get("show", False)),
                    action="store_true",
                    help="show an annotated OpenCV window (needs GUI OpenCV + display; "
                         "falls back to headless with a warning otherwise)")
    ap.add_argument("--no-bridge",
                    default=_as_bool(app_config.get("no-bridge", False)),
                    action="store_true",
                    help="do not signal the MCU over the Bridge (default: signal)")
    args = ap.parse_args()

    backend = load_backend(args.backend)

    verbalizer = None
    if not args.no_tinker:
        sampler_path = args.tinker_checkpoint
        run_hint = "--tinker-checkpoint"
        if sampler_path is None:
            runs_root = args.tinker_runs_root
            if not (runs_root / "tinker-verbalizer" / "runs").is_dir():
                runs_root = Path(__file__).resolve().parent.parent  # this app folder
            sampler_path, run_hint = find_latest_tinker_checkpoint(runs_root)
            if sampler_path:
                print(f"[uno-q-live] auto-detected Tinker checkpoint in run '{run_hint}'")
                run_hint = f"run '{run_hint}'"
        if sampler_path:
            try:
                verbalizer = TinkerVerbalizer(sampler_path)
                print("[uno-q-live] Tinker social coach ready (cloud-side prototype)")
            except Exception as exc:  # noqa: BLE001
                print(f"[uno-q-live] WARNING: Tinker social coach unavailable ({exc}); "
                      "falling back to fixed templates")
        else:
            print("[uno-q-live] no Tinker checkpoint found "
                  "(no tinker-verbalizer runs next to this app, or auto-detect failed); "
                  "falling back to fixed templates")
    else:
        print("[uno-q-live] --no-tinker set; using fixed templates")

    if is_still_path(args.source):
        # Still images: EXIF-transposed like the reference runner, scored once.
        rgb = load_still(args.source)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        explain_frame(rgb, backend, verbalizer, args, stamp)
        return 0

    source = decode_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"could not open camera source: {args.source}")

    show = args.show

    def imshow(window: str, img) -> bool:
        """Wrapper so headless OpenCV degrades gracefully instead of crashing."""
        nonlocal show
        try:
            cv2.imshow(window, img)
            return True
        except cv2.error as exc:
            print(f"[uno-q-live] --show unavailable ({exc}); continuing headless")
            show = False
            return False

    classified = 0
    last_run = None
    previous = None  # result of the prior sample, handed to the Tinker coach
    retries = 0
    print("[uno-complete] camera opened. Taking one sample every "
          f"{args.interval:.0f} seconds (frames do not pile up). "
          "Press Ctrl+C to stop.")
    try:
        while True:
            if args.max_frames >= 0 and classified >= args.max_frames:
                break
            # Wait until the next sample is due so we only analyze a single
            # frame once every `interval` seconds (default 10s) rather than
            # chewing through a continuous stream. Frames do not pile up.
            now = time.time()
            if last_run is None:
                next_due = now  # first sample happens immediately
            else:
                next_due = last_run + args.interval
            sleep_for = next_due - now
            if sleep_for > 0:
                if show:
                    # Keep the display responsive to the quit key while waiting.
                    deadline = now + sleep_for
                    while time.time() < deadline:
                        remaining = deadline - time.time()
                        if cv2.waitKey(int(min(remaining, 0.1) * 1000)) & 0xFF == ord("q"):
                            return 0
                        time.sleep(min(remaining, 0.1))
                else:
                    time.sleep(sleep_for)

            # Flush any internally buffered camera frames so the frame we
            # analyze was captured at the current moment, not pulled from a
            # stale backlog. Read and discard pending frames, then grab one.
            flushed = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                flushed += 1
                if flushed > 16:  # safety cap so a stuck camera can't loop forever
                    break
            ok, frame = cap.read()
            if not ok:
                retries += 1
                if retries > 5:
                    print("[uno-q-live] camera stopped returning frames; exiting")
                    break
                time.sleep(1.0)
                continue
            retries = 0
            last_run = time.time()
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            # Save the analyzed frame (the one captured at this moment) if
            # --save-frames was passed. Written with a timestamp filename.
            if args.save_frames is not None:
                args.save_frames.mkdir(parents=True, exist_ok=True)
                safe_stamp = stamp.replace(":", "-").replace(" ", "_")
                save_path = args.save_frames / f"frame_{safe_stamp}.jpg"
                try:
                    cv2.imwrite(str(save_path), frame)
                except cv2.error as exc:
                    print(f"[uno-q-live] could not save frame {save_path}: {exc}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            label, conf, class_index, previous = explain_frame(
                rgb, backend, verbalizer, args, stamp, previous=previous)
            if show:
                cv2.putText(frame, f"{label} {conf:.0%}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if show:
                if imshow("uno-q-live", frame):
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            classified += 1
            if args.one_shot:
                break
    except KeyboardInterrupt:
        print("\n[uno-q-live] interrupted")
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
