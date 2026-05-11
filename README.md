# clap~

> **Experimental work — vibe-coded.**
> This external is a research prototype developed at IRCAM (RepMus / REACH team). It is not production software. Expect rough edges, missing documentation, and breaking changes. Use at your own risk.

Real-time zero-shot audio classification for Max/MSP, powered by [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused).

The object listens to incoming audio, segments it into fixed-length windows, and classifies each window against a set of class prototypes using CLAP (Contrastive Language-Audio Pretraining). Class prototypes can be text descriptions, audio examples, or a mix of both.

**Contributors:** Nicolas Brochec, Claude Sonnet (Anthropic)

---

## Requirements

- macOS, Apple Silicon (arm64)
- Max 8 or later
- A exported model file — see `scripts/export_clap.py` (TorchScript) or `scripts/export_clap_onnx.py` (ONNX)

---

## Object structure

### Arguments

```
clap~ <model_path> [device]
```

| Argument | Type | Required | Description |
|---|---|---|---|
| `model_path` | symbol | yes | Path to `clap_tilde.ts` or to the directory containing the ONNX model files. `vocab.json` and `merges.txt` must be in the same directory. |
| `device` | symbol | no | `cpu` (default) or `mps`. MPS enables Metal acceleration via CoreML EP (ONNX backend only — TorchScript falls back to CPU with a warning). |

---

### Inlets

| Inlet | Type | Description |
|---|---|---|
| 1 (left) | signal | Audio input — mono, any sample rate (internally resampled to 48 kHz). |
| 2 (right) | list | Few-shot registration messages (`record`, see below). |

---

### Outlets

| Outlet | Type | Description |
|---|---|---|
| 1 (left) | int | Index of the winning class (0-based). Only sent when confidence is above the `confidence` threshold. |
| 2 | symbol | Name of the winning class. |
| 3 | list | Full probability distribution over all classes (floats, one per class, sum to 1). |
| 4 (dumpout) | list | Diagnostic output. Currently outputs `latency <ms>` after each inference cycle. |

---

### Attributes

| Attribute | Type | Default | Description |
|---|---|---|---|
| `enabled` | int (0/1) | 1 | Enable or disable inference without stopping DSP. |
| `threshold` | float | −120 dB | Energy threshold in dB. Audio below this level is ignored (gate). |
| `window` | int | 500 ms | Duration of the energy threshold look-back window in milliseconds. |
| `confidence` | float 0–1 | 0.0 | Minimum winning-class probability to output a result. Below this, outlets are silent. |
| `sensitivity` | float 0–1 | 1.0 | Smoothing amount applied to the probability distribution over time (leaky integrator). 0 = maximum smoothing, 1 = no smoothing. |
| `sensitivityrange` | int (ms) | 2000 | Maximum time constant for the leaky integrator in milliseconds. Scales the range of `sensitivity`. |
| `verbose` | int (0/1) | 0 | Enable verbose logging to the Max console. |

---

### Messages (inlet 1)

#### `set_classes <name1> <name2> ...`

Set the text class prototypes. Each atom is one class name (use quoted symbols for multi-word names). The model encodes each name as a text embedding on the next inference cycle.

```
; example
set_classes drums hi-hat voice bass
```

Sending new classes replaces the previous set. Audio examples registered with `record` whose label matches a class name will override the text embedding for that label.

#### `classnames`

Outputs the current active class names (text + audio) to the dumpout outlet as `classnames <name1> <name2> ...`.

---

### Messages (inlet 2)

#### `record <label> <buffer_name>`

Register an audio example for a class. The object reads the named `buffer~`, resamples it to 48 kHz if needed, and encodes it as an audio embedding. On the next inference cycle, this embedding replaces the text embedding for `<label>` (or adds a new class if the label is not in the current text class set).

```
; example — in a message box connected to inlet 2
record kick my_kick_buffer
```

- The buffer can be any sample rate and any length (only channel 0 is used).
- Multiple `record` calls with the same label overwrite the previous example.
- Audio-only use is supported: you can skip `set_classes` entirely and register all classes via `record`.

#### `clear_example <label>`

Remove the audio example for a single label. The class reverts to its text embedding if one was set via `set_classes`.

#### `clear_examples`

Remove all registered audio examples.

---

## Workflow

### Zero-shot (text only)

```
1. [clap~ /path/to/clap_tilde.ts]
2. Send: set_classes dog bark cat meow rain thunder
3. Connect audio to inlet 1 — classification starts automatically.
```

### Few-shot (audio examples)

```
1. [clap~ /path/to/model]
2. Send to inlet 1: set_classes label1 label2
3. Load audio into buffer~, then send to inlet 2: record label1 my_buffer
4. Repeat for label2.
5. Classification now uses audio prototypes instead of text descriptions.
```

### Audio-only (no text)

```
1. [clap~ /path/to/model]
2. (skip set_classes)
3. Send to inlet 2: record label1 buf1
4. Send to inlet 2: record label2 buf2
5. Inference runs using only audio prototypes.
```

---

## Architecture

```
Audio inlet (signal)
    │
    ▼
m_audio_fifo  (lock-free ring buffer)
    │
    ▼ (background inference thread)
ClapClassifier::process()
    ├── apply_pending_classes()   — encode new text labels via model
    ├── apply_pending_audio()     — encode new buffer~ examples via model
    ├── build_combined()          — merge text + audio prototypes into [N, 512] matrix
    ├── EnergyThreshold           — gate: skip silent frames
    └── IClapModel::classify()    — audio embedding → cosine similarity → softmax
            │
            ▼
    m_event_fifo  (lock-free ring buffer)
            │
            ▼ (Max scheduler thread, via timer)
    deliverer → outlets
```

**IClapModel** is an abstract interface with two concrete backends:

| Backend | File | Notes |
|---|---|---|
| TorchScript | `src/clap_model.h` | Mel preprocessing in Python (baked into the `.ts`). CPU only. |
| ONNX | `src/clap_model_onnx.h` | Mel preprocessing in C++ (LibTorch). Supports CoreML EP (MPS). |

Model export scripts are in `scripts/`.

---

## Known limitations

- macOS arm64 only (no Intel, no Windows, no Linux).
- TorchScript backend does not support MPS (torch.stft is CPU-only).
- No multi-channel input — only channel 0 is used.
- Audio examples use a single embedding per label (last `record` call wins); no averaging across multiple examples yet.
- Experimental software — no guarantee of stability across Max versions or macOS updates.
