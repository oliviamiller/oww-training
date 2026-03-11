# Training a Custom Wake Word with OpenWakeWord

This guide walks through the full process of training and evaluting
an openwakeword model that can be used with the filtered-audio module.

---

## Quick option: Automatic training

OpenWakeWord provides an [automatic training notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb?usp=sharing) that can train a model using only synthetic TTS samples — no recording required. This is a good starting point if you want to test a wake word quickly or are okay with lower accuracy. For best results, follow the full process below.

---

## Overview

The scripts and training notebook referenced are available at [github.com/oliviamiller/oww-training](https://github.com/oliviamiller/oww-training).

The pipeline has three phases:
1. **Record positive samples** locally
2. **Train the model** in Google Colab using `training.ipynb`
3. **Evaluate the model** locally using `test_oww_clips.py`

The training notebook generates thousands of synthetic TTS samples of your target phrase, mixes in your real recordings, applies audio augmentation, and trains a model that outputs a confidence score.

---

## Record Samples

Clone [oww-training](https://github.com/oliviamiller/oww-training) and run the following commands from within it. Samples are saved to `training_data/`.

### Positive samples

Positive samples are real recordings of your wake word. Record as many as possible — the more the better.

```bash
python3 record_samples.py positive --count 100
```

- Press **Enter** to start each clip, which records for ~1.5 seconds
- Say the wake word clearly and naturally — vary pace, tone, and distance
- Files are saved to `training_data/positive/`

### Evaluation samples

Eval clips are kept separate from training data and used only for final testing. Record these in a fresh session, ideally in a different environment and from different people you collected the training data from.

```bash
python3 record_samples.py eval --count 50 --label eval
```

Files are saved to `training_data/eval/`.

## Train the Model

Training runs in Google Colab. You can purchase credits for higher-tier GPU usage to make it complete faster. The notebook handles all dependency installation, dataset downloads, synthetic sample generation, augmentation, and model training.

### Setup

1. Upload the `training_data/positive/` folder and `training.ipynb` to Google Drive
2. Open `training.ipynb` in Google Colab
3. Set the runtime to **GPU** (Runtime → Change runtime type → T4 GPU)

### Configure the notebook

At the top of the notebook, edit these variables:

| Variable | Description | Default |
|---|---|---|
| `target_word` | Your wake word/phrase (underscores for spaces) |
| `number_of_examples` | Synthetic TTS samples to generate, recommended 50,000
| `number_of_training_steps` | Training steps | recommended 10,0000
| `false_activation_penalty` | Higher = fewer false positives, but may reduce recall | recommended 400
| `recompute_features` | Set to `True` the first time you train or any time you add/change your audio samples — this recomputes the audio features from scratch. Set to `False` to reuse previously computed features and save time when only changing training parameters (e.g. steps or penalty).

### Run the notebook

Run the cell. The pipeline executes these steps:

1. **Install dependencies** — openWakeWord, piper-sample-generator, SpeechBrain, audiomentations, etc.
2. **Download datasets** — AudioSet and FMA (background audio), MIT Room Impulse Responses for negative training data
3. **Load config** — generates `my_model.yaml` from the template with your settings
4. **Generate synthetic clips** — TTS creates thousands of speech variations of your target phrase using the `en_US-libritts_r-medium` voice model
5. **Add real recordings** — your positive samples from `training_data/positive/` are copied in with a `real_` prefix
6. **Augment clips** — applies pitch shift, time stretch, gain variation, and room impulse responses to simulate diverse acoustic conditions
7. **Train** — trains the neural network against the augmented positive clips and background negatives
8. **Export** — converts the trained model to both `.onnx` and `.tflite` formats

### Download the trained model

After training completes, download the files generated.
The notebook also saves these to Google Drive automatically.

**Typical training time: 1–3 hours on a T4 GPU.**

---

## Evaluate the Model

Use `test_oww_clips.py` to evaluate your model against pre-recorded clips without needing a microphone.

The script processes audio in 80ms chunks (1280 samples at 16 kHz) and reports detection results.

### Test against positive samples

```bash
python3 test_oww_clips.py \
  --model path/to/model.tflite \
  --clips training_data/positive/ \
  --threshold 0.5
```

If recall is low (< 80%) consider recording more samples or lowering `false_activation_penalty` and retraining.

### Test against evaluation samples

```bash
python3 test_oww_clips.py \
  --model path/to/file \
  --clips training_data/eval/ \
  --threshold 0.5
```

Eval clips were never seen during training, so this gives a fair accuracy assessment.

### Test for false positives (optional)

If you have negative recordings (non-wake-word speech/noise):

```bash
python3 test_oww_clips.py \
  --model path/to/okay_gambit.tflite \
  --clips training_data/negative/ \
  --threshold 0.5 \
  --expect-negative
```

Expected: false positive rate near 0%.

### Remove low-quality clips

If evaluation reveals clips that score poorly or unexpectedly, use `--purge` to delete them and re-run training:

```bash
python3 test_oww_clips.py \
  --model path/to/okay_gambit.tflite \
  --clips training_data/positive/ \
  --threshold 0.5 \
  --purge
```

---

