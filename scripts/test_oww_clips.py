#!/usr/bin/env python3
"""
Test an openWakeWord model against pre-recorded WAV clips.

Runs each clip through the model and reports detection results.
No microphone or robot needed.

Usage:
    # Test against positive clips (should detect)
    python scripts/test_oww_clips.py \
      --model path/to/okay_gambit.onnx \
      --clips training_data/positive/ \
      --threshold 0.5

    # Test against negative clips (should NOT detect)
    python scripts/test_oww_clips.py \
      --model path/to/okay_gambit.onnx \
      --clips training_data/negative/ \
      --threshold 0.5 \
      --expect-negative
"""

import argparse
import os
import sys
import wave

import numpy as np
import openwakeword
from openwakeword.model import Model

# Download openwakeword's bundled preprocessing models if missing
openwakeword.utils.download_models()


def read_wav(filepath: str) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (audio_int16, sample_rate)."""
    with wave.open(filepath, "rb") as wf:
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        audio = np.frombuffer(raw, dtype=np.int16)
        # Convert stereo to mono by averaging channels
        if wf.getnchannels() == 2:
            audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)
    return audio, sr


def test_clip(model: Model, model_name: str, audio: np.ndarray, threshold: float) -> tuple[bool, float]:
    """Run a clip through the model. Returns (detected, max_score)."""
    model.reset()

    # Feed audio in 1280-sample chunks (80ms at 16kHz) as OWW expects
    chunk_size = 1280
    max_score = 0.0

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        prediction = model.predict(chunk)
        score = prediction.get(model_name, 0.0)
        max_score = max(max_score, score)

    return max_score >= threshold, max_score


def main():
    parser = argparse.ArgumentParser(
        description="Test openWakeWord model against WAV clips",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to .onnx model file")
    parser.add_argument("--clips", required=True, help="Directory of WAV clips to test")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold (default: 0.5)")
    parser.add_argument("--expect-negative", action="store_true",
                        help="Clips should NOT trigger detection (for false positive testing)")
    parser.add_argument("--purge", type=float, default=None,
                        help="Delete clips scoring below this value (e.g., --purge 0.1)")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: model not found: {args.model}")
        sys.exit(1)

    if not os.path.isdir(args.clips):
        print(f"Error: directory not found: {args.clips}")
        sys.exit(1)

    # Load model
    model_path = os.path.abspath(args.model)
    oww_model = Model(wakeword_models=[model_path], inference_framework="onnx")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\nModel: {model_name}")
    print(f"Threshold: {args.threshold}")
    print(f"Mode: {'negative (expect no detections)' if args.expect_negative else 'positive (expect detections)'}")

    # Find clips
    clips = sorted([
        os.path.join(args.clips, f)
        for f in os.listdir(args.clips)
        if f.lower().endswith(".wav")
    ])

    if not clips:
        print(f"No WAV files found in {args.clips}")
        sys.exit(1)

    print(f"Clips: {len(clips)}")
    print(f"\n{'-' * 60}")

    detected = 0
    scores = []
    to_purge = []

    for clip_path in clips:
        filename = os.path.basename(clip_path)
        try:
            audio, sr = read_wav(clip_path)
            if sr != 16000:
                print(f"  SKIP  {filename} (sample rate {sr}, need 16000)")
                continue

            hit, score = test_clip(oww_model, model_name, audio, args.threshold)
            scores.append((score, clip_path))

            if hit:
                detected += 1
                print(f"  [HIT]  {score:.3f}  {filename}")
            else:
                print(f"  [MISS] {score:.3f}  {filename}")

            if args.purge is not None and score < args.purge:
                to_purge.append(clip_path)

        except Exception as e:
            print(f"  [ERR]  {filename}: {e}")

    # Summary
    total = len(scores)
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Clips tested:  {total}")
    print(f"  Detected:      {detected}")
    print(f"  Not detected:  {total - detected}")

    if args.expect_negative:
        fp_rate = (detected / total * 100) if total else 0
        print(f"\n  False positive rate: {fp_rate:.1f}%")
        if detected == 0:
            print(f"  PASS — no false positives")
        else:
            print(f"  FAIL — {detected} false positive(s)")
    else:
        recall = (detected / total * 100) if total else 0
        print(f"\n  Recall: {recall:.1f}%")
        if recall >= 90:
            print(f"  PASS — recall >= 90%")
        else:
            print(f"  NEEDS IMPROVEMENT — recall < 90%")

    if scores:
        score_vals = [s[0] for s in scores]
        print(f"\n  Score stats:")
        print(f"    Min:    {min(score_vals):.3f}")
        print(f"    Max:    {max(score_vals):.3f}")
        print(f"    Mean:   {np.mean(score_vals):.3f}")
        print(f"    Median: {np.median(score_vals):.3f}")

    print()


if __name__ == "__main__":
    main()
