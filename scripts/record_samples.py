#!/usr/bin/env python3
"""
Record wake word training samples from your microphone.

Records short WAV clips (16kHz, 16-bit, mono) suitable for training
an openWakeWord model via the Colab notebook.

Two modes:
  - positive: Record wake word utterances (1-2s each, press Enter between)
  - negative: Record background noise / non-wake-word speech (longer clips)

Usage:
    # Record 50 positive wake word samples
    python scripts/record_samples.py positive --count 50

    # Record negative samples (background noise, 10s each)
    python scripts/record_samples.py negative --clip-duration 10 --count 10

    # Record with a specific label prefix
    python scripts/record_samples.py positive --count 20 --label far_noisy

    # List available microphones
    python scripts/record_samples.py list-devices
"""

import argparse
import os
import sys
import time
import wave

import subprocess

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("Error: sounddevice is required. Install with:")
    print("  pip install sounddevice")
    sys.exit(1)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training_data",
)


def list_devices():
    """Print available audio input devices."""
    print("\nAvailable audio input devices:")
    print("-" * 60)
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            default = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {dev['name']}{default}")
            print(f"       Channels: {dev['max_input_channels']}, "
                  f"Sample Rate: {dev['default_samplerate']:.0f} Hz")
    print()


def save_wav(filepath: str, audio: np.ndarray):
    """Save audio as 16kHz 16-bit mono WAV."""
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())


def record_clip(duration: float, device=None) -> np.ndarray:
    """Record a single audio clip and return as int16 numpy array."""
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        device=device,
    )
    sd.wait()
    return audio.flatten()


def record_positive_samples(
    output_dir: str,
    count: int,
    clip_duration: float,
    label: str,
    device,
    subfolder: str = "positive",
):
    """Record positive (wake word) samples interactively."""
    pos_dir = os.path.join(output_dir, subfolder)
    os.makedirs(pos_dir, exist_ok=True)

    # Find highest existing number to continue numbering
    existing = [f for f in os.listdir(pos_dir) if f.endswith(".wav")]
    max_num = 0
    for f in existing:
        parts = f.rsplit("_", 1)
        if len(parts) == 2:
            try:
                max_num = max(max_num, int(parts[1].split(".")[0]))
            except ValueError:
                pass
    start_num = max_num + 1

    print(f"\n{'=' * 60}")
    print("  POSITIVE SAMPLE RECORDING")
    print(f"{'=' * 60}")
    print(f"\n  Output:    {pos_dir}")
    print(f"  Target:    {count} clips ({start_num - 1} existing)")
    print(f"  Duration:  {clip_duration}s per clip")
    print(f"  Format:    {SAMPLE_RATE} Hz, 16-bit, mono")
    print(f"\n  Press ENTER to record each sample.")
    print("  Type 'q' to stop early.")
    print("  Say the wake word clearly after the countdown.")
    print(f"\n{'-' * 60}\n")

    recorded = 0
    for i in range(start_num, start_num + count):
        resp = input(f"  [{i}/{start_num + count - 1}] Press ENTER to record (q to quit): ")
        if resp.strip().lower() == "q":
            break

        subprocess.Popen(
            ["afplay", "/System/Library/Sounds/Glass.aiff"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(0.7)
        print("RECORDING!", flush=True)

        audio = record_clip(clip_duration, device=device)

        # Check audio level
        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if peak < 500:
            print(f"    WARNING: Very quiet recording (peak={peak}). "
                  "Check your microphone.")

        filename = f"wake_{label}_{i:04d}.wav"
        filepath = os.path.join(pos_dir, filename)
        save_wav(filepath, audio)
        recorded += 1
        print(f"    Saved: {filename} (peak={peak}, rms={rms:.0f})")

    print(f"\n  Done! Recorded {recorded} positive samples.")
    print(f"  Total positive samples: {len(os.listdir(pos_dir))}")


def record_negative_samples(
    output_dir: str,
    count: int,
    clip_duration: float,
    label: str,
    device,
):
    """Record negative (non-wake-word) samples."""
    neg_dir = os.path.join(output_dir, "negative")
    os.makedirs(neg_dir, exist_ok=True)

    existing = [f for f in os.listdir(neg_dir) if f.endswith(".wav")]
    max_num = 0
    for f in existing:
        parts = f.rsplit("_", 1)
        if len(parts) == 2:
            try:
                max_num = max(max_num, int(parts[1].split(".")[0]))
            except ValueError:
                pass
    start_num = max_num + 1

    print(f"\n{'=' * 60}")
    print("  NEGATIVE SAMPLE RECORDING")
    print(f"{'=' * 60}")
    print(f"\n  Output:    {neg_dir}")
    print(f"  Target:    {count} clips ({start_num - 1} existing)")
    print(f"  Duration:  {clip_duration}s per clip")
    print(f"  Format:    {SAMPLE_RATE} Hz, 16-bit, mono")
    print(f"\n  Press ENTER to start each recording.")
    print("  Type 'q' to stop early.")
    print("  DO NOT say the wake word! Record:")
    print("    - General conversation")
    print("    - Similar-sounding words")
    print("    - Background noise (fan, music, TV)")
    print(f"\n{'-' * 60}\n")

    recorded = 0
    for i in range(start_num, start_num + count):
        resp = input(f"  [{i}/{start_num + count - 1}] Press ENTER to record (q to quit): ")
        if resp.strip().lower() == "q":
            break

        print(f"    RECORDING {clip_duration}s...", end=" ", flush=True)
        audio = record_clip(clip_duration, device=device)

        peak = np.max(np.abs(audio))
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))

        filename = f"neg_{label}_{i:04d}.wav"
        filepath = os.path.join(neg_dir, filename)
        save_wav(filepath, audio)
        recorded += 1
        print(f"done! Saved: {filename} (peak={peak}, rms={rms:.0f})")

    print(f"\n  Done! Recorded {recorded} negative samples.")
    print(f"  Total negative samples: {len(os.listdir(neg_dir))}")


def main():
    parser = argparse.ArgumentParser(
        description="Record wake word training samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Record 50 positive wake word samples
  python scripts/record_samples.py positive --count 50

  # Record 10 negative clips of 10s each
  python scripts/record_samples.py negative --clip-duration 10 --count 10

  # Record with a label (e.g., distance/condition)
  python scripts/record_samples.py positive --count 20 --label far_noisy

  # Use a specific microphone
  python scripts/record_samples.py list-devices
  python scripts/record_samples.py positive --device 2 --count 50
""",
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # list-devices subcommand
    subparsers.add_parser("list-devices", help="List available audio input devices")

    # positive subcommand
    pos_parser = subparsers.add_parser("positive", help="Record wake word samples")
    pos_parser.add_argument(
        "--count", type=int, default=50,
        help="Number of clips to record (default: 50)",
    )
    pos_parser.add_argument(
        "--clip-duration", type=float, default=2.0,
        help="Duration per clip in seconds (default: 2.0)",
    )
    pos_parser.add_argument(
        "--label", default="default",
        help="Label prefix for filenames (e.g., 'close_quiet', 'far_noisy')",
    )
    pos_parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (use list-devices to find)",
    )
    pos_parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    # eval subcommand
    eval_parser = subparsers.add_parser("eval", help="Record eval clips (kept separate from training data)")
    eval_parser.add_argument(
        "--count", type=int, default=20,
        help="Number of clips to record (default: 20)",
    )
    eval_parser.add_argument(
        "--clip-duration", type=float, default=2.0,
        help="Duration per clip in seconds (default: 2.0)",
    )
    eval_parser.add_argument(
        "--label", default="eval",
        help="Label prefix for filenames (default: 'eval')",
    )
    eval_parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (use list-devices to find)",
    )
    eval_parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    # negative subcommand
    neg_parser = subparsers.add_parser("negative", help="Record negative samples")
    neg_parser.add_argument(
        "--count", type=int, default=10,
        help="Number of clips to record (default: 10)",
    )
    neg_parser.add_argument(
        "--clip-duration", type=float, default=10.0,
        help="Duration per clip in seconds (default: 10.0)",
    )
    neg_parser.add_argument(
        "--label", default="default",
        help="Label prefix for filenames (e.g., 'conversation', 'music')",
    )
    neg_parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (use list-devices to find)",
    )
    neg_parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )

    args = parser.parse_args()

    if args.mode == "list-devices":
        list_devices()
        return

    if args.mode == "positive":
        record_positive_samples(
            output_dir=args.output_dir,
            count=args.count,
            clip_duration=args.clip_duration,
            label=args.label,
            device=args.device,
        )
    elif args.mode == "eval":
        eval_dir = os.path.join(args.output_dir, "eval")
        os.makedirs(eval_dir, exist_ok=True)
        record_positive_samples(
            output_dir=args.output_dir,
            count=args.count,
            clip_duration=args.clip_duration,
            label=args.label,
            device=args.device,
            subfolder="eval",
        )
    elif args.mode == "negative":
        record_negative_samples(
            output_dir=args.output_dir,
            count=args.count,
            clip_duration=args.clip_duration,
            label=args.label,
            device=args.device,
        )


if __name__ == "__main__":
    main()
