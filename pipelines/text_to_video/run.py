#!/usr/bin/env python3
"""Text-to-Video pipeline runner (CogVideoX-5b).

Usage:
    # Text → video
    python pipelines/text_to_video/run.py output.mp4 [workdir] \
        --text "A golden retriever running on a beach at sunset, cinematic, 4k"

    # Image + prompt → video (image-to-video)
    python pipelines/text_to_video/run.py output.mp4 [workdir] \
        --text "slow zoom in, gentle wind" --image first_frame.png

Steps:
    1. CogVideoX-5b generation (text→video, or image→video if --image given)
    2. Optional aspect-ratio scale/pad
    3. Optional Real-ESRGAN upscale (--upscale 2)
"""
import sys
import os
import argparse
import logging
import shutil
import subprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Add repo root to path so core/ is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from core.steps.text_to_video import generate_video_from_text, generate_video_from_image
from core.steps.video_sr import upscale_video

ASPECT_SIZES = {
    "9:16": (720, 1280),
    "1:1": (720, 720),
    "16:9": (1280, 720),
    "4:5": (720, 900),
    "4:3": (960, 720),
}


def step_done(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1000


def _scale_to_aspect(input_path: str, output_path: str, aspect: str):
    target_w, target_h = ASPECT_SIZES[aspect]
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vf", (f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=black"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        output_path,
    ], check=True, capture_output=True)


def run(output_path: str, workdir: str, text: str,
        image: str = None, aspect: str = None, upscale: int = 1,
        num_frames: int = 49, steps: int = 50, guidance: float = 6.0, fps: int = 8):

    os.makedirs(workdir, exist_ok=True)
    raw = os.path.join(workdir, "cogvideo_raw.mp4")

    # ── Step 1: Generate ──────────────────────────────────────────────────────
    if not step_done(raw):
        if image:
            log.info("Step 1 — CogVideoX image-to-video")
            generate_video_from_image(image, text, raw, num_frames=num_frames,
                                       steps=steps, guidance=guidance, fps=fps)
        else:
            log.info("Step 1 — CogVideoX text-to-video")
            generate_video_from_text(text, raw, num_frames=num_frames,
                                      steps=steps, guidance=guidance, fps=fps)
    else:
        log.info("Step 1 — skipped (cached)")

    current = raw

    # ── Step 2: Aspect scale/pad (optional) ───────────────────────────────────
    if aspect:
        scaled = os.path.join(workdir, "scaled.mp4")
        log.info(f"Step 2 — scale to aspect {aspect}")
        _scale_to_aspect(current, scaled, aspect)
        current = scaled

    # ── Step 3: Upscale (optional) ────────────────────────────────────────────
    if upscale > 1:
        log.info(f"Step 3 — Real-ESRGAN {upscale}x")
        try:
            upscale_video(current, output_path, upscale=upscale)
            log.info(f"Done: {output_path}")
            return
        except Exception as e:
            log.warning(f"Upscale failed (non-fatal): {e}")

    shutil.copy(current, output_path)
    log.info(f"Done: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("workdir", nargs="?", default="/tmp/text_to_video_run")
    parser.add_argument("--text", required=True, help="Generation prompt")
    parser.add_argument("--image", default=None, help="Optional source image for image-to-video mode")
    parser.add_argument("--aspect", default=None, choices=list(ASPECT_SIZES.keys()),
                        help="Optional output aspect ratio (scale + pad)")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2],
                        help="Real-ESRGAN super-resolution factor (1 = off, 2 = 2x)")
    parser.add_argument("--frames", type=int, default=49, dest="num_frames")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=6.0)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    run(args.output, args.workdir, text=args.text, image=args.image,
        aspect=args.aspect, upscale=args.upscale, num_frames=args.num_frames,
        steps=args.steps, guidance=args.guidance, fps=args.fps)
