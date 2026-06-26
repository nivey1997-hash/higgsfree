#!/usr/bin/env python3
"""End-to-end HeyGen-style pipeline test. Run on EC2 (not locally).

Correct pipeline (exactly what HeyGen does):
  consent video → best frame (their appearance) + audio (their voice)
  → F5-TTS voice clone profile
  → new script → TTS with cloned voice → speech.wav
  → best frame + speech.wav → EchoMimicV2 (audio-driven talking head)
  → final_video.mp4

Usage:
  python test_pipeline.py --video /tmp/consent.mp4
  python test_pipeline.py --video /tmp/consent.mp4 --text "Hello! I'm your AI presenter."
  python test_pipeline.py --video /tmp/consent.mp4 --skip-voiceclone --audio /tmp/speech.wav
  python test_pipeline.py --video /tmp/consent.mp4 --skip-echomimic  # test just voice clone
"""
import argparse
import os
import sys
import time
import logging
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

# Must set LD_LIBRARY_PATH BEFORE importing torch/onnxruntime/insightface
# so CUDAExecutionProvider loads correctly (needs CUDA 12 libs from nvidia packages)
from pipeline.gpu_env import gpu_env as _gpu_env
_env = _gpu_env()
os.environ["LD_LIBRARY_PATH"] = _env["LD_LIBRARY_PATH"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SCRIPT = (
    "Hello! I am your AI presenter. "
    "Today, we will talk about how you can grow your business using digital tools. "
    "Let's get started!"
)


def stage(name):
    log.info(f"\n{'='*55}\n  STAGE: {name}\n{'='*55}")
    return time.time()


def done(t0, output):
    log.info(f"  ✓ done in {time.time()-t0:.1f}s → {output}")
    return output


def extract_audio_from_video(video_path: str, out_wav: str):
    """Extract first 30s of audio for voice profile."""
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-t", "30",
        "-vn", "-acodec", "pcm_s16le", "-ar", "24000", "-ac", "1",
        out_wav,
    ], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to consent video (your selfie/speaking video)")
    ap.add_argument("--text", default=DEFAULT_SCRIPT, help="Script text to synthesize and animate")
    ap.add_argument("--audio", default=None, help="Skip voice clone — use this WAV directly")
    ap.add_argument("--skip-voiceclone", action="store_true", help="Use generic TTS instead of voice clone")
    ap.add_argument("--skip-echomimic", action="store_true", help="Stop after voice clone (skip animation)")
    ap.add_argument("--lipsync-only", action="store_true", help="Use LatentSync directly on portrait (skip EchoMimicV2 — sharper, faster)")
    ap.add_argument("--pose-dir", default=None, help="Pre-extracted pose dir. If not given, uses EchoMimicV2 template '01'")
    ap.add_argument("--pose-template", default="01", help="EchoMimicV2 built-in pose template (01/02/03/04)")
    ap.add_argument("--outdir", default="/tmp/avatar_test")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log.info(f"Output dir: {args.outdir}")
    log.info(f"Script: {args.text[:100]}...")

    # ── Stage 1: Extract portrait frame (768x768 square, face+shoulders) ──
    t = stage("1. Extract portrait frame (face + shoulders, 768x768)")
    from pipeline.avatar_gen import extract_portrait_frame
    face_jpg = os.path.join(args.outdir, "portrait.jpg")
    extract_portrait_frame(args.video, face_jpg)
    done(t, face_jpg)

    # ── Stage 2: Voice clone (or fallback to Kokoro TTS) ─────────────────
    speech_wav = os.path.join(args.outdir, "speech.wav")

    if args.audio:
        import shutil
        shutil.copy(args.audio, speech_wav)
        log.info(f"Using provided audio: {args.audio}")

    elif args.skip_voiceclone:
        t = stage("2. TTS (Kokoro generic — no voice clone)")
        from pipeline.tts import synthesize
        audio_bytes = synthesize(args.text)
        with open(speech_wav, "wb") as f:
            f.write(audio_bytes)
        done(t, speech_wav)

    else:
        # ── 2a. Extract reference audio from consent video ────────────────
        t = stage("2a. Extract reference audio from consent video")
        ref_wav = os.path.join(args.outdir, "ref_audio_raw.wav")
        extract_audio_from_video(args.video, ref_wav)
        done(t, ref_wav)

        # ── 2b. Build voice profile (Whisper transcription + resample) ────
        t = stage("2b. Build voice profile (Whisper + F5-TTS)")
        from pipeline.voiceclone import extract_voice_profile
        voice_profile_dir = os.path.join(args.outdir, "voice_profile")
        extract_voice_profile(ref_wav, voice_profile_dir)
        done(t, voice_profile_dir)

        # ── 2c. Synthesize new script in cloned voice ─────────────────────
        t = stage("2c. Synthesize cloned voice TTS")
        from pipeline.voiceclone import synthesize_with_cloned_voice
        synthesize_with_cloned_voice(args.text, voice_profile_dir, speech_wav)
        done(t, speech_wav)

    if args.skip_echomimic:
        log.info("\n" + "="*55)
        log.info("  PIPELINE STOPPED (--skip-echomimic)")
        log.info(f"  face:    {face_jpg}")
        log.info(f"  speech:  {speech_wav}")
        log.info("="*55)
        return

    # ── LatentSync-only mode: portrait + audio → lip-synced video ────────
    if args.lipsync_only:
        t = stage("3. LatentSync (portrait + audio → lip-synced video, no body animation)")
        from pipeline.lipsync import run_lipsync_on_image
        final_mp4 = os.path.join(args.outdir, "final.mp4")
        run_lipsync_on_image(face_jpg, speech_wav, final_mp4)
        done(t, final_mp4)
        log.info("\n" + "="*55)
        log.info("  PIPELINE COMPLETE (LatentSync mode)")
        log.info(f"  face:    {face_jpg}")
        log.info(f"  speech:  {speech_wav}")
        log.info(f"  final:   {final_mp4}")
        log.info("="*55)
        log.info(f"\nDownload:\n  scp -i ~/.ssh/graperoot-bench.pem ubuntu@<EC2_IP>:{final_mp4} ~/result.mp4")
        return

    # ── Stage 3: Extract pose from YOUR consent video ────────────────────
    pose_dir = args.pose_dir
    echomimic_dir = os.environ.get("ECHOMIMIC_DIR", "/home/ubuntu/echomimic_v2")
    if not pose_dir:
        t = stage("3. Extract pose from consent video (DWPose — your body/gestures)")
        from pipeline.pose_extract_echomimic import extract_pose_for_echomimic
        pose_dir = os.path.join(args.outdir, "pose_frames")
        extract_pose_for_echomimic(args.video, pose_dir, max_frames=300)
        done(t, pose_dir)
    else:
        log.info(f"  Using pre-extracted pose dir: {pose_dir}")

    # ── Stage 4: EchoMimicV2 — audio-driven talking head ─────────────────
    t = stage("4. EchoMimicV2 (audio-driven talking head animation)")
    from pipeline.echomimic import run_echomimic_with_pose_dir
    animated_mp4 = os.path.join(args.outdir, "animated.mp4")
    run_echomimic_with_pose_dir(face_jpg, speech_wav, pose_dir, animated_mp4)
    done(t, animated_mp4)

    # ── Stage 5: Face composite — paste sharp face onto static body ───────
    t = stage("5. Face composite (animated face on sharp reference body)")
    from pipeline.face_composite import composite_face_onto_reference
    final_mp4 = os.path.join(args.outdir, "final.mp4")
    composite_face_onto_reference(animated_mp4, face_jpg, final_mp4)
    done(t, final_mp4)

    log.info("\n" + "="*55)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  face:    {face_jpg}")
    log.info(f"  speech:  {speech_wav}")
    log.info(f"  animated: {animated_mp4}")
    log.info(f"  final:   {final_mp4}")
    log.info("="*55)
    log.info(f"\nDownload:\n  scp -i ~/.ssh/graperoot-bench.pem ubuntu@<EC2_IP>:{final_mp4} ~/result.mp4")


if __name__ == "__main__":
    main()
