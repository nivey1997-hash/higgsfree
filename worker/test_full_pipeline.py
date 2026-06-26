#!/usr/bin/env python3
"""End-to-end pipeline test: consent video → PuLID portrait → img2img → Sonic → CodeFormer.

Usage on EC2:
  python test_full_pipeline.py --video /home/ubuntu/shiv_singh.mp4
  python test_full_pipeline.py --video /home/ubuntu/shiv_singh.mp4 --text "Custom script here"
  python test_full_pipeline.py --video /home/ubuntu/shiv_singh.mp4 --skip-portrait  # skip to Sonic if portrait done
"""
import argparse
import os
import sys
import time
import logging
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

from pipeline.gpu_env import gpu_env as _gpu_env
_env = _gpu_env()
os.environ["LD_LIBRARY_PATH"] = _env["LD_LIBRARY_PATH"]

# Kill the SQS worker so it doesn't auto-stop the instance mid-pipeline
subprocess.run(["pkill", "-f", "worker.py"], capture_output=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_SCRIPT = (
    "Hello! I am your AI presenter. "
    "Today, we will explore how technology is changing the world around us. "
    "Let me walk you through the key highlights."
)


def stage(name):
    log.info(f"\n{'='*60}\n  STAGE: {name}\n{'='*60}")
    return time.time()


def done(t0, output):
    elapsed = time.time() - t0
    log.info(f"  done in {elapsed:.1f}s → {output}")
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Consent video path (mp4/webm)")
    ap.add_argument("--text", default=DEFAULT_SCRIPT, help="Script text to animate")
    ap.add_argument("--outdir", default="/home/ubuntu/pipeline_test_out")
    ap.add_argument("--face", default=None,
                    help="Use this image directly as face frame (skip video extraction)")
    ap.add_argument("--skip-portrait", action="store_true",
                    help="Skip PuLID+img2img, use existing avatar.png in outdir")
    ap.add_argument("--skip-voiceclone", action="store_true",
                    help="Skip Chatterbox, use generic TTS")
    ap.add_argument("--id-scale", type=float, default=0.8, help="PuLID id_scale (default 0.8)")
    ap.add_argument("--img2img-strength", type=float, default=0.2,
                    help="RealVisXL img2img strength (default 0.2)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    log.info(f"Output dir: {args.outdir}")
    log.info(f"Script: {args.text[:80]}...")

    face_jpg     = os.path.join(args.outdir, "face.jpg")
    portrait_png = os.path.join(args.outdir, "portrait_pulid.png")
    refined_png  = os.path.join(args.outdir, "portrait_refined.png")
    avatar_png   = os.path.join(args.outdir, "avatar.png")
    speech_wav   = os.path.join(args.outdir, "speech.wav")
    sonic_mp4    = os.path.join(args.outdir, "sonic_out.mp4")
    final_mp4    = os.path.join(args.outdir, "final.mp4")

    # ── Stage 1: Face extraction ─────────────────────────────────────────────
    if not args.skip_portrait:
        if args.face:
            # Use provided face image directly — skip video frame extraction
            import shutil
            shutil.copy(args.face, face_jpg)
            t = stage("1. Using provided face image (skipping video extraction)")
            done(time.time(), face_jpg)
        else:
            t = stage("1. Extract best face frame")
            from pipeline.avatar_gen import extract_best_face_frame
            extract_best_face_frame(args.video, face_jpg)
            done(t, face_jpg)

        # ── Stage 2: PuLID portrait ──────────────────────────────────────────
        t = stage("2. PuLID portrait (id_scale={}, 4 steps, SDXL-Lightning)".format(args.id_scale))
        from pipeline.avatar_gen import generate_pulid_portrait
        generate_pulid_portrait(face_jpg, portrait_png, id_scale=args.id_scale)
        done(t, portrait_png)

        # ── Stage 3: RealVisXL img2img realism ──────────────────────────────
        t = stage("3. RealVisXL img2img refinement (strength={})".format(args.img2img_strength))
        from pipeline.avatar_gen import refine_portrait_img2img
        try:
            refine_portrait_img2img(portrait_png, refined_png, strength=args.img2img_strength)
            source_for_crop = refined_png
            done(t, refined_png)
        except Exception as e:
            log.warning(f"img2img failed ({e}), using raw PuLID portrait")
            source_for_crop = portrait_png

        # ── Stage 4: Crop head+shoulders ─────────────────────────────────────
        t = stage("4. Crop head+shoulders (9:16 for Sonic)")
        from pipeline.avatar_gen import crop_head_shoulders
        crop_head_shoulders(source_for_crop, avatar_png)
        done(t, avatar_png)

    else:
        log.info("Skipping portrait stages — using existing avatar.png")
        if not os.path.exists(avatar_png):
            raise FileNotFoundError(f"--skip-portrait used but {avatar_png} not found")

    # ── Stage 5: Voice profile + TTS ─────────────────────────────────────────
    t = stage("5. Voice profile extraction (Chatterbox, first 30s)")
    voice_profile_dir = os.path.join(args.outdir, "voice_profile")
    if not args.skip_voiceclone:
        try:
            from pipeline.voiceclone import extract_voice_profile
            extract_voice_profile(args.video, voice_profile_dir)
            done(t, voice_profile_dir)
        except Exception as e:
            log.warning(f"Voice profile failed ({e}), will use generic TTS")
            voice_profile_dir = None
    else:
        voice_profile_dir = None
        log.info("Skipping voice clone (--skip-voiceclone)")

    t = stage("5b. TTS synthesis")
    if voice_profile_dir and os.path.exists(os.path.join(voice_profile_dir, "ref_audio.wav")):
        from pipeline.voiceclone import synthesize_with_cloned_voice
        synthesize_with_cloned_voice(args.text, voice_profile_dir, speech_wav)
        log.info("Using cloned voice (Chatterbox)")
    else:
        from pipeline.tts import synthesize
        audio_bytes = synthesize(args.text)
        with open(speech_wav, "wb") as f:
            f.write(audio_bytes)
        log.info("Using generic TTS (Kokoro fallback)")
    done(t, speech_wav)

    # ── Stage 6: Sonic lipsync ───────────────────────────────────────────────
    t = stage("6. Sonic lipsync (avatar + speech → video)")
    from pipeline.sonic_lipsync import run_sonic
    run_sonic(avatar_png, speech_wav, sonic_mp4, dynamic_scale=1.0)
    done(t, sonic_mp4)

    # ── Stage 7: CodeFormer per-frame polish ─────────────────────────────────
    t = stage("7. CodeFormer per-frame polish (fidelity=0.7)")
    try:
        from pipeline.codeformer_polish import polish_video
        polish_video(sonic_mp4, final_mp4, fidelity=0.7)
        done(t, final_mp4)
    except Exception as e:
        log.warning(f"CodeFormer failed ({e}), final = sonic output")
        final_mp4 = sonic_mp4

    log.info("\n" + "="*60)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  face:     {face_jpg}")
    log.info(f"  portrait: {portrait_png}")
    log.info(f"  avatar:   {avatar_png}")
    log.info(f"  speech:   {speech_wav}")
    log.info(f"  sonic:    {sonic_mp4}")
    log.info(f"  final:    {final_mp4}")
    log.info("="*60)
    log.info(f"\nDownload:\n  scp -i ~/.ssh/graperoot-bench.pem ubuntu@<EC2_IP>:{final_mp4} ~/shiv_final.mp4")
    log.info(f"  scp -i ~/.ssh/graperoot-bench.pem ubuntu@<EC2_IP>:{avatar_png} ~/shiv_avatar.png")


if __name__ == "__main__":
    main()
