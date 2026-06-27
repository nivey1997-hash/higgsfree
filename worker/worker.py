#!/usr/bin/env python3
"""SQS polling worker for Avatar.graperoot video generation pipeline."""

import os
import json
import time
import logging
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import psycopg2
from psycopg2.extras import RealDictCursor

from pipeline.tts import synthesize
from pipeline.avatar_gen import generate_avatar_image, generate_avatar_from_video, extract_best_face_frame, crop_head_shoulders
from pipeline.voiceclone import extract_voice_profile, synthesize_with_cloned_voice
from pipeline.sonic_lipsync import run_sonic
from pipeline.codeformer_polish import polish_video
from pipeline.video_sr import upscale_video
from pipeline.soul_id import build_soul_id, load_soul_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Detect GPU availability
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    log.info(f"GPU available: {HAS_GPU}")
except ImportError:
    HAS_GPU = False
    log.info("PyTorch not installed. Using Replicate fallback.")

if not HAS_GPU:
    from pipeline.replicate_fallback import run_lipsync as _replicate_lipsync
else:
    _replicate_lipsync = None

# AWS / DB config from env
SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ["S3_BUCKET"]
DATABASE_URL = os.environ["DATABASE_URL"]

sqs = boto3.client("sqs", region_name=AWS_REGION)
s3 = boto3.client("s3", region_name=AWS_REGION)


def get_db():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def update_status(video_id: str, status: str, error_message: str = None):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            if error_message:
                cur.execute(
                    'UPDATE "Video" SET status=%s, "errorMessage"=%s, "updatedAt"=NOW() WHERE id=%s',
                    (status, error_message, video_id),
                )
            else:
                cur.execute(
                    'UPDATE "Video" SET status=%s, "updatedAt"=NOW() WHERE id=%s',
                    (status, video_id),
                )
        conn.commit()
    finally:
        conn.close()


def update_done(video_id: str, s3_key: str, cloudfront_url: str, duration: float):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE "Video" SET status=\'DONE\', "s3Key"=%s, "cloudfrontUrl"=%s, '
                '"durationSeconds"=%s, "updatedAt"=NOW() WHERE id=%s',
                (s3_key, cloudfront_url, duration, video_id),
            )
        conn.commit()
    finally:
        conn.close()


def get_avatar(avatar_id: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT * FROM "Avatar" WHERE id=%s', (avatar_id,))
            return cur.fetchone()
    finally:
        conn.close()


def download_s3(key: str, local_path: str):
    s3.download_file(S3_BUCKET, key, local_path)


def upload_s3(local_path: str, key: str, content_type: str = "video/mp4") -> str:
    s3.upload_file(local_path, S3_BUCKET, key, ExtraArgs={"ContentType": content_type})
    cloudfront = os.environ.get("CLOUDFRONT_DOMAIN", "")
    return f"https://{cloudfront}/{key}" if cloudfront else ""


def _download_voice_profile(avatar: dict, avatar_id: str, tmpdir: str) -> str | None:
    """Return local voice profile dir if one is cached in S3, else None."""
    profile_key_prefix = avatar.get("voiceProfileKey")
    if not profile_key_prefix:
        return None
    profile_dir = os.path.join(tmpdir, "voice_profile")
    os.makedirs(profile_dir, exist_ok=True)
    try:
        download_s3(f"{profile_key_prefix}/ref_audio.wav", os.path.join(profile_dir, "ref_audio.wav"))
        log.info(f"Reusing cached voice profile: {profile_key_prefix}")
        return profile_dir
    except Exception as e:
        log.warning(f"Could not load cached voice profile: {e}")
        return None


def _download_soul_id(avatar_id: str, tmpdir: str):
    """Return the avatar's locked Soul ID embedding if cached in S3, else None."""
    soul_key = f"avatars/{avatar_id}/soul_id.npy"
    local_path = os.path.join(tmpdir, "soul_id.npy")
    try:
        download_s3(soul_key, local_path)
        emb = load_soul_id(local_path)
        if emb is not None:
            log.info(f"Reusing Soul ID: {soul_key}")
        return emb
    except Exception:
        return None


def _synth_audio(text: str, idx: int, tmpdir: str, voice_profile_dir: str | None) -> str:
    """Synthesize speech for one segment — Chatterbox cloned voice if available, Kokoro fallback."""
    audio_path = os.path.join(tmpdir, f"audio_{idx}.wav")
    if voice_profile_dir:
        try:
            synthesize_with_cloned_voice(text, voice_profile_dir, audio_path)
            return audio_path
        except Exception as e:
            log.warning(f"Chatterbox failed for segment {idx}, falling back to Kokoro: {e}")
    audio_bytes = synthesize(text)
    with open(audio_path, "wb") as f:
        f.write(audio_bytes)
    return audio_path



def _loop_consent_video(consent_video: str, audio_path: str, out_path: str) -> str:
    """Loop consent video to match audio duration at 512x512 25fps for LatentSync."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 10.0

    subprocess.run([
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", consent_video,
        "-t", str(duration),
        "-vf", "scale=512:512",
        "-c:v", "libx264", "-r", "25", "-pix_fmt", "yuv420p",
        "-an", out_path,
    ], check=True, capture_output=True)
    return out_path


def _generate_avatar_segment(
    idx: int,
    audio_path: str,
    avatar_image_path: str,
    tmpdir: str,
) -> str:
    """Generate talking avatar segment: Sonic → CodeFormer → mux audio."""
    sonic_out = os.path.join(tmpdir, f"sonic_{idx}.mp4")
    run_sonic(avatar_image_path, audio_path, sonic_out, dynamic_scale=1.0)

    polished_out = os.path.join(tmpdir, f"polished_{idx}.mp4")
    try:
        polish_video(sonic_out, polished_out, fidelity=0.7)
    except Exception as e:
        log.warning(f"CodeFormer polish failed segment {idx} (non-fatal): {e}")
        polished_out = sonic_out

    # Mux TTS audio into the video (Sonic/CodeFormer output is video-only)
    final_out = os.path.join(tmpdir, f"avatar_{idx}.mp4")
    audio_dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True,
    ).stdout.strip() or "0")

    subprocess.run([
        "ffmpeg", "-y",
        "-i", polished_out,
        "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac",
        "-t", f"{audio_dur:.3f}",
        "-shortest",
        final_out,
    ], check=True, capture_output=True)
    return final_out


def process_job(payload: dict):
    """Main video generation job.

    Pipeline: avatar portrait → TTS per segment → Sonic lipsync → CodeFormer → ffmpeg concat → S3
    """
    video_id = payload["videoId"]
    avatar_id = payload["avatarId"]
    timeline_json = payload["timelineJson"]

    log.info(f"Processing video {video_id}")
    timeline = json.loads(timeline_json)
    segments = timeline.get("segments", [])

    with tempfile.TemporaryDirectory() as tmpdir:
        avatar = get_avatar(avatar_id)
        outfit_id = payload.get("outfitId", "casual-white")
        scene_id = payload.get("sceneId", "studio-clean")

        # ── Step 1: Download avatar portrait ─────────────────────────────────
        update_status(video_id, "PROCESSING_AVATAR_GEN")
        avatar_image_path = None
        consent_video_local = None

        consent_video_key = avatar.get("consentVideoKey")
        cached_portrait_key = avatar.get("cachedFrameKey")

        # Locked Soul ID identity (if onboarding cached one) — used to pick the
        # most on-identity consent frame for PuLID.
        soul_embedding = _download_soul_id(avatar_id, tmpdir)

        if HAS_GPU:
            if consent_video_key:
                consent_ext = ".webm" if consent_video_key.endswith(".webm") else ".mp4"
                consent_video_local = os.path.join(tmpdir, f"consent{consent_ext}")
                download_s3(consent_video_key, consent_video_local)

            if cached_portrait_key:
                avatar_image_path = os.path.join(tmpdir, "avatar_portrait.png")
                download_s3(cached_portrait_key, avatar_image_path)
                log.info(f"Reusing cached portrait: {cached_portrait_key}")

            if avatar_image_path is None:
                if consent_video_local:
                    face_jpg = os.path.join(tmpdir, "face.jpg")
                    avatar_image_path = os.path.join(tmpdir, "avatar_portrait.png")
                    try:
                        generate_avatar_from_video(
                            consent_video_local, outfit_id, scene_id,
                            face_jpg, avatar_image_path,
                            soul_embedding=soul_embedding,
                        )
                        portrait_key = f"avatars/{avatar_id}/portrait.png"
                        upload_s3(avatar_image_path, portrait_key, content_type="image/png")
                        _update_avatar_portrait(avatar_id, portrait_key)
                    except Exception as e:
                        log.warning(f"Avatar portrait gen failed: {e}")
                        avatar_image_path = face_jpg if os.path.exists(face_jpg) else None
                elif avatar.get("frontImageKey"):
                    face_jpg = os.path.join(tmpdir, "face.jpg")
                    download_s3(avatar["frontImageKey"], face_jpg)
                    avatar_image_path = os.path.join(tmpdir, "avatar_portrait.png")
                    try:
                        generate_avatar_image(face_jpg, outfit_id, scene_id, avatar_image_path)
                        portrait_key = f"avatars/{avatar_id}/portrait.png"
                        upload_s3(avatar_image_path, portrait_key, content_type="image/png")
                        _update_avatar_portrait(avatar_id, portrait_key)
                    except Exception as e:
                        log.warning(f"PuLID from frontImageKey failed, using crop: {e}")
                        try:
                            crop_head_shoulders(face_jpg, avatar_image_path)
                        except Exception:
                            avatar_image_path = face_jpg

        # Crop avatar to head+shoulders (once) so Sonic never sees hands/body
        if avatar_image_path and os.path.exists(avatar_image_path):
            try:
                cropped_path = os.path.join(tmpdir, "avatar_shoulder.png")
                crop_head_shoulders(avatar_image_path, cropped_path)
                avatar_image_path = cropped_path
            except Exception as e:
                log.warning(f"Head+shoulders crop failed, using full portrait: {e}")

        # ── Step 2: TTS for all segments ─────────────────────────────────────
        update_status(video_id, "PROCESSING_TTS")
        voice_profile_dir = _download_voice_profile(avatar, avatar_id, tmpdir)

        audio_map: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=4) as pool:
            tts_futures = {
                pool.submit(_synth_audio, seg["text"], i, tmpdir, voice_profile_dir): i
                for i, seg in enumerate(segments) if seg.get("text")
            }
            for fut in as_completed(tts_futures):
                i = tts_futures[fut]
                try:
                    audio_map[i] = fut.result()
                except Exception as e:
                    log.error(f"TTS failed segment {i}: {e}")

        # ── Step 3: Sonic + CodeFormer per segment (serial — VRAM) ───────────
        lipsync_videos: dict[int, str] = {}
        if HAS_GPU and avatar_image_path:
            update_status(video_id, "PROCESSING_LIPSYNC")
            for i, seg in enumerate(segments):
                if i not in audio_map:
                    log.warning(f"No audio for segment {i}, skipping lipsync")
                    continue
                try:
                    lipsync_videos[i] = _generate_avatar_segment(
                        i, audio_map[i], avatar_image_path, tmpdir
                    )
                except Exception as e:
                    log.error(f"Sonic failed segment {i}: {e}")
                    if _replicate_lipsync:
                        try:
                            vid = os.path.join(tmpdir, f"avatar_{i}.mp4")
                            _replicate_lipsync(avatar_image_path, audio_map[i], vid)
                            lipsync_videos[i] = vid
                        except Exception as e2:
                            log.error(f"Replicate fallback failed segment {i}: {e2}")
                            raise RuntimeError(f"Lipsync failed segment {i}: Sonic={e}, Replicate={e2}")
                    else:
                        raise RuntimeError(f"Lipsync failed segment {i}: {e}")

        # ── Step 4: ffmpeg concat + scale to aspect ratio ────────────────────
        update_status(video_id, "PROCESSING_COMPOSITE")
        output_path = os.path.join(tmpdir, f"output_{video_id}.mp4")

        clip_paths = [lipsync_videos[i] for i in sorted(lipsync_videos.keys())]
        if not clip_paths:
            raise RuntimeError("No lipsync segments produced")

        concat_list = os.path.join(tmpdir, "concat.txt")
        with open(concat_list, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        aspect = payload.get("aspectRatio", "9:16")
        aspect_sizes = {
            "9:16": (720, 1280),
            "1:1": (720, 720),
            "16:9": (1280, 720),
            "4:5": (720, 900),
            "4:3": (960, 720),
        }
        target_w, target_h = aspect_sizes.get(aspect, (720, 1280))

        concat_tmp = os.path.join(tmpdir, "concat_raw.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_list, "-c", "copy", concat_tmp,
        ], check=True, capture_output=True)

        # Scale to fit, pad with studio grey background (matches PuLID portrait bg)
        subprocess.run([
            "ffmpeg", "-y", "-i", concat_tmp,
            "-vf", f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:color=#D4D4D4",
            "-c:v", "libx264", "-c:a", "copy", "-pix_fmt", "yuv420p",
            output_path,
        ], check=True, capture_output=True)

        # ── Optional: Real-ESRGAN super-resolution upscale (e.g. 2x → ~4K) ────
        upscale = int(payload.get("upscale", 1))
        if HAS_GPU and upscale > 1:
            update_status(video_id, "PROCESSING_UPSCALE")
            upscaled_path = os.path.join(tmpdir, f"output_{video_id}_x{upscale}.mp4")
            try:
                upscale_video(output_path, upscaled_path, upscale=upscale)
                output_path = upscaled_path
            except Exception as e:
                log.warning(f"Upscale failed (non-fatal), using base output: {e}")

        out_key = f"videos/{video_id}/output.mp4"
        cloudfront_url = upload_s3(output_path, out_key)

        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", output_path],
            capture_output=True, text=True,
        )
        duration = 0.0
        try:
            duration = float(json.loads(probe.stdout)["format"]["duration"])
        except Exception:
            pass

        update_done(video_id, out_key, cloudfront_url, duration)
        log.info(f"Video {video_id} done: {cloudfront_url}")


def _update_avatar_portrait(avatar_id: str, portrait_key: str):
    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                'UPDATE "Avatar" SET "cachedFrameKey"=%s, "updatedAt"=NOW() WHERE id=%s',
                (portrait_key, avatar_id),
            )
        conn.commit()
    finally:
        conn.close()


def process_consent_video(payload: dict):
    """Process a newly uploaded consent video.

    Triggered when user finishes recording their avatar consent video.
    Does the slow work upfront so video generation later is faster:
      1. Extract best face frame
      2. Generate portrait avatar image
      3. Extract Chatterbox voice profile (ref_audio.wav)
      4. Upload face + portrait + voice profile to S3, update Avatar DB record
    """
    avatar_id = payload["avatarId"]
    consent_video_key = payload["consentVideoKey"]
    outfit_id = payload.get("outfitId", "casual-white")
    scene_id = payload.get("sceneId", "studio-clean")

    log.info(f"Processing consent video for avatar {avatar_id}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Preserve original extension so ffmpeg can detect codec correctly
        consent_ext = ".webm" if consent_video_key.endswith(".webm") else ".mp4"
        consent_video_local = os.path.join(tmpdir, f"consent{consent_ext}")
        download_s3(consent_video_key, consent_video_local)

        face_jpg = os.path.join(tmpdir, "face.jpg")
        portrait_png = os.path.join(tmpdir, "portrait.png")

        # ── Soul ID: lock the avatar's face identity once, upfront ───────────
        # Averaged InsightFace embedding from the consent video → cached in S3.
        # Reused on every future generation to pick on-identity frames.
        soul_embedding = None
        try:
            soul_local = os.path.join(tmpdir, "soul_id.npy")
            if build_soul_id(consent_video_local, soul_local):
                soul_embedding = load_soul_id(soul_local)
                upload_s3(soul_local, f"avatars/{avatar_id}/soul_id.npy",
                          content_type="application/octet-stream")
                log.info(f"Soul ID locked for avatar {avatar_id}")
        except Exception as e:
            log.warning(f"Soul ID build failed (non-fatal): {e}")

        if HAS_GPU:
            try:
                # If user uploaded a face photo, use it directly (better quality than video frame)
                face_photo_key = None
                conn_check = get_db()
                try:
                    with conn_check.cursor() as cur:
                        cur.execute('SELECT "frontImageKey" FROM "Avatar" WHERE id=%s', (avatar_id,))
                        row = cur.fetchone()
                        if row:
                            face_photo_key = row.get("frontImageKey")
                finally:
                    conn_check.close()

                if face_photo_key:
                    log.info(f"Using uploaded face photo: {face_photo_key}")
                    download_s3(face_photo_key, face_jpg)
                    # PuLID → img2img → crop from clean photo
                    from pipeline.avatar_gen import generate_pulid_portrait, refine_portrait_img2img, crop_head_shoulders
                    portrait_tmp = os.path.join(tmpdir, "_pulid.png")
                    refined_tmp  = os.path.join(tmpdir, "_refined.png")
                    generate_pulid_portrait(face_jpg, portrait_tmp)
                    try:
                        refine_portrait_img2img(portrait_tmp, refined_tmp)
                        crop_head_shoulders(refined_tmp, portrait_png)
                    except Exception:
                        crop_head_shoulders(portrait_tmp, portrait_png)
                else:
                    # Full onboarding: frame pick → PuLID → img2img → crop → avatar.png
                    generate_avatar_from_video(
                        consent_video_local, outfit_id, scene_id,
                        face_jpg, portrait_png,
                        soul_embedding=soul_embedding,
                    )
            except Exception as e:
                log.error(f"Avatar generation failed: {e}")
                raise RuntimeError(f"Avatar portrait generation failed: {e}")
        else:
            # No GPU — just extract face frame, worker will handle portrait later
            extract_best_face_frame(consent_video_local, face_jpg)
            portrait_png = face_jpg

        # Extract Chatterbox voice profile from consent video audio
        voice_profile_key = None
        try:
            voice_profile_local = os.path.join(tmpdir, "voice_profile")
            extract_voice_profile(consent_video_local, voice_profile_local)
            voice_prefix = f"avatars/{avatar_id}/voice_profile"
            ref_wav = os.path.join(voice_profile_local, "ref_audio.wav")
            if os.path.exists(ref_wav):
                upload_s3(ref_wav, f"{voice_prefix}/ref_audio.wav", content_type="audio/wav")
            voice_profile_key = voice_prefix
            log.info(f"Voice profile cached: {voice_profile_key}")
        except Exception as e:
            log.warning(f"Voice profile extraction failed (non-fatal): {e}")

        # Upload face + portrait
        face_key = f"avatars/{avatar_id}/face.jpg"
        portrait_key = f"avatars/{avatar_id}/portrait.png"
        upload_s3(face_jpg, face_key, content_type="image/jpeg")
        if os.path.exists(portrait_png) and portrait_png != face_jpg:
            upload_s3(portrait_png, portrait_key, content_type="image/png")
            cached_frame_key = portrait_key
        else:
            # Fallback: portrait is the same as face, store under portrait_key too
            upload_s3(face_jpg, portrait_key, content_type="image/jpeg")
            cached_frame_key = portrait_key

        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE "Avatar" SET "frontImageKey"=%s, "cachedFrameKey"=%s, '
                    '"consentVideoKey"=%s, "voiceProfileKey"=%s, '
                    '"outfitId"=%s, "sceneId"=%s, "isProcessed"=true, "updatedAt"=NOW() WHERE id=%s',
                    (face_key, cached_frame_key, consent_video_key,
                     voice_profile_key, outfit_id, scene_id, avatar_id),
                )
            conn.commit()
        finally:
            conn.close()

        log.info(f"Consent video processed for avatar {avatar_id}: portrait={portrait_key}, voice={voice_profile_key}")


def process_avatar_preview(payload: dict):
    """Generate a full-body avatar preview image and upload to S3."""
    avatar_id = payload["avatarId"]
    outfit_id = payload.get("outfitId", "casual-white")
    scene_id = payload.get("sceneId", "studio-clean")

    log.info(f"Generating avatar preview: {avatar_id} outfit={outfit_id} scene={scene_id}")

    conn = get_db()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT "frontImageKey", "cachedFrameKey" FROM "Avatar" WHERE id=%s', (avatar_id,))
            avatar = cur.fetchone()
    finally:
        conn.close()

    if not avatar or not avatar.get("frontImageKey"):
        raise RuntimeError(f"Avatar {avatar_id} has no face image")

    with tempfile.TemporaryDirectory() as tmpdir:
        face_local = os.path.join(tmpdir, "face.jpg")
        download_s3(avatar["frontImageKey"], face_local)

        output_path = os.path.join(tmpdir, "avatar_preview.png")
        generate_avatar_image(face_local, outfit_id, scene_id, output_path)

        # Upload to S3
        preview_key = f"avatars/{avatar_id}/preview.png"
        upload_s3(output_path, preview_key, content_type="image/png")

        # Update avatar record with preview key
        conn = get_db()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    'UPDATE "Avatar" SET "cachedFrameKey"=%s, "outfitId"=%s, "sceneId"=%s, "updatedAt"=NOW() WHERE id=%s',
                    (preview_key, outfit_id, scene_id, avatar_id),
                )
            conn.commit()
        finally:
            conn.close()

        log.info(f"Avatar preview done: {preview_key}")


def _get_instance_id() -> str | None:
    """Get own EC2 instance ID from instance metadata service."""
    try:
        import urllib.request
        # IMDSv2: get token first
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
            method="PUT",
        )
        token = urllib.request.urlopen(req, timeout=2).read().decode()
        req2 = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-id",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req2, timeout=2).read().decode()
    except Exception as e:
        log.warning(f"Could not get instance ID from metadata: {e}")
        return None


def _self_stop(instance_id: str):
    """Stop this EC2 instance — called when queue has been empty long enough."""
    log.info(f"Queue empty — stopping instance {instance_id}")
    try:
        ec2_client = boto3.client("ec2", region_name=AWS_REGION)
        ec2_client.stop_instances(InstanceIds=[instance_id])
        log.info("Stop requested. Exiting worker.")
    except Exception as e:
        log.error(f"Failed to stop instance: {e}")


# How many consecutive empty long-polls (20s each) before auto-stopping.
# 3 × 20s = 60s idle grace period after last job.
IDLE_POLLS_BEFORE_STOP = int(os.environ.get("IDLE_POLLS_BEFORE_STOP", "3"))


def main():
    log.info("Worker started. Polling SQS...")
    instance_id = _get_instance_id()
    if instance_id:
        log.info(f"Running on instance {instance_id} — auto-stop enabled after {IDLE_POLLS_BEFORE_STOP} idle polls")
    else:
        log.info("Could not detect instance ID — auto-stop disabled (running locally?)")

    idle_count = 0

    while True:
        try:
            response = sqs.receive_message(
                QueueUrl=SQS_QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=2700,  # 45 min — covers worst-case Sonic run
            )
            messages = response.get("Messages", [])

            if not messages:
                idle_count += 1
                log.info(f"Queue empty ({idle_count}/{IDLE_POLLS_BEFORE_STOP} idle polls)")
                if instance_id and idle_count >= IDLE_POLLS_BEFORE_STOP:
                    _self_stop(instance_id)
                    break
                continue

            # Got a job — reset idle counter
            idle_count = 0
            message = messages[0]
            receipt = message["ReceiptHandle"]
            payload = json.loads(message["Body"])

            try:
                job_type = payload.get("type", "video")
                if job_type == "consent_video":
                    process_consent_video(payload)
                elif job_type == "avatar_preview":
                    process_avatar_preview(payload)
                else:
                    process_job(payload)
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt)
            except Exception as e:
                log.error(f"Job failed: {e}\n{traceback.format_exc()}")
                video_id = payload.get("videoId", "unknown")
                if video_id != "unknown":
                    update_status(video_id, "FAILED", str(e))
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt)

        except KeyboardInterrupt:
            log.info("Shutting down.")
            break
        except Exception as e:
            log.error(f"Polling error: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
