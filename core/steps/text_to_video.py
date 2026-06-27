"""CogVideoX text-to-video / image-to-video (THUDM, Apache 2.0).

Open-source equivalent of Higgsfield's text-to-video generators. Generates a
short clip from a text prompt (T2V) or animates a still image with a prompt
(I2V). Like every heavy model in this repo, inference runs in an isolated venv
subprocess so its diffusers/torch versions never conflict with Sonic/PuLID and
VRAM is fully released when it exits.

CogVideoX-5b fits on a 24 GB A10G with ``enable_model_cpu_offload()`` +
``vae.enable_tiling()``. Default output: 49 frames @ 8 fps (~6 s, 720x480).
"""
import os
import logging
import subprocess
import tempfile

log = logging.getLogger(__name__)

COGVIDEO_DIR = os.environ.get("COGVIDEO_DIR", "/home/ubuntu/CogVideo")
VENV_PYTHON = os.environ.get("COGVIDEO_VENV_PYTHON", "/home/ubuntu/venv-cogvideo/bin/python")
T2V_MODEL = os.environ.get("COGVIDEO_T2V_MODEL", "THUDM/CogVideoX-5b")
I2V_MODEL = os.environ.get("COGVIDEO_I2V_MODEL", "THUDM/CogVideoX-5b-I2V")

_SCRIPT = r'''
import sys, os, torch
from diffusers.utils import export_to_video

mode        = sys.argv[1]            # "t2v" | "i2v"
model_id    = sys.argv[2]
prompt      = sys.argv[3]
output_path = sys.argv[4]
num_frames  = int(sys.argv[5])
steps       = int(sys.argv[6])
guidance    = float(sys.argv[7])
fps         = int(sys.argv[8])
seed        = int(sys.argv[9])
image_path  = sys.argv[10] if len(sys.argv) > 10 and sys.argv[10] else None

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

if mode == "i2v":
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import load_image
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_id, torch_dtype=dtype)
else:
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=dtype)

# Memory-friendly: offload weights to CPU, tile the VAE — fits 24GB GPUs.
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

generator = torch.Generator(device="cpu").manual_seed(seed)

kwargs = dict(
    prompt=prompt,
    num_frames=num_frames,
    num_inference_steps=steps,
    guidance_scale=guidance,
    num_videos_per_prompt=1,
    generator=generator,
)
if mode == "i2v":
    kwargs["image"] = load_image(image_path)

video = pipe(**kwargs).frames[0]

os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
export_to_video(video, output_path, fps=fps)
print(f"T2V_DONE:{output_path}", flush=True)
'''


def _run(mode: str, model_id: str, prompt: str, output_path: str,
         num_frames: int, steps: int, guidance: float, fps: int,
         seed: int, image_path: str | None) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(_SCRIPT)
        script_path = f.name

    args = [
        VENV_PYTHON, script_path, mode, model_id, prompt, output_path,
        str(num_frames), str(steps), str(guidance), str(fps), str(seed),
        image_path or "",
    ]
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=2400,
            encoding="utf-8", errors="replace",
        )
    finally:
        os.unlink(script_path)

    if proc.returncode != 0 or "T2V_DONE:" not in proc.stdout:
        log.error(f"CogVideoX stderr: {proc.stderr[-2000:]}")
        raise RuntimeError(f"CogVideoX {mode} generation failed:\n{proc.stderr[-1500:]}")

    log.info(f"CogVideoX {mode} done: {output_path}")
    return output_path


def generate_video_from_text(prompt: str, output_path: str,
                             num_frames: int = 49, steps: int = 50,
                             guidance: float = 6.0, fps: int = 8,
                             seed: int = 42) -> str:
    """Generate a video clip from a text prompt using CogVideoX-5b.

    Args:
        prompt: Text description of the scene. CogVideoX responds best to long,
            detailed prompts (one or two descriptive sentences).
        output_path: Destination .mp4.
        num_frames: Number of frames (49 ≈ 6s at 8fps; must be 8k+1).
        steps: Denoising steps (50 is the quality default).
        guidance: Classifier-free guidance scale.
        fps: Output frame rate.
        seed: RNG seed for reproducibility.

    Returns:
        output_path
    """
    return _run("t2v", T2V_MODEL, prompt, output_path,
                num_frames, steps, guidance, fps, seed, None)


def generate_video_from_image(image_path: str, prompt: str, output_path: str,
                              num_frames: int = 49, steps: int = 50,
                              guidance: float = 6.0, fps: int = 8,
                              seed: int = 42) -> str:
    """Animate a still image into a video clip using CogVideoX-5b-I2V.

    Args:
        image_path: Source still image (the first frame of the video).
        prompt: Text description of the desired motion / scene.
        output_path: Destination .mp4.

    Returns:
        output_path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    return _run("i2v", I2V_MODEL, prompt, output_path,
                num_frames, steps, guidance, fps, seed, image_path)
