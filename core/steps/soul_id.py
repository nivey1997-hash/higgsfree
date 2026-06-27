"""Soul ID — persistent face-identity embedding per avatar.

Higgsfield's "Soul ID" locks an avatar's face across every generation by
training on many photos. This is the open-source equivalent: at onboarding we
compute a single averaged InsightFace (buffalo_l) identity embedding from the
consent video / photos and persist it as a tiny ``.npy``. On every later
generation we reuse that embedding to:

  1. pick the consent-video frame whose face best matches the locked identity
     (``select_best_face_frame``) — so PuLID always conditions on the most
     on-identity frame, not a random frontal one, and
  2. verify the rendered output still matches the avatar (``verify_identity``)
     for QA / scoring.

The embedding is a 512-d L2-normalised vector; cosine similarity maps to a
[0, 1] identity score. Storage is intentionally a plain ``.npy`` so the worker
can cache it in S3 under ``avatars/<id>/soul_id.npy`` with no schema change.
"""
import os
import logging
import cv2
import numpy as np

log = logging.getLogger(__name__)

_app = None

# Frames scoring below this cosine similarity to the soul embedding are treated
# as "not this person" and rejected during identity-locked frame selection.
MIN_IDENTITY_SIMILARITY = float(os.environ.get("SOUL_ID_MIN_SIM", "0.45"))


def _load_app():
    """Lazily load InsightFace buffalo_l (detection + recognition embeddings)."""
    global _app
    if _app is not None:
        return _app
    from insightface.app import FaceAnalysis
    log.info("Soul ID: loading InsightFace buffalo_l...")
    _app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    _app.prepare(ctx_id=0, det_size=(640, 640))
    log.info("Soul ID: InsightFace loaded.")
    return _app


def _largest_face(faces):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _normed(emb: np.ndarray) -> np.ndarray:
    return emb / (np.linalg.norm(emb) + 1e-8)


def similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """Cosine similarity between two embeddings, mapped to [0, 1]."""
    a, b = _normed(np.asarray(emb_a)), _normed(np.asarray(emb_b))
    cos = float(np.dot(a, b))
    return max(0.0, min(1.0, (cos + 1) / 2))


def compute_embedding_from_image(image_path: str) -> np.ndarray | None:
    """Return the normalised identity embedding of the largest face in an image."""
    app = _load_app()
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")
    faces = app.get(img)
    if not faces:
        return None
    emb = _largest_face(faces).normed_embedding
    return _normed(np.asarray(emb)) if emb is not None else None


def compute_embedding_from_video(video_path: str, max_samples: int = 30) -> np.ndarray | None:
    """Average identity embedding across frontal frames of a video.

    Samples up to ``max_samples`` frames evenly, keeps only reasonably frontal
    detections, and returns the mean (then re-normalised) embedding — a more
    stable identity than any single frame.
    """
    app = _load_app()
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, total // max_samples) if total else 5

    embeddings = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            faces = app.get(frame)
            if faces:
                face = _largest_face(faces)
                # Frontal-ish only (InsightFace pose = [pitch, yaw, roll])
                yaw = float(face.pose[1]) if getattr(face, "pose", None) is not None else 0.0
                if abs(yaw) <= 30 and getattr(face, "normed_embedding", None) is not None:
                    embeddings.append(np.asarray(face.normed_embedding))
        frame_idx += 1
    cap.release()

    if not embeddings:
        return None
    return _normed(np.mean(embeddings, axis=0))


def save_soul_id(embedding: np.ndarray, out_path: str) -> str:
    """Persist a soul embedding to disk as ``.npy``."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, np.asarray(embedding, dtype=np.float32))
    log.info(f"Soul ID saved: {out_path}")
    return out_path


def load_soul_id(path: str) -> np.ndarray | None:
    """Load a soul embedding from ``.npy`` (returns None if missing/unreadable)."""
    if not path or not os.path.exists(path):
        return None
    try:
        return np.load(path)
    except Exception as e:
        log.warning(f"Could not load soul id {path}: {e}")
        return None


def build_soul_id(source_path: str, out_path: str) -> str | None:
    """Compute and persist a soul embedding from a video or image.

    Returns the saved path, or None if no usable face was found.
    """
    ext = os.path.splitext(source_path)[1].lower()
    if ext in (".mp4", ".webm", ".mov", ".avi", ".mkv"):
        emb = compute_embedding_from_video(source_path)
    else:
        emb = compute_embedding_from_image(source_path)
    if emb is None:
        log.warning(f"Soul ID: no face found in {source_path}")
        return None
    return save_soul_id(emb, out_path)


def select_best_face_frame(video_path: str, output_jpg: str,
                           soul_embedding: np.ndarray,
                           min_similarity: float = MIN_IDENTITY_SIMILARITY) -> str:
    """Pick the frame whose face best matches the locked soul identity.

    Combines identity match (60%), detection confidence (20%), face area (10%)
    and sharpness (10%). Frames below ``min_similarity`` to the soul embedding
    are skipped so a wrong person / bad detection never wins. Falls back to the
    best-scoring frame ignoring the similarity floor if nothing clears it.
    """
    app = _load_app()
    soul = _normed(np.asarray(soul_embedding))

    cap = cv2.VideoCapture(video_path)
    best_score = -1.0
    best_frame = None
    best_relaxed_score = -1.0
    best_relaxed_frame = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 5 == 0:
            faces = app.get(frame)
            if faces:
                face = _largest_face(faces)
                emb = getattr(face, "normed_embedding", None)
                if emb is not None:
                    sim = similarity(soul, np.asarray(emb))
                    area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                    x1, y1, x2, y2 = [max(0, int(v)) for v in face.bbox]
                    roi = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) if (y2 > y1 and x2 > x1) else None
                    sharp = cv2.Laplacian(roi, cv2.CV_64F).var() if roi is not None and roi.size else 0
                    score = (sim * 0.6
                             + float(face.det_score) * 0.2
                             + (area / (frame.shape[0] * frame.shape[1])) * 0.1
                             + min(sharp / 500, 1) * 0.1)

                    if score > best_relaxed_score:
                        best_relaxed_score = score
                        best_relaxed_frame = frame.copy()
                    if sim >= min_similarity and score > best_score:
                        best_score = score
                        best_frame = frame.copy()
        frame_idx += 1
    cap.release()

    chosen = best_frame if best_frame is not None else best_relaxed_frame
    if chosen is None:
        raise RuntimeError(f"No face detected in video: {video_path}")

    if best_frame is None:
        log.warning(f"Soul ID: no frame cleared similarity {min_similarity}; "
                    f"using best relaxed match (score={best_relaxed_score:.3f})")
    else:
        log.info(f"Soul ID frame selected (score={best_score:.3f}): {output_jpg}")

    os.makedirs(os.path.dirname(os.path.abspath(output_jpg)), exist_ok=True)
    cv2.imwrite(output_jpg, chosen)
    return output_jpg


def verify_identity(video_path: str, soul_embedding: np.ndarray,
                    threshold: float = 0.6) -> tuple[bool, float]:
    """Check whether a rendered video still matches the locked soul identity.

    Returns ``(passed, similarity_score)`` where the score is the mean identity
    similarity across sampled output frames. Useful as a generation QA gate.
    """
    out_emb = compute_embedding_from_video(video_path)
    if out_emb is None:
        return False, 0.0
    score = similarity(soul_embedding, out_emb)
    return score >= threshold, score
