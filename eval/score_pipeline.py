#!/usr/bin/env python3
"""Score pipeline output quality.

Usage: python eval/score_pipeline.py <source_video> <output_video>
Prints a single float score to stdout (0.0 - 1.0).

Score = 0.5 * face_similarity + 0.5 * lipsync_confidence
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'worker'))


def extract_face_embedding(app, video_path, max_frames=10):
    """Sample frames from video and return mean face embedding."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    embeddings = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            faces = app.get(frame)
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                if hasattr(f, 'normed_embedding') and f.normed_embedding is not None:
                    embeddings.append(f.normed_embedding)
        frame_idx += 1

    cap.release()
    if not embeddings:
        return None
    return np.mean(embeddings, axis=0)


def face_similarity_score(source_video: str, output_video: str) -> float:
    """Cosine similarity between source face identity and output video faces."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l',
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    src_emb = extract_face_embedding(app, source_video)
    out_emb = extract_face_embedding(app, output_video)

    if src_emb is None or out_emb is None:
        print("WARNING: could not extract face embeddings", file=sys.stderr)
        return 0.0

    sim = float(np.dot(src_emb, out_emb) /
                (np.linalg.norm(src_emb) * np.linalg.norm(out_emb) + 1e-6))
    # Clamp to [0, 1]
    return max(0.0, min(1.0, (sim + 1) / 2))


def lipsync_confidence_score(output_video: str) -> float:
    """Estimate lip sync quality via mouth motion variance across frames.

    A well-synced video has high mouth variance (lips moving with speech).
    A static/failed video has near-zero variance.
    """
    cap = cv2.VideoCapture(output_video)
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name='buffalo_l',
                       providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                       allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    mouth_heights = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 3 == 0:
            faces = app.get(frame)
            if faces:
                f = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                if hasattr(f, 'landmark_2d_106') and f.landmark_2d_106 is not None:
                    lm = f.landmark_2d_106
                    # Upper lip: 52, lower lip: 57 (106-point scheme)
                    mouth_h = abs(lm[57][1] - lm[52][1])
                    mouth_heights.append(mouth_h)
        frame_idx += 1
    cap.release()

    if len(mouth_heights) < 5:
        return 0.0

    variance = float(np.std(mouth_heights))
    # Normalize: ~5px std = good sync, scale to [0,1]
    return max(0.0, min(1.0, variance / 8.0))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: score_pipeline.py <source_video> <output_video>", file=sys.stderr)
        sys.exit(1)

    source_video = sys.argv[1]
    output_video = sys.argv[2]

    face_score    = face_similarity_score(source_video, output_video)
    lipsync_score = lipsync_confidence_score(output_video)
    total_score   = 0.5 * face_score + 0.5 * lipsync_score

    print(f"Face similarity:    {face_score:.3f}", file=sys.stderr)
    print(f"Lipsync confidence: {lipsync_score:.3f}", file=sys.stderr)
    print(f"Total score:        {total_score:.3f}", file=sys.stderr)

    # Only stdout output — Jenkins captures this as the score
    print(f"{total_score:.3f}")
