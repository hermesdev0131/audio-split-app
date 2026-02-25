"""
FastAPI application — handles uploads, job processing, and file downloads.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from html_parser import parse_html
from transcriber import transcribe, get_audio_duration, get_sample_rate
from aligner import align_blocks
from boundary_enforcer import enforce_boundaries, validate_boundaries
from audio_cutter import convert_to_wav_pcm, cut_segments
from packager import package_zip

app = FastAPI(title="Audio Auto-Cutter", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Job storage directory
JOBS_DIR = Path(__file__).parent.parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# Whisper model (configurable)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".flac"}
ALLOWED_HTML = {".html", ".htm"}


class JobStatus(str, Enum):
    PENDING = "pending"
    PARSING_HTML = "parsing_html"
    TRANSCRIBING = "transcribing"
    ALIGNING = "aligning"
    CUTTING = "cutting"
    PACKAGING = "packaging"
    COMPLETED = "completed"
    FAILED = "failed"


# In-memory job tracker (sufficient for MVP single-process)
jobs: dict[str, dict] = {}


def _update_job(job_id: str, **kwargs):
    if job_id in jobs:
        jobs[job_id].update(kwargs)
        # Persist status to disk for resilience
        status_path = JOBS_DIR / job_id / "status.json"
        with open(status_path, "w") as f:
            json.dump(jobs[job_id], f, default=str)


def process_job(job_id: str):
    """Main processing pipeline — runs as a background task."""
    job_dir = JOBS_DIR / job_id
    audio_path = str(job_dir / "audio_original")
    html_path = str(job_dir / "input.html")
    segments_dir = str(job_dir / "segments")
    wav_path = str(job_dir / "audio.wav")

    try:
        # Stage 1: Parse HTML
        _update_job(job_id, status=JobStatus.PARSING_HTML, progress=10)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        blocks = parse_html(html_content)
        _update_job(
            job_id,
            block_count=len(blocks),
            blocks_preview=[b.to_dict() for b in blocks[:3]],
            progress=20,
        )

        # Stage 2: Transcribe audio
        _update_job(job_id, status=JobStatus.TRANSCRIBING, progress=25)
        audio_duration = get_audio_duration(audio_path)
        sample_rate = get_sample_rate(audio_path)
        _update_job(job_id, audio_duration=audio_duration, sample_rate=sample_rate)

        words = transcribe(audio_path, model_name=WHISPER_MODEL)
        _update_job(job_id, progress=60, word_count=len(words))

        # Stage 3: Align blocks to timestamps
        _update_job(job_id, status=JobStatus.ALIGNING, progress=65)
        alignments = align_blocks(blocks, words, audio_duration)
        _update_job(job_id, progress=70)

        # Stage 4: Enforce boundaries
        segments = enforce_boundaries(alignments, audio_duration, sample_rate)
        invariants = validate_boundaries(segments, audio_duration, sample_rate)
        _update_job(job_id, progress=75, invariants=invariants)

        # Stage 5: Convert and cut audio
        _update_job(job_id, status=JobStatus.CUTTING, progress=80)
        convert_to_wav_pcm(audio_path, wav_path, sample_rate=sample_rate)
        segment_files = cut_segments(wav_path, segments, segments_dir, output_format="wav")
        _update_job(job_id, progress=90)

        # Stage 6: Package ZIP
        _update_job(job_id, status=JobStatus.PACKAGING, progress=95)
        zip_path = str(job_dir / "result.zip")
        package_zip(segment_files, segments, audio_duration, invariants, WHISPER_MODEL, zip_path)

        _update_job(
            job_id,
            status=JobStatus.COMPLETED,
            progress=100,
            zip_path=zip_path,
            segments=[
                {
                    "block_id": s.block_id,
                    "header": s.header,
                    "start_sec": round(s.start_sec, 3),
                    "end_sec": round(s.end_sec, 3),
                    "confidence": s.confidence,
                }
                for s in segments
            ],
        )

    except Exception as e:
        _update_job(job_id, status=JobStatus.FAILED, error=str(e))


# ── Routes ──────────────────────────────────────────────────────────────────


@app.post("/api/upload")
async def upload(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    html: UploadFile = File(...),
):
    """Accept audio + HTML files, create a job, start processing."""
    # Validate audio
    audio_ext = Path(audio.filename or "").suffix.lower()
    if audio_ext not in ALLOWED_AUDIO:
        raise HTTPException(400, f"Unsupported audio format: {audio_ext}. Allowed: {ALLOWED_AUDIO}")

    # Validate HTML
    html_ext = Path(html.filename or "").suffix.lower()
    if html_ext not in ALLOWED_HTML:
        raise HTTPException(400, f"Unsupported HTML format: {html_ext}. Allowed: {ALLOWED_HTML}")

    # Create job
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "segments").mkdir()

    # Save uploaded files
    audio_path = job_dir / ("audio_original" + audio_ext)
    with open(audio_path, "wb") as f:
        content = await audio.read()
        f.write(content)
    # Create symlink/copy with generic name for processing
    shutil.copy2(str(audio_path), str(job_dir / "audio_original"))

    html_path = job_dir / "input.html"
    with open(html_path, "wb") as f:
        content = await html.read()
        f.write(content)

    # Initialize job
    jobs[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "progress": 0,
        "created_at": datetime.utcnow().isoformat(),
        "audio_filename": audio.filename,
        "html_filename": html.filename,
    }
    _update_job(job_id, status=JobStatus.PENDING, progress=5)

    # Start background processing
    background_tasks.add_task(process_job, job_id)

    return {"job_id": job_id, "status": "pending"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll job status."""
    if job_id not in jobs:
        # Try loading from disk
        status_path = JOBS_DIR / job_id / "status.json"
        if status_path.exists():
            with open(status_path) as f:
                jobs[job_id] = json.load(f)
        else:
            raise HTTPException(404, "Job not found")

    return jobs[job_id]


@app.get("/api/download/{job_id}")
async def download(job_id: str):
    """Download the result ZIP."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    job = jobs[job_id]
    if job.get("status") != JobStatus.COMPLETED:
        raise HTTPException(400, f"Job not ready. Status: {job.get('status')}")

    zip_path = job.get("zip_path")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(404, "ZIP file not found")

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"audio_segments_{job_id[:8]}.zip",
    )


# Serve frontend static files
frontend_dir = Path(__file__).parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
