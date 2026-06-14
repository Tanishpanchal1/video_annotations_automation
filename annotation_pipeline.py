"""
Automated Annotation System
============================
Usage:
    python annotation_pipeline.py --image question.png --audio narration.mp3
 
Requires:
    pip install -r requirements.txt
    ffmpeg installed on system
    GROQ_API_KEY in .env file
"""
 
import os
import sys
import json
import math
import random
import argparse
import subprocess
import tempfile
import base64

# Fix for Python 3.13+ where audioop was removed
# Add a stub module to sys.modules before importing pydub
import sys
if sys.version_info >= (3, 13):
    try:
        import audioop
    except ImportError:
        # Create a stub audioop module for pydub compatibility
        import types
        audioop_stub = types.ModuleType('audioop')
        sys.modules['audioop'] = audioop_stub
 
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pydub import AudioSegment
from dotenv import load_dotenv, parser
from groq import Groq
 
load_dotenv()

# Platform-specific ffmpeg setup
# On Arch/Linux: ffmpeg is typically in /usr/bin or /usr/local/bin via pacman
# On macOS: ffmpeg is in standard paths via homebrew
# Only add custom paths if they exist
import platform
ffmpeg_bin = None
if platform.system() == 'Windows':
    # Windows: check for ffmpeg installed via winget or chocolatey
    ffmpeg_bin = os.path.join(
        os.getenv('LOCALAPPDATA', ''),
        'Microsoft', 'WinGet', 'Packages',
        'Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe',
        'ffmpeg-8.1.1-full_build', 'bin'
    )
elif platform.system() == 'Darwin':
    # macOS: check common homebrew paths
    ffmpeg_bin = '/usr/local/bin'

if ffmpeg_bin and os.path.isdir(ffmpeg_bin):
    os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + ffmpeg_bin
else:
    # On Linux/Arch, ffmpeg is typically in /usr/bin via pacman, no action needed
    if platform.system() != 'Linux':
        print(f'⚠️  WARNING: Expected ffmpeg bin not found at {ffmpeg_bin}')

def sync_annotations_with_transcript(annotations: list, segments: list) -> list:
    """Synchronize annotation timestamps with transcript segments.
    For each annotation, find the first transcript segment whose text contains the annotation's displayed text.
    This provides tighter alignment with the audio narration.
    """
    for ann in annotations:
        text = ann.get('text', '').lower()
        matched = False
        if text:
            for seg in segments:
                seg_text = seg.get('text', '').lower()
                if text in seg_text:
                    ann['time_start'] = seg.get('start', 0)
                    ann['time_end'] = seg.get('end', 0)
                    matched = True
                    break
        if not matched:
            # fallback to existing times or defaults
            ann['time_start'] = ann.get('time_start', 0)
            ann['time_end'] = ann.get('time_end', ann.get('time_start', 0) + 2.0)
    return annotations

# ── CONFIG ────────────────────────────────────────────────
FPS = 24
# Use a Windows-safe temp path in the script directory
TEMP_VIDEO = os.path.join(os.path.abspath(os.path.dirname(__file__)), "annotation_temp.mp4")

# ── GROQ CLIENT ───────────────────────────────────────────
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── PREFLIGHT CHECKS ────────────────────────────────────
def validate_environment():
    """Check required API key and system tools before running pipeline."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key.strip() == "":
        raise RuntimeError(
            "ERROR: GROQ_API_KEY not set or empty.\n"
            "  1. Get a free key at https://console.groq.com\n"
            "  2. Create a .env file in this directory with:\n"
            "     GROQ_API_KEY=gsk_your_actual_key_here\n"
            "  3. Restart the app."
        )
    
    # Check for ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        if result.returncode != 0:
            print("WARNING: ffmpeg found but returned error. Video rendering may fail.")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("WARNING: ffmpeg not found in PATH. Video rendering will fail.")
        print("  Install: winget install ffmpeg")


# CLI helpers and UI progress emitter
def emit_progress(step, status='running', detail=None):
    payload = {'step': step, 'status': status}
    if detail is not None:
        payload['detail'] = detail
    print('PROGRESS:' + json.dumps(payload), flush=True)


# ══════════════════════════════════════════════════════════
# STEP 1 — TRANSCRIBE AUDIO WITH WHISPER VIA GROQ
# ══════════════════════════════════════════════════════════
def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio using Whisper via Groq API.
    Returns segments with start/end timestamps.
    """
    print("\n[1/5] Transcribing audio...")
    emit_progress('transcribe', 'running')
 
    # Convert to mp3 if needed (Groq Whisper accepts mp3/wav/m4a/mpeg)
    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in [".mp3", ".wav", ".m4a", ".mpeg", ".mp4", ".webm"]:
        print("      Converting audio format...")
        audio = AudioSegment.from_file(audio_path)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio.export(tmp.name, format="mp3")
        audio_path = tmp.name
 
    with open(audio_path, "rb") as f:
        transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model="whisper-large-v3",
            response_format="verbose_json",
            timestamp_granularities=["segment", "word"],
        )

    # Support multiple return shapes (object-like or plain dict)
    segments = []
    segments_raw = []
    if hasattr(transcription, 'segments'):
        try:
            segments_raw = getattr(transcription, 'segments') or []
        except Exception:
            segments_raw = []
    elif isinstance(transcription, dict):
        segments_raw = transcription.get('results', transcription.get('segments', []))

    for seg in segments_raw or []:
        if hasattr(seg, 'start') or hasattr(seg, 'end'):
            start = getattr(seg, 'start', None)
            end = getattr(seg, 'end', None)
            text = getattr(seg, 'text', '')
        elif isinstance(seg, dict):
            start = seg.get('start')
            end = seg.get('end')
            text = seg.get('text', '')
        else:
            start = None
            end = None
            text = str(seg)

        segments.append({
            "start": start if start is not None else 0.0,
            "end": end if end is not None else 0.0,
            "text": (text or '').strip()
        })
 
    # Get total duration
    audio_obj = AudioSegment.from_file(audio_path)
    duration = len(audio_obj) / 1000.0
 
    print(f"      SUCCESS: Transcribed {len(segments)} segments, {duration:.1f}s total")
    emit_progress('transcribe', 'done', {'segments': len(segments), 'duration': duration})
    for seg in segments[:3]:
        print(f"        [{seg['start']:.1f}s] {seg['text'][:60]}")
    if len(segments) > 3:
        print(f"        ... and {len(segments)-3} more")
 
    return {"segments": segments, "duration": duration}
 
 
# ══════════════════════════════════════════════════════════
# STEP 2 — UNDERSTAND IMAGE WITH GROQ VISION
# ══════════════════════════════════════════════════════════
def analyze_image(image_path: str) -> dict:
    """
    Use Groq vision model to understand the question image —
    what's on it, where things are, what space is available.
    """
    print("\n[2/5] Analyzing image with Groq Vision...")
    emit_progress('analyze', 'running')
 
    # Read and encode image
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")
 
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    mime = f"image/{'jpeg' if ext in ['jpg','jpeg'] else ext}"
 
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    H, W = img_cv.shape[:2]
 
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"}
                },
                {
                    "type": "text",
                    "text": f"""Analyze this educational question image (width={W}px, height={H}px).
 
Respond ONLY with JSON, no markdown, no preamble:
{{
  "question_text": "full question text",
  "question_region": {{"x1": int, "y1": int, "x2": int, "y2": int}},
  "answer_options": [
    {{"label": "A", "text": "...", "x": int, "y": int}},
    {{"label": "B", "text": "...", "x": int, "y": int}},
    {{"label": "C", "text": "...", "x": int, "y": int}},
    {{"label": "D", "text": "...", "x": int, "y": int}}
  ],
  "correct_answer": "C",
  "working_area": {{"x": int, "y": int, "width": int, "height": int}},
  "subject": "math/physics/chemistry/etc",
  "topic": "specific topic e.g. distance formula"
}}"""
                }
            ]
        }],
        max_tokens=800
    )
 
    # Retrieve raw content with fallbacks
    raw = None
    if hasattr(response, 'choices'):
        try:
            raw = response.choices[0].message.content
        except Exception:
            raw = str(response)
    elif isinstance(response, dict):
        try:
            raw = response.get('choices', [{}])[0].get('message', {}).get('content')
        except Exception:
            raw = str(response)
    else:
        raw = str(response)

    clean = (raw or "").replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(clean) if clean else {}
    except Exception:
        result = {}
    result["image_width"] = W
    result["image_height"] = H
    print(f"      SUCCESS: Question: {result.get('question_text','')[:70]}...")
    print(f"      SUCCESS: Subject: {result.get('subject')} | Topic: {result.get('topic')}")
    print(f"      SUCCESS: Correct answer: {result.get('correct_answer')}")
    emit_progress('analyze', 'done', {'correct_answer': result.get('correct_answer')})
    return result
 
 
# ══════════════════════════════════════════════════════════
# STEP 3 — PLAN ANNOTATIONS WITH GROQ LLM
# ══════════════════════════════════════════════════════════
def plan_annotations(image_info: dict, transcript: dict) -> list:
    """
    Use Groq LLM to generate a timed annotation plan —
    what to draw, where, and when — based on transcript + image layout.
    """
    print("\n[3/5] Planning annotations with Groq LLM...")
    emit_progress('plan', 'running')
 
    W = image_info["image_width"]
    H = image_info["image_height"]
    duration = transcript["duration"]
    working = image_info.get("working_area", {"x": 30, "y": 280, "width": 600, "height": 250})
 
    # Compute the right-column start position: beside options, starting at top option's y
    options_list = image_info.get('answer_options', [])
    if options_list:
        option_ys = [o.get('y', 150) for o in options_list if o.get('y') is not None]
        annotation_start_y = min(option_ys) if option_ys else 130
        # Find right edge of options text to determine left boundary of annotation column
        option_xs = [o.get('x', 30) for o in options_list if o.get('x') is not None]
        max_option_text_width = 120  # approximate width of "(A) 3 units"
        annotation_x = (max(option_xs) if option_xs else 30) + max_option_text_width + 20
    else:
        annotation_start_y = 130
        annotation_x = 220
    annotation_x = max(annotation_x, 250)  # ensure at least 200px from left

    prompt = f"""You are an annotation planner for educational math videos.

IMAGE LAYOUT:
- Image size: {W}x{H}px
- Question: "{image_info.get('question_text')}"
- Answer options: {json.dumps(image_info.get('answer_options', []), indent=2)}
- Correct answer: option {image_info.get('correct_answer')}
- Answer options are on the LEFT side. The RIGHT side (x >= {annotation_x}) is EMPTY.
- All write_text annotations must use x={annotation_x}.


AUDIO TRANSCRIPT SEGMENTS:
{json.dumps(transcript['segments'], indent=2)}

TASK: Generate EXACTLY 6 write_text annotations for the solution steps in order.

CRITICAL RULES — FOLLOW EXACTLY:
1. Every annotation must have a UNIQUE time_start. NO two annotations can share the same time.
   Stagger them by at least 2-3 seconds apart.
2. For the substitution step: the text MUST start with √ (include the square root).
   Write: "√((4-1)² + (6-2)²)". This appears at ~30 seconds.
3. Each subsequent step is a NEW annotation line below the previous one.
4. Write ONLY math symbols/numbers — NO English words in the text field, except for the final answer which should include the word "units" (e.g. "d = 5 units").



REQUIRED ANNOTATION SEQUENCE (match to transcript segments):
  Step 1 (~13.4s): "d =  √ ((x₂ - x₁)² + (y₂ - y₁)²)"  font_scale: 1
  Step 2 (~31.0s): "√((4 - 1)² + (6 - 2)²)"              font_scale: 1
  Step 3 (~41.6s): "√(3² + 4²)"                      font_scale: 1
  Step 4 (~44.6s): "√(9 + 16)"                        font_scale: 1
  Step 5 (~48.0s): "√25"                               font_scale: 1
  Step 6 (~59.3s): "d = 5 units"                         font_scale: 1



Output ONLY a JSON array (no markdown, no explanation):
[
  {{
    "time_start": 13.4,
    "time_end": 29.0,
    "type": "write_text",
    "x": {annotation_x},
    "y": 0,
    "text": "d = √((x₂ - x₁)² + (y₂ - y₁)²)",
    "color": [0, 0, 0],
    "font_scale": 0.7,
    "label": "write formula"
  }},
  {{
    "time_start": 31.0,
    "time_end": 41.6,
    "type": "write_text",
    "x": {annotation_x},
    "y": 0,
    "text": "√((4 - 1)² + (6 - 2)²)",
    "color": [0, 0, 0],
    "font_scale": 0.7,
    "label": "substitute values"
  }},

  {{
    "time_start": 41.6,
    "time_end": 44.6,
    "type": "write_text",
    "x": {annotation_x},
    "y": 0,
    "text": "√(3² + 4²)",
    "color": [0, 0, 0],
    "font_scale": 0.7,
    "label": "calculate differences"
  }},
  {{
    "time_start": 44.6,
    "time_end": 48.0,
    "type": "write_text",
    "x": {annotation_x},
    "y": 0,
    "text": "√(9 + 16)",
    "color": [0, 0, 0],
    "font_scale": 0.7,
    "label": "calculate squares"
  }},
  {{
    "time_start": 48.0,
    "time_end": 59.3,
    "type": "write_text",
    "x": {annotation_x},
    "y": 0,
    "text": "√25",
    "color": [0, 0, 0],
    "font_scale": 0.7,
    "label": "sum under root"
  }},
  {{
    "time_start": 59.3,
    "time_end": 65.0,
    "type": "write_text",
    "x": {annotation_x},
    "y": 0,
    "text": "d = 5 units",
    "color": [0, 0, 0],
    "font_scale": 0.7,
    "label": "final answer"
  }}
]"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000
    )

    if hasattr(response, 'choices'):
        try:
            raw = response.choices[0].message.content
        except Exception:
            raw = str(response)
    elif isinstance(response, dict):
        try:
            raw = response.get('choices', [{}])[0].get('message', {}).get('content')
        except Exception:
            raw = str(response)
    else:
        raw = str(response)

    clean = (raw or "").replace("```json", "").replace("```", "").strip()
    annotations = []
    if clean:
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict):
                if isinstance(parsed.get('annotations'), list):
                    annotations = parsed['annotations']
                elif isinstance(parsed.get('steps'), list):
                    annotations = parsed['steps']
                else:
                    annotations = [parsed]
            else:
                annotations = parsed if isinstance(parsed, list) else []
        except Exception:
            annotations = []

        # Ensure we have a list of annotation dicts
        if not isinstance(annotations, list):
            annotations = []

    # Enforce "d = 5 units" on the final result for safety/robustness
    for ann in annotations:
        txt = str(ann.get("text", "")).strip()
        if txt in ("= 5", "5", "= 5 units", "5 units", "d = 5", "d = 5.0", "d = 5.0 units", "d = 5 units"):
            ann["text"] = "d = 5 units"

    print(f"      SUCCESS: {len(annotations)} annotations planned:")
    emit_progress('plan', 'done', {'count': len(annotations)})
    for a in annotations:
        print(f"        t={a.get('time_start',0):.1f}s -> [{a.get('type')}] {a.get('label','')}")
    # Sync annotation times to audio transcript
    annotations = sync_annotations_with_transcript(annotations, transcript.get('segments', []))

    # ── STAGGER simultaneous annotations ───────────────────────────
    # If two annotations share the same time_start, push later ones forward
    annotations.sort(key=lambda a: a.get('time_start', 0))
    for i in range(1, len(annotations)):
        prev_t = annotations[i-1].get('time_start', 0)
        curr_t = annotations[i].get('time_start', 0)
        if curr_t <= prev_t:  # same or out-of-order
            annotations[i]['time_start'] = prev_t + 2.0
            if annotations[i].get('time_end', 0) <= annotations[i]['time_start']:
                annotations[i]['time_end'] = annotations[i]['time_start'] + 2.0

    # ── POSITION OVERRIDE ──────────────────────────────────────────
    # All write_text/underroot stacked in the RIGHT column.
    # LINE_SPACING is computed dynamically so all lines fit on screen.
    options_list = image_info.get('answer_options', [])
    if options_list:
        option_ys = [o.get('y', 150) for o in options_list if o.get('y') is not None]
        base_y = max(20, min(option_ys) - 30) if option_ys else 100
        option_xs = [o.get('x', 30) for o in options_list if o.get('x') is not None]
        base_x = max(option_xs) + 130
    else:
        base_y = 100
        base_x = 220
    base_x = max(base_x, 250)

    # Count how many stacked lines we need
    stack_count = sum(1 for a in annotations if a.get('type') in ('write_text', 'underroot'))
    # Define gaps to visually separate annotation steps
    gap_after_3rd = 15
    gap_after_4th = 15
    gap_after_5th = 15
    gap_before_result = 20
    total_gaps = gap_after_3rd + gap_after_4th + gap_after_5th + gap_before_result
    
    # Calculate spacing so all lines fit within the image height, subtracting the gaps
    available_h = H - base_y - 20 - total_gaps  # leave 20px bottom margin & account for gaps
    if stack_count > 1:
        LINE_SPACING = min(70, max(50, available_h // stack_count))
    else:
        LINE_SPACING = 70

    write_index = 0
    processed = []
    for ann in annotations:
        atype = ann.get('type', '')
        if atype in ('write_text', 'underroot'):
            ann['x'] = base_x


            extra_y = 0
            if write_index >= 3:
                extra_y += gap_after_3rd
            if write_index >= 4:
                extra_y += gap_after_4th
            if write_index >= 5:
                extra_y += gap_after_5th
            if write_index == stack_count - 1:
                extra_y += gap_before_result
            ann['y'] = base_y + write_index * LINE_SPACING + extra_y
            write_index += 1
        processed.append(ann)

    # ── DRAW BOX ON THE FINAL ANSWER ───────────────────────────────
    # Find the final "d = 5 units" annotation to draw a hand-drawn box around it
    result_ann = None
    for ann in processed:
        if ann.get('type') == 'write_text' and '5 units' in ann.get('text', ''):
            result_ann = ann
            break
            
    if result_ann:
        rx = result_ann['x']
        ry = result_ann['y']
        rscale = result_ann.get('font_scale', 0.9)
        rfont_size = int(rscale * 56)
        # Load font to measure text width/height
        try:
            font_path = os.path.join(os.path.dirname(__file__), "Caveat.ttf")
            rfont = ImageFont.truetype(font_path, rfont_size)
        except Exception:
            rfont = ImageFont.load_default()
            
        # Measure bounding box of "d = 5 units"
        padding_x = 18
        padding_y = 12
        try:
            mask = rfont.getmask(result_ann['text'])
            bbox = mask.getbbox()
            if bbox:
                left, top, right, bottom = bbox
                x1 = rx + left - padding_x
                x2 = rx + right + padding_x
                y1 = ry + top - padding_y
                y2 = ry + bottom + padding_y
            else:
                tw = len(result_ann['text']) * (rfont_size * 0.5)
                th = rfont_size
                x1 = rx - padding_x
                x2 = rx + tw + padding_x
                y1 = ry - padding_y
                y2 = ry + th + padding_y
        except Exception:
            tw = len(result_ann['text']) * (rfont_size * 0.5)
            th = rfont_size
            x1 = rx - padding_x
            x2 = rx + tw + padding_x
            y1 = ry - padding_y
            y2 = ry + th + padding_y
        
        # Schedule the highlighter to start slightly before the result text is fully drawn
        highlighter_start = result_ann['time_start'] + 2.0
        highlighter_end = highlighter_start + 3.0
        
        highlight_ann = {
            "time_start": highlighter_start,
            "time_end": highlighter_end,
            "type": "highlight",
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "color": [255, 255, 0],  # Yellow highlighter (RGB: [255, 255, 0] -> BGR: [0, 255, 255])
            "label": "final answer highlight"
        }
        processed.append(highlight_ann)
        
        # Schedule the box to start right after the text finishes drawing
        box_start = result_ann['time_start'] + 5.0
        box_end = box_start + 2.0

        
        box_ann = {
            "time_start": box_start,
            "time_end": box_end,
            "type": "box",
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "color": [0, 0, 0],
            "label": "final answer box"
        }
        processed.append(box_ann)

    # ── UNDERLINE POINTS A AND B IN THE QUESTION TEXT ──────────────
    # From image analysis: question is near top, points A(1,2) and B(4,6)
    # Use question_region to approximate positions of the point coordinates
    q_region = image_info.get('question_region', {})
    q_y2 = q_region.get('y2', 80)  # bottom of question text line
    q_x1 = q_region.get('x1', 30)
    # Approximate positions based on the question text layout
    # "A (1, 2)" is roughly at 60-70% of question width
    # "(4, 6)" is near the end of the question text
    q_width = q_region.get('x2', W - 100) - q_x1
    
    # Underline "A (1, 2)" — appears early in the video
    a_underline_x1 = q_x1 + int(q_width * 0.53)
    a_underline_x2 = q_x1 + int(q_width * 0.72)
    a_underline_y = q_y2 - 5
    processed.append({
        "time_start": 2.0,
        "time_end": 10.0,
        "type": "underline",
        "x1": a_underline_x1,
        "y1": a_underline_y,
        "x2": a_underline_x2,
        "y2": a_underline_y,
        "color": [255, 0, 0],  # Red underline
        "label": "underline point A"
    })
    
    # Underline "(4, 6)" — the B point
    b_underline_x1 = q_x1 + int(q_width * 0.78)
    b_underline_x2 = q_x1 + int(q_width * 0.93)
    b_underline_y = q_y2 - 5
    processed.append({
        "time_start": 3.0,
        "time_end": 10.0,
        "type": "underline",
        "x1": b_underline_x1,
        "y1": b_underline_y,
        "x2": b_underline_x2,
        "y2": b_underline_y,
        "color": [255, 0, 0],  # Red underline
        "label": "underline point B"
    })

    return processed


 
 
# ══════════════════════════════════════════════════════════
# STEP 4 — RENDER ANNOTATION VIDEO WITH OPENCV
# ══════════════════════════════════════════════════════════
def draw_stroke(img, pts, color, thickness, progress=1.0):
    """Draw a polyline through a list of points."""
    if len(pts) < 2:
        return img
    
    total_segments = len(pts) - 1
    segments_to_draw = int(total_segments * progress)
    
    for i in range(segments_to_draw):
        cv2.line(img, pts[i], pts[i+1], color, thickness, cv2.LINE_AA)
        
    if progress < 1.0 and segments_to_draw < total_segments:
        seg_prog = (total_segments * progress) - segments_to_draw
        if seg_prog > 0:
            x1, y1 = pts[segments_to_draw]
            x2, y2 = pts[segments_to_draw+1]
            px = int(x1 + (x2 - x1) * seg_prog)
            py = int(y1 + (y2 - y1) * seg_prog)
            cv2.line(img, (x1, y1), (px, py), color, thickness, cv2.LINE_AA)
    return img
 
def get_hand_line_pts(x1, y1, x2, y2):
    steps = max(abs(x2-x1), abs(y2-y1), 10)
    pts = []
    if steps == 0:
        return [(x1, y1)]
    for i in range(int(steps)+1):
        t_step = i / steps
        x = int(x1 + (x2-x1)*t_step + random.gauss(0, 0.1))
        y = int(y1 + (y2-y1)*t_step + random.gauss(0, 0.1))
        pts.append((x, y))
    return pts

def hand_line(img, x1, y1, x2, y2, color, thickness=2, progress=1.0):
    """Slightly wobbly line for handwritten feel."""
    pts = get_hand_line_pts(x1, y1, x2, y2)
    return draw_stroke(img, pts, color, thickness, progress)
 


def draw_annotation(img, ann, H, W, t=None):
    """Draw a single annotation onto the image."""
    # Convert RGB to BGR for OpenCV
    c = tuple(reversed(ann.get("color", [0, 0, 0])))  # default black
    atype = ann.get("type", "")

    draw_duration = ann.get("duration", 3.0)
    if t is not None:
        progress = max(0.0, min(1.0, (t - ann.get("time_start", 0)) / draw_duration))
    else:
        progress = 1.0

    if atype == "highlight":
        x1, y1 = max(0, ann.get("x1", 0)), max(0, ann.get("y1", 0))
        x2, y2 = min(W, ann.get("x2", W)), min(H, ann.get("y2", H))
        overlay = img.copy()
        cy = (y1 + y2) // 2
        thickness = max(6, (y2 - y1) - 4)
        overlay = hand_line(overlay, x1, cy, x2, cy, c, thickness, progress)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    elif atype == "write_text":
        base_x = ann.get("x", 450)
        base_y = ann.get("y", 120)
        scale = ann.get("font_scale", 0.9)
        font_size = int(scale * 40)  # reduced font size for layout optimization

        # Positions are pre-computed in plan_annotations — use directly
        x = max(5, min(base_x, W - 10))
        y = max(20, min(base_y, H - 5))

        text = str(ann.get("text", ""))

        # ── Keep subscript Unicode for proper mathematical notation ──
        # text subscripts are preserved as-is

        # ── Fix: render any prefix (e.g. "d = ") BEFORE the √ symbol ──────────
        prefix_text = ""
        for sqrt_token in ("√(", "sqrt(", "√"):
            if sqrt_token in text:
                idx = text.index(sqrt_token)
                if idx > 0:
                    prefix_text = text[:idx].replace("=", " = ").rstrip()  # e.g. "d ="
                    text = text[idx:].replace("=", " = ")                  # e.g. "√((x2-x1)²...)"
                break

        # Draw the prefix ("d =") as plain text first
        if prefix_text:
            try:
                font_path = os.path.join(os.path.dirname(__file__), "Caveat.ttf")
                pfont = ImageFont.truetype(font_path, font_size)
            except Exception:
                pfont = ImageFont.load_default()
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pdraw = ImageDraw.Draw(img_pil)
            rgb_c = tuple(ann.get("color", [0, 0, 0]))
            pdraw.text((x, y), prefix_text + " ", font=pfont, fill=rgb_c)
            # Measure prefix width so the root part starts after it
            pbbox = pdraw.textbbox((x, y), prefix_text + " ", font=pfont)
            x = pbbox[2]  # shift x to right of prefix
            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        # Math symbols translation
        text = text.replace("^2", "²").replace("^3", "³")
        draw_root = False
        if "sqrt(" in text or "√(" in text:
            # Strip only the sqrt wrapper — keep INNER brackets intact
            # e.g. "√((4-1)² + (6-2)²)" → "(4-1)² + (6-2)²"
            text = text.replace("sqrt(", "").replace("√(", "")
            # Remove only the single outermost trailing ")"
            if text.endswith(")"):
                text = text[:-1]
            draw_root = True
        elif "√" in text or "sqrt" in text.lower():
            # Handle √ without parentheses, e.g. "√25" → "25" + draw root
            text = text.replace("√", "").replace("sqrt", "").strip()
            draw_root = True
        else:
            text = text.replace("???", "").replace("underroot", "")
        
        try:
            font_path = os.path.join(os.path.dirname(__file__), "Caveat.ttf")
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
            
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil, "RGBA")
        
        if text:
            full_bbox = draw.textbbox((x, y), text, font=font)
            tw = full_bbox[2] - full_bbox[0]
            th = full_bbox[3] - full_bbox[1]
        else:
            tw, th = 0, font_size
            
        rgb_c = tuple(ann.get("color", [0, 0, 0]))
        
        text_progress = progress / 0.7 if draw_root else progress
        text_progress = min(1.0, max(0.0, text_progress))
        
        mask_w = int(tw * text_progress)
        if mask_w > 0:
            txt_img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((x, y), text, font=font, fill=rgb_c + (255,))
            crop_box = (0, 0, x + mask_w + 20, H)
            txt_cropped = txt_img.crop(crop_box)
            img_pil.paste(txt_cropped, (0, 0), txt_cropped)
        
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        if draw_root:
            root_progress = (progress - 0.7) / 0.3
            root_progress = min(1.0, max(0.0, root_progress))
            if root_progress > 0:
                rx1 = x - 20
                ry1 = y + th // 2
                rx2 = x - 10
                ry2 = y + th + 5
                rx3 = x
                ry3 = y - 5
                rx4 = x + tw + 10
                ry4 = y - 5
                
                pts = get_hand_line_pts(rx1, ry1, rx2, ry2) + get_hand_line_pts(rx2, ry2, rx3, ry3) + get_hand_line_pts(rx3, ry3, rx4, ry4)
                img = draw_stroke(img, pts, c, 1, root_progress)

    elif atype == "underline":
        x1, y1 = ann.get("x1", 0), ann.get("y1", 0)
        x2, y2 = ann.get("x2", 100), ann.get("y2", y1)
        img = hand_line(img, x1, y1 + 4, x2, y2 + 4, c, 1, progress)

    elif atype == "circle":
        cx = max(20, min(W - 20, ann.get("cx", W // 4)))
        cy = max(20, min(H - 20, ann.get("cy", H // 2)))
        rx = max(10, min(W // 3, ann.get("rx", 40)))
        ry = max(8, min(H // 4, ann.get("ry", 15)))
        pts = []
        for i in range(105):
            angle = 2 * math.pi * i / 100
            jitter = random.gauss(1.0, 0.015)
            px = int(cx + rx * math.cos(angle) * jitter)
            py = int(cy + ry * math.sin(angle) * jitter)
            pts.append((px, py))
        img = draw_stroke(img, pts, c, 2, progress)

    elif atype == "arrow":
        x1, y1 = ann.get("x1", 100), ann.get("y1", 300)
        x2, y2 = ann.get("x2", 200), ann.get("y2", 300)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        angle = math.atan2(y2 - y1, x2 - x1)
        L = 15
        p1_x = int(x2 - L * math.cos(angle - math.pi/6))
        p1_y = int(y2 - L * math.sin(angle - math.pi/6))
        p2_x = int(x2 - L * math.cos(angle + math.pi/6))
        p2_y = int(y2 - L * math.sin(angle + math.pi/6))
        pts = get_hand_line_pts(x1, y1, x2, y2) + get_hand_line_pts(x2, y2, p1_x, p1_y) + get_hand_line_pts(x2, y2, p2_x, p2_y)
        img = draw_stroke(img, pts, c, 1, progress)

    elif atype == "checkmark":
        x, y = ann.get("x", 400), ann.get("y", 300)
        pts = get_hand_line_pts(x, y + 10, x + 8, y + 20) + get_hand_line_pts(x + 8, y + 20, x + 22, y)
        img = draw_stroke(img, pts, c, 2, progress)

    elif atype == "box":
        x1, y1 = max(0, ann.get("x1", 0)), max(0, ann.get("y1", 0))
        x2, y2 = min(W, ann.get("x2", W)), min(H, ann.get("y2", H))
        pts = get_hand_line_pts(x1, y1, x2, y1) + get_hand_line_pts(x2, y1, x2, y2) + get_hand_line_pts(x2, y2, x1, y2) + get_hand_line_pts(x1, y2, x1, y1)
        img = draw_stroke(img, pts, c, 1, progress)

    elif atype == "cross":
        cx, cy = ann.get("x", W // 2), ann.get("y", H // 2)
        size = ann.get("size", 20)
        half = size // 2
        pts = get_hand_line_pts(cx - half, cy - half, cx + half, cy + half) + get_hand_line_pts(cx - half, cy + half, cx + half, cy - half)
        img = draw_stroke(img, pts, c, 1, progress)

    elif atype == "square":
        cx, cy = ann.get("x", W // 2), ann.get("y", H // 2)
        size = ann.get("size", 40)
        half = size // 2
        pts = get_hand_line_pts(cx - half, cy - half, cx + half, cy - half) + get_hand_line_pts(cx + half, cy - half, cx + half, cy + half) + get_hand_line_pts(cx + half, cy + half, cx - half, cy + half) + get_hand_line_pts(cx - half, cy + half, cx - half, cy - half)
        img = draw_stroke(img, pts, c, 1, progress)

    elif atype == "underroot":
        y_index = ann.get("y_index", 0)
        x = ann.get("x", 450)
        scale = ann.get("font_scale", 0.9)
        font_size = int(scale * 40)
        y = ann.get("y", 120 + y_index * (font_size + 5)) + 15
        size = ann.get("size", 50)
        thickness = ann.get("thickness", 2)
        pts = get_hand_line_pts(x, y, x, y + size // 2) + get_hand_line_pts(x, y + size // 2, x + size // 2, y - size // 2) + get_hand_line_pts(x + size // 2, y - size // 2, x + size, y - size // 2)
        img = draw_stroke(img, pts, c, thickness, progress)

    return img


 

def _get_fourcc(code):
    """Get a VideoWriter fourcc code, with fallbacks."""
    try:
        return cv2.VideoWriter_fourcc(*code)
    except Exception:
        pass
    if hasattr(cv2, 'CV_FOURCC'):
        try:
            return cv2.CV_FOURCC(*code)
        except Exception:
            pass
    return 0


def render_video(image_path: str, annotations: list, duration: float, output_path: str):
    """
    Render all frames with annotations appearing progressively.
    Annotations are cumulative — once drawn they stay.
    """
    print(f"\n[4/5] Rendering {int(duration * FPS)} frames at {FPS}fps...")
    emit_progress('render', 'running')
 
    bg = cv2.imread(image_path)
    if bg is None:
        raise FileNotFoundError(f"Background image not found or unreadable: {image_path}")
    H, W = bg.shape[:2]
 
    # Sort annotations by time
    annotations = sorted(annotations, key=lambda a: a.get("time_start", 0))

    write_count = 0
    for i, ann in enumerate(annotations):
        t_start = ann.get("time_start", 0)
        t_end = ann.get("time_end", t_start + 2.0)
        # Annotation writing duration: 85% of the original interval (slightly longer than before)
        original_interval = t_end - t_start
        fast_interval = max(0.5, original_interval * 0.85)
        ann["duration"] = fast_interval
            
        if ann.get("type") in ["write_text", "underroot"]:
            pass  # y positions are pre-computed in plan_annotations

    fourcc = _get_fourcc("mp4v")
    out = cv2.VideoWriter(output_path, fourcc, FPS, (W, H))
    if not getattr(out, 'isOpened', lambda: True)():
        raise RuntimeError(f"cv2.VideoWriter failed to open output: {output_path}")
 
    random.seed(42)  # Reproducible jitter
    total_frames = int(duration * FPS)
 
    for frame_idx in range(total_frames):
        t = frame_idx / FPS
        frame = bg.copy()
 
        # Draw all annotations that have started by time t
        for ann in annotations:
            if ann.get("time_start", 0) <= t:
                frame = draw_annotation(frame, ann, H, W, t)
 
        out.write(frame)
 
        if frame_idx % (FPS * 10) == 0:
            print(f"      {t:.0f}s / {duration:.0f}s rendered...")
 
    out.release()
    print(f"      [OK] Video frames written")
    emit_progress('render', 'done', {'frames': total_frames})
 
 
# ══════════════════════════════════════════════════════════
# STEP 5 — COMBINE VIDEO + AUDIO WITH FFMPEG
# ══════════════════════════════════════════════════════════
def combine_audio_video(video_path: str, audio_path: str, output_path: str):
    """Mux annotation video with original audio using ffmpeg."""
    print(f"\n[5/5] Combining video + audio...")
    emit_progress('compose', 'running')
 
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("      ✗ ffmpeg error:", result.stderr[-500:])
        raise RuntimeError("ffmpeg failed")
 
    size_mb = os.path.getsize(output_path) / (1024*1024)
    print(f"      [OK] Output saved: {output_path} ({size_mb:.1f} MB)")
    emit_progress('compose', 'done', {'size_mb': size_mb})
 
 
# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Automated Annotation System")
    
    
    # parser.add_argument("--image", required=True, help="Path to question image (PNG/JPG)")
    # parser.add_argument("--audio", required=True, help="Path to audio narration (MP3/WAV/MPEG)")
    
    
    
    # Default files in uploads folder
    default_image = os.path.join("uploads", "image.png")
    default_audio = os.path.join("uploads", "Audio.mpeg")  # change extension if this is actually an audio file

    parser.add_argument(
        "--image",
        default=default_image,
        help=f"Path to question image (default: {default_image})"
    )

    parser.add_argument(
        "--audio",
        default=default_audio,
        help=f"Path to audio narration (default: {default_audio})"
    )
    
    parser.add_argument("--output", default="annotated_output_result.mp4", help="Output video path")
    args = parser.parse_args()
 
    # Validate environment (API key, ffmpeg, etc)
    try:
        validate_environment()
    except RuntimeError as e:
        print(str(e))
        sys.exit(1)
    
    # Validate file inputs
    if not os.path.exists(args.image):
        print(f"✗ Image not found: {args.image}"); sys.exit(1)
    if not os.path.exists(args.audio):
        print(f"✗ Audio not found: {args.audio}"); sys.exit(1)
 
    print("=" * 55)
    print("  Automated Annotation System")
    print(f"  Image : {args.image}")
    print(f"  Audio : {args.audio}")
    print(f"  Output: {args.output}")
    print("=" * 55)
 
    # Run pipeline
    transcript   = transcribe_audio(args.audio)
    image_info   = analyze_image(args.image)
    annotations  = plan_annotations(image_info, transcript)
    # Shift all annotations to start after 4 seconds
    for ann in annotations:
        ann['time_start'] = ann.get('time_start', 0) + 4.0
        ann['time_end'] = ann.get('time_end', ann.get('time_start', 0) + 2.0) + 4.0
    render_video(args.image, annotations, transcript["duration"] + 4.0, TEMP_VIDEO)
    combine_audio_video(TEMP_VIDEO, args.audio, args.output)
 
    # Clean up temp video
    if os.path.exists(TEMP_VIDEO):
        try:
            os.remove(TEMP_VIDEO)
        except Exception:
            pass

    print("  [OK] Done! Output: " + args.output)
    emit_progress('done', 'done', {'output': args.output})
 
if __name__ == "__main__":
    main()
