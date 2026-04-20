# 🥚 EGG-O: Robotics Data Pipeline

> *"From human hands to robot minds — one frame, one action, one dataset at a time."*

**An end-to-end AI-powered egocentric video annotation pipeline for robot learning.**  
Built by **The Rizz-Bizz Company** as a capstone project targeting India's critical gap in robotics training data infrastructure.

---

## 🧠 What Is EGG-O?

Modern robots that perform physical tasks — folding garments, assembling components, sorting packages — must be trained on **egocentric (first-person perspective) video data**. Third-person camera footage misses the fine-grained hand-object interactions, grasp geometry, and spatial relationships that actually matter for robot policy learning.

EGG-O is a **fully automated, zero-manual-labelling pipeline** that takes a raw egocentric video as input and produces a structured, robot-ready JSON annotation dataset as output. It:

1. Ingests any first-person video (MP4 / MOV / AVI, up to 4 GB)
2. Preprocesses and filters frames (blur, brightness, dead-time removal)
3. Detects action boundaries via scene segmentation
4. Profiles motion intensity per scene using optical flow
5. Detects objects + hand/wrist keypoints per frame
6. Generates natural-language captions for each action segment
7. Outputs a schema-validated JSON dataset ready for robot learning frameworks

---

## 📁 Project Structure

```
egg-o/
├── pipeline/
│   ├── ingest.py          # Stage 1 — Video ingestion & metadata extraction
│   ├── preprocess.py      # Stage 2 — Frame extraction, blur/brightness filter, dead-time removal
│   ├── segment.py         # Stage 3 — Scene segmentation via PySceneDetect
│   ├── motion.py          # Stage 4 — Optical flow motion profiling per scene
│   ├── annotate.py        # Stage 5 — YOLOv8 object + pose detection, IoU contact detection
│   └── caption.py         # Stage 6 — NLP caption generation (GPT-4o / LLaMA)
│
├── models/                # (auto-downloaded on first run — not committed to git)
│   ├── yolov8n.pt
│   └── yolov8n-pose.pt
│
├── output/                # Pipeline outputs — per-video subfolders
│   └── <video_name>/
│       ├── frames/        # Extracted usable frames
│       ├── scenes/        # Scene clips with timestamps
│       ├── annotations/   # Per-frame YOLO + pose JSON
│       └── dataset.json   # Final robot-ready annotation dataset
│
├── tests/                 # Unit tests per pipeline stage
├── .env.example           # Template for API key configuration
├── requirements.txt       # All Python dependencies
├── run_pipeline.py        # Single entrypoint — runs all stages end-to-end
└── README.md
```

---

## ⚙️ Tech Stack

All tools are **free and open-source** — zero infrastructure cost, fully reproducible.

| Component | Tool | Purpose |
|---|---|---|
| Video I/O | `OpenCV` + `FFmpeg` | Frame extraction, codec handling, bitrate/metadata |
| Scene Segmentation | `PySceneDetect` | Diff-based action boundary detection |
| Object Detection | `YOLOv8n` (Ultralytics) | 80 COCO classes, bounding box annotation |
| Pose / Hand Detection | `YOLOv8n-pose` + `MediaPipe` | 17 COCO keypoints; wrist trajectory (indices 9 & 10) |
| Motion Profiling | `OpenCV` optical flow | Mean/peak motion, entropy, percent-active per scene |
| Caption Generation | `GPT-4o API` / `LLaMA` | Natural-language action descriptions per segment |
| Data Validation | `jsonschema` + `Pydantic` | Schema-based output validation |
| Blur Detection | `Laplacian variance` (OpenCV) | Calibrated threshold = 15 for egocentric video |
| Contact Detection | Bounding-box `IoU` | Threshold = 0.05 for hand-object contact |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/rizz-bizz/egg-o.git
cd egg-o
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `ffmpeg` must also be installed at the system level.  
> macOS: `brew install ffmpeg` | Ubuntu: `sudo apt install ffmpeg` | Windows: [ffmpeg.org](https://ffmpeg.org/download.html)

### 4. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-...
```

To use LLaMA instead of GPT-4o (offline, free), set `CAPTION_BACKEND=llama` in `.env`.

### 5. Run the pipeline

```bash
python run_pipeline.py --input your_video.mp4 --output output/
```

For a specific stage only:
```bash
python run_pipeline.py --input your_video.mp4 --stage preprocess
python run_pipeline.py --input your_video.mp4 --stage annotate
```

---

## 🔬 Pipeline Stages in Detail

### Stage 1 — Ingestion
Extracts video metadata: codec, resolution, framerate, duration, bitrate, color space. Accepts MP4 / MOV / AVI up to 4 GB.

### Stage 2 — Preprocessing
Four sub-phases:
- **Frame extraction** at target FPS
- **Blur detection** via Laplacian variance scoring (threshold = 15, calibrated for egocentric video)
- **Brightness/contrast normalization** for variable factory lighting
- **Dead-time removal** (frames with no meaningful motion)

*Validated result on our test video: 6,639 raw frames → 5,847 usable frames (11.9% reduction)*

### Stage 3 — Scene Segmentation
Uses `PySceneDetect` with diff-based detection (threshold = 5.0, recalibrated from the default 27.0 for egocentric footage). Each detected scene is tagged with start/end timestamp, duration, and action category.

*Test result: 17 cuts → 18 valid scenes ranging from 2.6 s to 35.2 s*

### Stage 4 — Motion Profiling
Per-scene features via frame-difference optical flow:
- **Mean motion** — average pixel displacement
- **Peak motion** — maximum single-frame displacement
- **Entropy** — motion randomness (high = varied, low = repetitive/rhythmic)
- **Percent active** — fraction of frames above motion threshold

### Stage 5 — Object & Hand Annotation
- `YOLOv8n` for 80-class object detection
- `YOLOv8n-pose` for 17 COCO keypoints (wrist indices 9 & 10)
- Hand-object contact via bounding-box IoU (threshold = 0.05)
- Wrist trajectory tracking: (x, y, speed) per frame per hand
- Blur-filtered: frames below Laplacian variance threshold are skipped

### Stage 6 — Caption Generation *(Planned / Beta)*
For the top-N scenes, generates natural language descriptions combining:
- Action label from motion profiling
- Hand detection results (dominant hand, contact %)
- Wrist speed (movement intensity)
- Object detections

---

## 📊 Output Format

Each processed video produces a `dataset.json` in this structure:

```json
{
  "video_id": "shirt_fold_01",
  "metadata": {
    "resolution": "1920x1080",
    "fps": 30,
    "duration_s": 221,
    "usable_frames": 5847
  },
  "scenes": [
    {
      "scene_id": 1,
      "start_time": 0.0,
      "end_time": 12.3,
      "duration_s": 12.3,
      "motion": {
        "mean": 4.21,
        "peak": 18.7,
        "entropy": 0.63,
        "percent_active": 0.78
      },
      "annotations": [
        {
          "frame": 142,
          "objects": [{"label": "shirt", "bbox": [x, y, w, h], "conf": 0.91}],
          "hands": {
            "left": {"wrist_xy": [312, 480], "speed": 3.4, "contact": true},
            "right": {"wrist_xy": [510, 460], "speed": 2.1, "contact": false}
          }
        }
      ],
      "caption": "Person picks up a folded shirt with left hand and places it into a bag."
    }
  ]
}
```

---

## 👥 Team

| Role | Name | Responsibilities |
|---|---|---|
| Pipeline & Data Processing Lead | Mayukh | Video preprocessing, scene segmentation, motion profiling |
| NLP & AI/ML Expert | Madhav (Madhav Arora) | Caption generation, JSON structuring, schema validation |

---

## 📈 Success Metrics

| Metric | Target |
|---|---|
| Annotated action segments generated automatically | 500+ |
| End-to-end pipeline runtime per video clip | < 60 seconds |
| Reduction in manual annotation effort | 10x |

---

## 🌐 Use Cases

- **Factory robotics** (garment, pharma, food processing) — hand-action annotation for pick-and-place training
- **Surgical robotics** — surgeon hand movement annotation for robot-assisted surgery ($14.4B market by 2030)
- **Warehouse logistics** — novel object configuration training for Amazon/Flipkart-style picking
- **Agricultural robotics** — crop harvesting hand motions (India: 138M farms, $1.98B robotics market 2025)
- **Physiotherapy & rehabilitation** — patient movement annotation for AI-driven assessment

---

## ⚠️ Known Limitations

| Limitation | Notes |
|---|---|
| **Privacy at scale** | Every clip may capture bystanders or proprietary processes; GDPR / DPDP Act compliance requires consent, anonymization, and secure storage |
| **Sensor gaps** | Smartphone video has no depth, force data, or eye tracking — signals that research rigs capture |
| **Automation quality** | SAM-2 fails on occlusions and harsh lighting; at 1,000+ videos/day a human QA loop is non-negotiable |
| **LLM captioning cost** | GPT-4o at $0.02/clip = ~$73K/year at scale; a fine-tuned open model (LLaMA) is needed for economic viability |

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in:

```bash
OPENAI_API_KEY=sk-...          # For GPT-4o caption generation
CAPTION_BACKEND=openai         # or "llama" for local inference
OUTPUT_DIR=./output
LOG_LEVEL=INFO
TARGET_FPS=5                   # Frames per second to extract
BLUR_THRESHOLD=15              # Laplacian variance floor
SCENE_DIFF_THRESHOLD=5.0       # PySceneDetect diff sensitivity
IOL_CONTACT_THRESHOLD=0.05     # Hand-object contact IoU threshold
```

---

## 📚 References & Datasets

| Resource | Why It Matters |
|---|---|
| [Ego4D (Meta AI)](https://ego4d-data.org/) | 3,025 h of egocentric video; standard benchmark for first-person perception |
| [EPIC-Kitchens (U. Bristol)](https://epic-kitchens.github.io/) | 45 kitchens, 39K action segments; canonical egocentric action recognition benchmark |
| [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/) | State-of-the-art real-time object detection & pose estimation |
| [MediaPipe (Google)](https://mediapipe.dev/) | On-device hand landmark detection |
| [PySceneDetect](https://www.scenedetect.com/) | Video scene detection & content analysis library |
| [Snorkel AI](https://snorkel.ai/) | Programmatic weak supervision — inspiration for scaling annotation |

---

## 📄 License

MIT License — see `LICENSE` for details.

---

*EGG-O is a capstone project by The Rizz-Bizz Company. Domain: Physical AI · Robotics Training Data · Egocentric Computer Vision.*
