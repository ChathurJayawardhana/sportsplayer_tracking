# Sports Player Tracking AI

This project focuses on **detecting players in sports videos** using **YOLOv8** models as part of the *DS5216 - Artificial Intelligence (2024/25)* coursework.  
It demonstrates how computer vision and deep learning can be applied to automate **player detection** in dynamic sports scenes across a diverse set of sports video footage.

---

## Project Overview

This work implements a **YOLO-like object detection framework** for identifying players in short sports video clips.  
As a bonus component, **keypoint detection** was also implemented to identify body joints per player.

The pre-trained **YOLOv8n** model from Ultralytics was used for player detection, with **YOLOv8n-pose** used for the keypoint detection bonus task.  
Both models detect the **'person'** class from the **COCO dataset**, which aligns well with player detection requirements.

---

## Methodology

### Dataset

Seventeen short video clips were collected from publicly available online sources and stored on Google Drive:

| Video | Resolution | FPS | Duration (s) | Total Frames |
|-------|------------|-----|--------------|--------------|
| vedio1.mp4 | 720×1280 | 29 | 12.14 | 352 |
| vedio2.mp4 | 720×1280 | 25 | 15.12 | 378 |
| vedio3.mp4 | 720×1280 | 24 | 18.25 | 438 |
| vedio4.mp4 | 720×1280 | 50 | 8.62 | 431 |
| vedio5.mp4 | 720×1280 | 30 | 15.13 | 454 |
| vedio6.mp4 | 720×1280 | 29 | 8.69 | 252 |
| vedio7.mp4 | 576×1024 | 30 | 11.83 | 355 |
| vedio8.mp4 | 720×1280 | 30 | 8.10 | 243 |
| vedio9.mp4 | 720×1280 | 25 | 9.80 | 245 |
| vedio10.mp4 | 720×1280 | 25 | 9.64 | 241 |
| vedio11.mp4 | 720×1280 | 25 | 12.52 | 313 |
| vedio12.mp4 | 720×1280 | 25 | 10.96 | 274 |
| vedio13.mp4 | 720×1280 | 25 | 15.80 | 395 |
| vedio14.mp4 | 720×1280 | 30 | 8.10 | 243 |
| vedio15.mp4 | 720×1280 | 25 | 27.08 | 677 |
| vedio16.mp4 | 720×1280 | 25 | 16.68 | 417 |
| vedio17.mp4 | 720×1280 | 25 | 11.64 | 291 |

> **Total frames processed: ~6,600 across all 17 clips**

### Model Details

- **Framework:** PyTorch  
- **Detection Model:** YOLOv8n (pre-trained on COCO dataset)  
- **Keypoint Model:** YOLOv8n-pose *(Bonus)*  
- **Detection Class:** `person`  
- **Confidence Threshold:** 0.5  
- **Device:** CUDA (GPU)  
- **Visualisation:** Matplotlib, OpenCV  

No fine-tuning was performed; the pre-trained COCO weights were used directly for inference.

---

## Results

### Player Detection — YOLOv8n

| Video | Avg Players/Frame | Max Detected | Min Detected | Avg Confidence | FPS |
|-------|:-----------------:|:------------:|:------------:|:--------------:|:---:|
| vedio1.mp4 | 1.94 | 5 | 0 | 0.697 | 81.97 |
| vedio7.mp4 | 1.78 | 6 | 0 | 0.765 | 99.24 |
| vedio5.mp4 | 0.86 | 4 | 0 | 0.521 | 83.98 |
| vedio6.mp4 | 1.27 | 5 | 0 | — | — |
| vedio4.mp4 | 1.47 | 7 | 0 | — | — |
| vedio2.mp4 | 1.74 | 5 | 0 | — | — |
| vedio16.mp4 | 4.68 | 14 | 1 | — | — |
| vedio14.mp4 | 3.60 | 11 | 0 | — | — |
| vedio15.mp4 | 4.41 | 11 | 0 | — | — |
| vedio11.mp4 | 6.39 | 11 | 3 | — | — |
| vedio10.mp4 | 5.16 | 9 | 1 | — | — |
| vedio13.mp4 | 3.32 | 8 | 1 | — | — |
| vedio12.mp4 | 1.21 | 3 | 0 | — | — |
| vedio17.mp4 | 2.66 | 5 | 0 | — | — |
| vedio9.mp4 | 2.47 | 5 | 1 | — | — |
| vedio8.mp4 | 2.69 | 8 | 0 | — | — |
| vedio3.mp4 | 0.88 | 4 | 0 | — | — |

**Summary:**
- Average confidence score (sampled): **0.661**
- Average inference speed: **~88.4 FPS** (~11.3 ms per frame)
- Peak detection: **14 players simultaneously** (vedio16.mp4)

### Keypoint Detection — YOLOv8n-pose *(Bonus)*

| Video | Avg Persons with Keypoints | Frames Processed |
|-------|:--------------------------:|:----------------:|
| vedio1.mp4 | 1.91 | 350 |
| vedio7.mp4 | 2.28 | 352 |
| vedio5.mp4 | 1.54 | 450 |
| vedio6.mp4 | 3.76 | 249 |
| vedio4.mp4 | 2.34 | 425 |
| vedio3.mp4 | 1.48 | 436 |
| vedio2.mp4 | 2.44 | 374 |
| vedio16.mp4 | 2.58 | 414 |
| vedio14.mp4 | 2.46 | 240 |
| vedio17.mp4 | 2.64 | 289 |
| vedio15.mp4 | 3.54 | 675 |
| vedio13.mp4 | 2.89 | 392 |
| vedio12.mp4 | 1.33 | 272 |
| vedio11.mp4 | 2.20 | 309 |
| vedio10.mp4 | 1.61 | 238 |
| vedio9.mp4 | 1.40 | 242 |
| vedio8.mp4 | 2.65 | 241 |

> 17 body keypoints detected per person: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles.

---

## Visual Results

Detection output screenshots are available in the `ss/` folder. Sample detections include:

- **Team sports (vedio11, vedio16):** High player counts (6–14 players/frame), bounding boxes with confidence labels
- **Lower-density clips (vedio3, vedio5):** Fewer players visible; confidence drops under partial occlusion
- **Keypoint outputs:** 17-joint skeleton overlaid on each detected person

Performance charts are available in the `plots/` folder (`performance_metrics.png`).

---

## Discussion

### Model Performance

- **YOLOv8n** achieved strong real-time performance at approximately **88 FPS** on CUDA hardware, well above real-time requirements.
- Detection confidence ranged from **0.521 to 0.765**, varying with video quality, camera angle, and player density.
- Videos with higher player counts (e.g., vedio11: avg 6.39 players/frame) showed increased occlusion events, reducing per-player confidence.

### Limitations

- No ground-truth annotations were available; direct precision/recall computation was not possible.
- Pre-trained COCO weights may misdetect referees, coaches, or bystanders as players.
- No temporal tracking — players are not assigned consistent IDs across frames.
- Motion blur and player occlusion reduce detection accuracy in fast-action sequences.
- YOLOv8n's lightweight architecture can miss small or distant players compared to larger variants.

### Future Improvements

- Fine-tune on annotated sports datasets (e.g., SoccerNet, SportsMOT) for improved precision.
- Integrate a tracking algorithm (e.g., ByteTrack, DeepSORT) for consistent player IDs across frames.
- Upgrade to **YOLOv8m or YOLOv8l** for improved accuracy on dense scenes.
- Apply sport-specific spatial filters (e.g., pitch homography) to exclude non-player detections.
- Extend keypoint tracking with temporal smoothing for action recognition.

---

## 🔗 Useful Links
- colab - https://colab.research.google.com/drive/1v19lO4noTUi4POARQ65-mXr-ZoZSV-PI?usp=sharing

---

## Repository Structure

```
sports-player-tracking/
├── ProgrammeAssignment.ipynb     # Main Colab notebook
├── README.md                     # This file
├── reports/
│   └── performance_report.txt    # Full per-video metrics
├── plots/
│   └── performance_metrics.png   # Performance visualisation chart
└── ss/
    └── detected_vedio*_frame*.jpg # Sample detection screenshots
```

---

## Author
H.M.C.P.Jayawardhana
**DS5216: Artificial Intelligence (2024/25)**  
Postgraduate Institute of Science, University of Peradeniya
