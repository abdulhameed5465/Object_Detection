# Object Detection System using SSD MobileNet V3

A real-time Object Detection project built with Python and OpenCV. This application leverages a pre-trained **SSD MobileNet V3** deep learning architecture combined with the **COCO dataset** configuration to detect and classify a wide variety of common entities (such as people, vehicles, and objects) instantly through a camera or video stream.

---

## 🚀 Key Features
* **Real-Time Classification:** Detects moving and static targets on live camera inputs.
* **Pre-Trained Accuracy:** Uses the specialized Single Shot MultiBox Detector (SSD) optimized for lightweight edge execution.
* **Visual Bounding Overlays:** Draws automated localized indicators alongside high-confidence classification text tags.

---

## 📂 Repository Breakdown
The repository contains the following core files and structure:
* `main.py` — The primary Python initialization script parsing configurations and handling the active window stream loop.
* `frozen_inference_graph.pb` — The frozen pre-trained TensorFlow graph weights containing the core network parameters.
* `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt.txt` — The structured text configuration file outlining model topology details.
* `labels.txt` — An explicit line-by-line index listing target classification names derived from the COCO dataset mapping layout.
* `SampleVideo1.mp4` — A template media clip used for debugging tracking components without relying entirely on physical capture devices.
* `LICENSE` — Explicitly defines distribution rights using an open-source MIT template framework.

---

## ⚙️ Environment Setup & Installation

### 1. Initialize Your Environment
Open your local terminal inside the project directory and create a clean isolated workspace mapping back to your system's global core interpreter:
```bash
python -m venv .venv
```

Activate the environment block to isolate structural configurations:
* **Windows:**
  ```cmd
  .venv\Scripts\activate
  ```
* **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 2. Install Required Dependencies
Bring in standard visual processing wrappers by verifying tracking components:
```bash
pip install opencv-python numpy
```

---

## 🏁 Execution Guide
Ensure the configuration layout files (`.pb`, `.pbtxt`, and `labels.txt`) are resting side-by-side in your parent directories, then activate execution by prompting:

```bash
python main.py
```

### Stream Shortcuts
* Press **`q`** inside the actively playing visual capture pipeline matrix pane anytime to flush buffers, kill active handles, and terminate program execution safely.

---

## 📝 License
This project is open-sourced under the terms of the [MIT License](LICENSE).
