# 🦺 VisionGuard-AI
### AI-Powered Personal Protective Equipment (PPE) Detection System

VisionGuard-AI is a computer vision application that detects whether workers are wearing the required Personal Protective Equipment (PPE) in real time. It uses a deep learning object detection model to identify safety equipment from images, videos, or live webcam feeds, helping improve workplace safety and compliance.

---

## 🚀 Features

- Real-time PPE detection
- Image upload support
- Video upload support
- Live webcam detection
- Detection of multiple PPE categories
- Bounding boxes with confidence scores
- User-friendly Streamlit interface
- Fast inference using YOLO-based object detection

---

## 🛠️ Tech Stack

<p align="left">

<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
<img src="https://img.shields.io/badge/YOLO-111111?style=for-the-badge&logo=yolo&logoColor=white" />
<img src="https://img.shields.io/badge/Ultralytics-00FFFF?style=for-the-badge&logoColor=black" />
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
<img src="https://img.shields.io/badge/Pillow-8A2BE2?style=for-the-badge&logoColor=white" />

</p>

---
## 🛠️ Technology Stack

| Category | Technologies Used |
|:---------|:------------------|
| **Programming Language** | Python 3.11 |
| **Framework** | Streamlit |
| **Deep Learning Model** | Ultralytics YOLOv8 |
| **Computer Vision** | OpenCV |
| **Image Processing** | Pillow (PIL) |
| **Numerical Computing** | NumPy |
| **Object Detection** | YOLOv8 |
| **Machine Learning** | PyTorch (via Ultralytics) |
| **Model Format** | `.pt` (PyTorch Weights) |
| **Dataset** | Custom PPE Detection Dataset |
| **Development Environment** | Visual Studio Code |
| **Package Manager** | pip |
| **Version Control** | Git & GitHub |
| **Deployment** | Streamlit |
| **Operating System** | Windows 11 |

---

## 📁 Project Structure

```text
VisionGuard-AI/
│
├── .vscode/
│   └── settings.json              # VS Code workspace settings
│
├── dataset/
│   ├── css-data/                  # PPE dataset
│   ├── results_yolov8n_100e/      # Model training results
│   └── source_files/              # Original dataset files
│
├── app.py                         # Main Streamlit application
├── best.pt                        # Trained YOLO model weights
├── requirements.txt               # Python dependencies
├── runtime.txt                    # Runtime configuration
└── README.md                      # Project documentation
```


> 

---

## ⚙️ Installation

### Clone the repository

```bash
git clone https://github.com/BhavyaShany/VisionGuard-AI.git
cd VisionGuard-AI
```

### Create a virtual environment (Recommended)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The application will automatically open in your default web browser.

---

## 🎯 Supported Input Sources

- Image Upload
- Video Upload
- Webcam
- RTSP CameraSetup
- Local WebCam

---

## 🧠 Model

The project uses a YOLO-based object detection model trained for PPE detection.

The model can detect safety equipment such as:

- 🪖 Hard Hat
- 🦺 Safety Vest
- 😷 Face Mask
- 👷 Person


---

## 📷 Sample Workflow

1. Launch the Streamlit application.
2. Select an input source.
3. Upload an image or video, or start the webcam.
4. The model detects PPE items in real time.
5. View detection results with confidence scores.
6. The Violations get stored in separate log.
7. You can download the Violations in a CSV file.

---

## 📦 Requirements

Install all dependencies using:

```bash
pip install -r requirements.txt
```

Common libraries include:

- streamlit
- ultralytics
- opencv-python
- numpy
- pillow

---

## 💡 Future Improvements

- Safety compliance percentage
- Worker tracking
- Violation alerts
- Helmet color classification
- Attendance integration
- Cloud deployment
- Email/SMS notifications

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a feature branch.

```bash
git checkout -b feature-name
```

3. Commit your changes.

```bash
git commit -m "Add new feature"
```

4. Push the branch.

```bash
git push origin feature-name
```

5. Open a Pull Request.

---

## 📄 License

This project is for educational purpose only.

---

## 👩‍💻 Author

**Bhavya Shany**

B.Tech CSE (AI & ML)

GitHub: https://github.com/BhavyaShany

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
