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

- Python
- Streamlit
- OpenCV
- Ultralytics YOLO
- NumPy
- Pillow

---

## 📁 Project Structure

```
VisionGuard-AI/
│
├── app.py                 # Main Streamlit application
├── models/                # Trained YOLO model
├── utils/                 # Helper functions
├── uploads/               # Uploaded media
├── outputs/               # Detection results
├── requirements.txt
├── README.md
└── LICENSE
```

> *Project structure may vary depending on your implementation.*

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
- Live Camera Feed (if supported)

---

## 🧠 Model

The project uses a YOLO-based object detection model trained for PPE detection.

The model can detect safety equipment such as:

- 🪖 Hard Hat
- 🦺 Safety Vest
- 😷 Face Mask
- 👷 Person

*(Classes depend on the model used.)*

---

## 📷 Sample Workflow

1. Launch the Streamlit application.
2. Select an input source.
3. Upload an image or video, or start the webcam.
4. The model detects PPE items in real time.
5. View detection results with confidence scores.

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

This project is licensed under the MIT License.

---

## 👩‍💻 Author

**Bhavya Shany**

B.Tech CSE (AI & ML)

GitHub: https://github.com/BhavyaShany

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
