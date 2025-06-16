# Smart Safety Gear Detection System 🦺👷‍♂️

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.27.0-orange.svg)](https://streamlit.io/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics%20YOLO-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)


This AI-powered system monitors safety compliance in industrial settings by detecting the presence (or absence) of crucial safety gear using computer vision.  Built with Python, Streamlit, and the powerful Ultralytics YOLO model, it offers a user-friendly interface for real-time analysis and comprehensive reporting.


## 🚀 Features

- **Real-time Safety Gear Detection:** Identifies helmets, safety vests, and other protective equipment in images and videos.
- **Streamlit Web Interface:** Provides an intuitive and user-friendly web application for easy interaction.
- **Image & Video Processing:** Supports both image uploads and video streaming for versatile analysis.
- **Detailed Analytics & Reporting:** Generates comprehensive compliance reports, including confusion matrices, precision-recall curves, and F1-score curves (see `runs/detect/train/` directory).
- **Historical Data Storage:** Stores processed data for trend analysis and improved performance over time (Implementation inferred from `st.session_state`).
- **Notification System:**  Provides alerts and notifications within the Streamlit application (Implementation inferred from `st.session_state`).


## ⚙️ Installation

This project requires Python 3.8+ and several libraries.  Install them using pip:

```bash
pip install -r requirements.txt
```

**Note:** You'll need to create a `requirements.txt` file listing all project dependencies.  A good starting point would include:

```
streamlit
ultralytics
opencv-python
Pillow
```

You may also need to install additional dependencies based on the full `app.py` code and the specific YOLO model used.


## 🏃 Usage

1.  **Run the Streamlit app:** `streamlit run app.py`
2.  Access the web interface at `http://localhost:8501`
3.  **Upload an image or video:** Use the interface to upload your media file.
4.  **View the results:** The system will process the media and display the detection results, along with relevant metrics and visualizations.
5.  **Review historical data:** Access previous analyses and reports from the application's interface (Functionality inferred from session state).


## 📁 Project Structure

```
Smart-Safety-Gear-Detection-System/
├── app.py                # Main Streamlit application
├── SSGDS/                # Backend folder (presumably for model loading/management)
│   └── start.py          # Likely contains model initialization
├── runs/                 # Results from model training and inference
│   └── detect/train/    # Contains images, curves and metrics from the model training.
├── stored_media/         # Processed media files (Images and videos)
├── READ.md               # Additional Documentation
└── README.md             # This file
```


## 🛠️ Technology Stack

- **Python:** Programming language
- **Streamlit:** Web framework for building interactive web apps
- **Ultralytics YOLO:** State-of-the-art object detection model
- **OpenCV:** Computer vision library
- **Pillow:** Image processing library

## ⚠️ Project Status and Rights

This project was developed during an internship at Infosys Limited. All rights and intellectual property associated with this project belong to Infosys Limited. This repository serves as a demonstration of the work completed during the internship period. Any use, modification, or distribution of this code should be in accordance with Infosys Limited's policies and guidelines.

## 🙏 Acknowledgments

- Infosys Limited for providing the internship opportunity and resources
- Project mentors and team members at Infosys
- Dataset contributors and supporters

## 📧 Contact

- GitHub: [@vyshnavi-12](https://github.com/vyshnavi-12)
- LinkedIn: [@sri-vyshnavi-nakka](https://www.linkedin.com/in/sri-vyshnavi-nakka-38136428b/)
- Email: [srivyshnavinakka@gmail.com](mailto:srivyshnavinakka@gmail.com)

## Support 🙋‍♂️

If you have any questions or need help, please open an issue in the repository.

---
Made by Sri Vyshnavi

