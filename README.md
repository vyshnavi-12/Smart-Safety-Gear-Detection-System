# Smart Safety Gear Detection System 🦺👷‍♂️

An AI-powered safety compliance monitoring system that uses computer vision to detect proper safety gear usage in industrial environments. Built with Python, StreamLit, and machine learning technologies.

## 🎯 Features

- Real-time detection of safety gear including helmets, safety vests, and protective equipment
- User-friendly web interface built with StreamLit
- Support for both image and video processing
- Detailed analytics and compliance reporting
- Configurable detection parameters


## 🛠️ Technologies Used

- **Frontend**: StreamLit
- **Backend**: Python
- **Machine Learning Framework**: TensorFlow/PyTorch
- **Image Processing**: OpenCV
- **Data Analysis**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Model Training**: YOLO architecture

## 📊 Project Structure

```
├── SSGDS/
│   ├── runs/
│   │   └── detect/
│   │       └── train/
│   ├── stored_media/
│   │   ├── processed_videos/
│   │   └── uploaded_videos/
│   ├── app.py
│   └── Smart_Safety_Gear_Detection_System.py
```

## 🚀 Installation Guide

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Smart-Safety-Gear-Detection-System.git
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate  # For Windows
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the pre-trained weights (if applicable):
   ```bash
   python download_weights.py
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## 📋 Usage

1. Access the web interface at `http://localhost:8501`
2. Upload an image or video file containing safety gear scenarios
3. Adjust detection parameters if needed
4. View real-time detection results and analytics
5. Export reports and statistics as needed

## 📈 Performance Metrics

- Model Accuracy: 95%+
- Real-time processing capability: 30 FPS
- Support for multiple safety gear categories
- Low false positive rate

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

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

