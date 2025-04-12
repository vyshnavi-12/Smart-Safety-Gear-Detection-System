import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import numpy as np
import os
import time
from datetime import datetime
from pathlib import Path
import shutil
from collections import deque

# Initialize session states
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = []
if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = True
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

MODEL_PATH = "runs/detect/train/weights/best.pt"

# Initialize storage directories
STORAGE_DIR = "stored_media"
PROCESSED_VIDEOS_DIR = os.path.join(STORAGE_DIR, "processed_videos")
VIOLATION_VIDEOS_DIR = os.path.join(STORAGE_DIR, "violation_videos")
UPLOADED_VIDEOS_DIR = os.path.join(STORAGE_DIR, "uploaded_videos")


# Create all required directories
for directory in [STORAGE_DIR, PROCESSED_VIDEOS_DIR, VIOLATION_VIDEOS_DIR, 
                 UPLOADED_VIDEOS_DIR]:
    os.makedirs(directory, exist_ok=True)

def cleanup_old_files(directory, max_age_days=7):
    """Clean up files older than specified days."""
    current_time = datetime.now()
    for file_path in Path(directory).glob('*'):
        if file_path.is_file():
            file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_age.days > max_age_days:
                try:
                    os.remove(file_path)
                except Exception as e:
                    st.warning(f"Failed to remove old file {file_path}: {e}")

def send_notification(message, processed_path=None, original_path=None):
    """Send a notification with both processed and original media paths."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    notification = {
        'timestamp': timestamp,
        'message': message,
    }
    
    # Save image paths if they exist
    if processed_path and os.path.exists(processed_path):
        notification['processed_media_path'] = processed_path
    if original_path and os.path.exists(original_path):
        notification['original_media_path'] = original_path
        
    st.session_state.notifications.append(notification)

class SafetySystem:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.info("Ensure the model file exists in the weights directory.")
            self.model = None
        
        self.safety_vest_colors = [
            {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])},
            {'lower': np.array([5, 100, 100]), 'upper': np.array([15, 255, 255])},
            {'lower': np.array([45, 100, 100]), 'upper': np.array([75, 255, 255])},
            {'lower': np.array([170, 100, 100]), 'upper': np.array([180, 255, 255])},
            {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])},
        ]
        self.frame_buffer = deque(maxlen=30)
        self.recording = False
        self.violation_frames = []
        self.last_violation_time = 0
        self.violation_cooldown = 5

    def store_video(self, uploaded_file):
        """Store uploaded video in the videos directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Ensure it's a video file
        if file_extension not in ['.mp4', '.mov', '.avi']:
            raise ValueError("Invalid video format. Only MP4, MOV, and AVI are supported.")
        
        stored_path = os.path.join(UPLOADED_VIDEOS_DIR, f'video_{timestamp}{file_extension}')
        
        with open(stored_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        return stored_path

    
    def process_camera_frame(self, frame, confidence_threshold):
        """Process a camera frame without violation notifications."""
        results, violations = self.process_image(frame, confidence_threshold)
        if results is not None:
            annotated_frame = results[0].plot()
            return annotated_frame, violations
        return frame, []

    # Update the process_image method:
    def process_image(self, image, confidence_threshold):
        """Process a single image and return results."""
        if not self.model:
            return None, []
        
        results = self.model(image, conf=confidence_threshold)
        violations = self.check_safety_violations(results)
        return results, violations

    def check_safety_violations(self, results):
        """Check for safety violations and return summary."""
        required_gear = {'hardhat': False, 'mask': False, 'safety_vest': False}
        violations = []
        
        for detection in results[0].boxes.data:
            class_id = int(detection[5])
            if class_id == 0:
                required_gear['hardhat'] = True
            elif class_id == 1:
                required_gear['mask'] = True
            elif class_id == 2:
                required_gear['safety_vest'] = True
        
        for gear, present in required_gear.items():
            if not present:
                violations.append(gear)
        
        return violations
    
    def store_media(self, uploaded_file):
        """Store uploaded file temporarily and return file path"""
        file_extension = Path(uploaded_file.name).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        temp_file.write(uploaded_file.read())
        temp_file.close()
        return temp_file.name

    def store_uploaded_video(self, uploaded_file):
        """Store uploaded video in a dedicated directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(uploaded_file.name).suffix
        stored_path = os.path.join(UPLOADED_VIDEOS_DIR, f'upload_{timestamp}{file_extension}')
        
        with open(stored_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        
        return stored_path

    def process_video(self, video_path, confidence_threshold):
        """Process video and store both original and processed versions."""
        if not self.model:
            return None, {}
        
        cap = cv2.VideoCapture(video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(PROCESSED_VIDEOS_DIR, f'processed_{timestamp}.mp4')
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        total_violations = set()
        frames_with_violations = 0
        total_frames = 0
        
        progress_bar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            results = self.model(frame, conf=confidence_threshold)
            violations = self.check_safety_violations(results)
            
            if violations:
                frames_with_violations += 1
                total_violations.update(violations)
            
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            total_frames += 1
            progress_bar.progress(total_frames / int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        cap.release()
        out.release()
        
        violation_summary = {
            'total_frames': total_frames,
            'frames_with_violations': frames_with_violations,
            'violation_types': list(total_violations),
            'output_path': output_path,
            'original_video_path': video_path
        }
        
        return output_path, violation_summary
    
def toggle_sidebar():
    st.session_state.show_sidebar = not st.session_state.show_sidebar

def main():
    # Clean up old files at startup
    for directory in [PROCESSED_VIDEOS_DIR, VIOLATION_VIDEOS_DIR, UPLOADED_VIDEOS_DIR]:
        cleanup_old_files(directory)

    st.title("🚨 Smart Safety Gear Detection System")
    
    # Initialize safety system
    safety_system = SafetySystem(MODEL_PATH)
    if not safety_system.model:
        return
    
    # Sidebar content
    if st.session_state.show_sidebar:
        with st.sidebar:
            st.markdown("### Input Type:")
            input_type = st.radio("", ["Camera", "Photo", "Video"])
            st.markdown("### Confidence Threshold")
            confidence_threshold = st.slider("", 0.0, 1.0, 0.5, 0.05)
            
            st.markdown("---")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown("### 📋 Notifications")
            with col2:
                if st.button("🗑️"):
                    st.session_state.notifications = []

            for notification in st.session_state.notifications:
                with st.expander(f"Notification at {notification['timestamp']}"):
                    st.write(notification['message'])
                    if 'processed_media_path' in notification:
                        if Path(notification['processed_media_path']).suffix.lower() in ['.mp4', '.mov', '.avi']:
                            st.markdown("### Processed Video")
                            st.video(notification['processed_media_path'])
                        elif Path(notification['processed_media_path']).suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            st.markdown("### Processed Image")
                            st.image(notification['processed_media_path'])
                    

    # Main content
    if input_type == "Camera":
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.session_state.camera_active:
                if st.button("Stop Camera"):
                    st.session_state.camera_active = False
            else:
                if st.button("Start Camera"):
                    st.session_state.camera_active = True
        
        if st.session_state.camera_active:
            stframe = st.empty()
            cap = cv2.VideoCapture(0)
            
            while st.session_state.camera_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                annotated_frame, violations = safety_system.process_camera_frame(frame, confidence_threshold)
                stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                time.sleep(0.1)
            
            cap.release()
            stframe.empty()
            
    elif input_type == "Video":
        uploaded_file = st.file_uploader(
            "Drag & Drop or Click to Upload",
            type=["mp4", "mov", "avi"]
        )
        
        if uploaded_file:
            try:
                # Store the original uploaded video
                original_video_path = safety_system.store_video(uploaded_file)
                st.video(original_video_path)
                
                if st.button("Analyze Video"):
                    with st.spinner("Processing video..."):
                        output_path, violation_summary = safety_system.process_video(
                            original_video_path, 
                            confidence_threshold
                        )
                        
                        if violation_summary['frames_with_violations'] > 0:
                            violation_message = (
                                f"⚠️ Safety Violations Detected in {violation_summary['frames_with_violations']} frames: "
                                f"{', '.join(violation_summary['violation_types'])}"
                            )
                            st.error(violation_message)
                            send_notification(
                                violation_message, 
                                processed_path=output_path,
                                original_path=original_video_path
                            )
                        else:
                            st.success("✅ No Safety Violations Detected")
                            send_notification(
                                "✅ No Safety Violations Detected",
                                processed_path=output_path,
                                original_path=original_video_path
                            )
                        
                        st.video(output_path)
                        
                        st.session_state.historical_data.append({
                            'timestamp': datetime.now(),
                            'type': 'video',
                            'violation_count': violation_summary['frames_with_violations'],
                            'violation_types': violation_summary['violation_types']
                        })
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
    
    else:  # Photo processing
        uploaded_file = st.file_uploader(
            "Drag & Drop or Click to Upload",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file:
            try:
                # Store the original image
                file_path = safety_system.store_media(uploaded_file)
                image = Image.open(file_path)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                with st.spinner("Analyzing safety measures..."):
                    results, violations = safety_system.process_image(image, confidence_threshold)
                    
                    if results is not None:
                        # Save the annotated image
                        annotated_image = results[0].plot()
                        processed_path = os.path.join(PROCESSED_VIDEOS_DIR, f'processed_{int(time.time())}.jpg')
                        cv2.imwrite(processed_path, annotated_image)
                        
                        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption='Analysis Result')
                        
                        if violations:
                            violation_message = f"⚠️ Safety Violations Detected: {', '.join(violations)}"
                            st.error(violation_message)
                            send_notification(violation_message, processed_path=processed_path, original_path=file_path)
                        else:
                            st.success("✅ All Safety Measures Complied")
                            send_notification("✅ All Safety Measures Complied", processed_path=processed_path, original_path=file_path)
                        
                        st.session_state.historical_data.append({
                            'timestamp': datetime.now(),
                            'type': 'image',
                            'violation_count': len(violations),
                            'violation_types': violations
                        })
                    else:
                        st.error("Failed to process image")
                        
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
