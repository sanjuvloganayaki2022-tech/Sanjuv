"""
Main entry point for the OpenCV face recognition application.
"""

from opencv_module.face_recognition import FaceRecognition


def main():
    """Main function to initialize and run the face recognition system."""
    try:
        # Create FaceRecognition instance
        face_rec = FaceRecognition()
        
        # Start camera with face detection
        face_rec.start_camera()
        
    except RuntimeError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
