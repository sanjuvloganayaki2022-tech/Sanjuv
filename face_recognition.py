"""
Face recognition module using OpenCV Haar cascades.
"""

import cv2


class FaceRecognition:
    """A class for face detection and recognition using OpenCV."""
    
    def __init__(self):
        """Initialize the FaceRecognition class with camera and cascade classifier."""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Load Haar Cascade classifier for frontal face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
    
    def start_camera(self):
        """Start camera feed with face detection until 'q' is pressed."""
        print("Press 'q' to quit the camera feed")
        
        try:
            while True:
                # Read frame from webcam
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces using detectMultiScale()
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Draw rectangles around detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Display the frame
                cv2.imshow('Face Recognition', frame)
                
                # Exit when 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nCamera feed interrupted by user")
        finally:
            # Release camera and destroy windows properly
            self.release_resources()
    
    def release_resources(self):
        """Release camera and destroy all OpenCV windows."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows destroyed")
    
    def __del__(self):
        """Destructor to ensure resources are properly released."""
        self.release_resources()
