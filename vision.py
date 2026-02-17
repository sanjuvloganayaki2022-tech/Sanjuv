"""
Vision module for camera operations using OpenCV.
"""

import cv2


class Vision:
    """A class for handling computer vision operations with OpenCV."""
    
    def __init__(self):
        """Initialize the Vision class with camera capture and face detection."""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load face cascade classifier")
    
    def show_camera(self):
        """Display webcam feed with face detection until user presses 'q' to quit."""
        print("Press 'q' to quit the camera feed")
        
        try:
            while True:
                # Capture frame-by-frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Detect faces in the frame
                frame_with_faces = self.detect_faces(frame)
                
                # Display the resulting frame
                cv2.imshow('Camera Feed - Face Detection', frame_with_faces)
                
                # Wait for 'q' key to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nCamera feed interrupted by user")
        finally:
            self.release_camera()
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame and draw rectangles around them.
        
        Args:
            frame: Input image frame
            
        Returns:
            Frame with rectangles drawn around detected faces
        """
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Add label "Face" above each detected face
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display face count on the frame
        face_count = len(faces)
        cv2.putText(frame, f'Faces: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def release_camera(self):
        """Release camera and destroy all OpenCV windows."""
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows destroyed")
    
    def __del__(self):
        """Destructor to ensure camera is properly released."""
        self.release_camera()
