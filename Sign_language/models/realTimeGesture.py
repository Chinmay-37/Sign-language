import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("models/gesture_model.keras")

# Define class labels (ensure correct order)
class_labels = {0: "Hello", 1: "No", 2: "Thank You", 3: "Yes"}

def detect_gesture(model):
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame (optional for correct orientation)
        frame = cv2.flip(frame, 1)

        # Convert frame to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize and preprocess the frame
        img = cv2.resize(frame_rgb, (224, 224))  # Resize to model input size
        img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict gesture
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels.get(predicted_class, "Unknown")  # Use dictionary mapping

        # Display the prediction on the video frame
        cv2.putText(frame, f"Prediction: {predicted_label}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Gesture Recognition", frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return predicted_label  # Return the recognized gesture for backend response
