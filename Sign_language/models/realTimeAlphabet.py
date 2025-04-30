import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import time

class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                'del', 'nothing', 'space']

class AlphabetRecognizer:
    def __init__(self, model):
        self.model = model
        self.prediction_history = deque(maxlen=20)  # Slightly smaller buffer for 3 sec detection
        self.word_history = []
        self.current_letter = None
        self.letter_start_time = None
        self.is_running = True
        self.detection_threshold = 3  # Changed to 3 seconds

    def detect_alphabet(self):
        cap = cv2.VideoCapture(0)
        x1, y1, x2, y2 = 200, 100, 450, 350

        

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = frame_rgb[y1:y2, x1:x2]

            hand_img = cv2.resize(roi, (64, 64))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            predictions = self.model.predict(hand_img, verbose=0)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)
            predicted_label = class_labels[predicted_class]

            if confidence > 0.7:
                self.prediction_history.append(predicted_label)
                most_common = Counter(self.prediction_history).most_common(1)[0][0]
                
                if most_common != self.current_letter:
                    self.current_letter = most_common
                    self.letter_start_time = time.time()
                
                # Changed to 3-second threshold
                if (time.time() - self.letter_start_time) > self.detection_threshold:
                    if self.current_letter not in ['nothing', 'del']:
                        if self.current_letter == 'space':
                            self.word_history.append(' ')
                        else:
                            self.word_history.append(self.current_letter)
                    self.letter_start_time = time.time()
                    self.prediction_history.clear()

            # Display with countdown
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            if self.current_letter and self.letter_start_time:
                elapsed = time.time() - self.letter_start_time
                remaining = max(0, self.detection_threshold - elapsed)
                status_text = f"Hold {self.current_letter}: {remaining:.1f}s"
                color = (0, 255, 0) if remaining < 1 else (0, 165, 255)  # Orange -> Green
            else:
                status_text = "Show your sign"
                color = (0, 0, 255)

            cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, color, 2, cv2.LINE_AA)
            
            cv2.putText(frame, f"Word: {''.join(self.word_history)}", (50, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("ASL Alphabet Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.is_running = False
                break

        cap.release()
        cv2.destroyAllWindows()
        return ''.join(self.word_history)

def detect_alphabet(model):
    recognizer = AlphabetRecognizer(model)
    return recognizer.detect_alphabet()