from flask import Flask, Response, render_template, jsonify
import cv2
import mediapipe as mp
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize pipeline for image classification
processor = AutoImageProcessor.from_pretrained("RavenOnur/Sign-Language")
model = AutoModelForImageClassification.from_pretrained("RavenOnur/Sign-Language")
pipe = pipeline("image-classification", model=model, feature_extractor=processor)

# List to store detected sentence
sentence = []

def classify_hand_gesture(image):
    try:
        # Convert OpenCV image (numpy array) to PIL image
        pil_image = Image.fromarray(image)
        # Use pipeline for classification
        results = pipe(pil_image)
        return results
    except Exception as e:
        print(f"Classification error: {e}")
        return []

def draw_detected_text(image, sentence):
    # Display detected sentence at the top of the frame
    h, w, _ = image.shape
    text = ''.join(sentence)  # Join the list to form a string
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

def generate_frames(camera_index=0):
    cap = cv2.VideoCapture(camera_index)  # Use the specified camera index
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert frame to RGB format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Hand detection processing
        results = hands.process(image)

        # Draw landmarks on the original frame
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Crop hand image from the frame
                h, w, c = image.shape
                xmin, ymin, xmax, ymax = w, h, 0, 0
                for lm in hand_landmarks.landmark:
                    xmin = min(xmin, int(lm.x * w))
                    ymin = min(ymin, int(lm.y * h))
                    xmax = max(xmax, int(lm.x * w))
                    ymax = max(ymax, int(lm.y * h))

                # Add margin to bounding box
                margin = 20  # Adjust margin as needed
                xmin = max(xmin - margin, 0)
                ymin = max(ymin - margin, 0)
                xmax = min(xmax + margin, w)
                ymax = min(ymax + margin, h)

                hand_img = image[ymin:ymax, xmin:xmax]

                # Draw a box around the detected hand
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

                if hand_img.size > 0:
                    # Classify the hand image
                    classification_results = classify_hand_gesture(hand_img)

                    # Display classification results below the hand
                    if classification_results:
                        label = classification_results[0]['label']

                        # Append to sentence based on the detected label
                        if label == 'space':
                            # Append a space for space gesture
                            sentence.append(' ')
                        elif label == 'clear':
                            # Clear the sentence for reset gesture
                            sentence.clear()
                        else:
                            # Add detected letter to the sentence
                            sentence.append(label)

        # Display detected sentence on the frame
        draw_detected_text(image, sentence)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_current_sentence')
def get_current_sentence():
    return jsonify(sentence=''.join(sentence))

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    sentence.clear()
    return jsonify(success=True)

if __name__ == "__main__":
    app.run(debug=True)
