import os
import cv2
import time
import random
import pickle
import numpy as np
import mediapipe as mp
import tkinter as tk
from tkinter import Button
import threading
import sys

# Load the pre-trained model (No scaler)
try:
    model_dict = pickle.load(open('./model_KNN.p', 'rb'))
    model = model_dict['model']  # Only load the model (No scaler)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

# Load the reference image
image_path = './assets/hand_signs.jpg'
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Image at {image_path} not found!")
    sys.exit()
print("Image loaded successfully!")

# Set window dimensions
window_width = 900  
window_height = 600  
status_bar_height = 60
max_image_height = 200  

# Resize the reference image while maintaining aspect ratio
scale_factor = max_image_height / image.shape[0]
new_width = int(image.shape[1] * scale_factor)
new_height = int(image.shape[0] * scale_factor)
image_resized = cv2.resize(image, (new_width, new_height))

# Game variables
points = 5  # Defined globally
labels_dict = {i: str(i) for i in range(10)}  
game_running = False
cap = None  # Initialize camera variable

# Initialize Tkinter
root = tk.Tk()
root.title("Hand Gesture Game")
root.geometry("200x80")
root.geometry(f"200x80+{root.winfo_screenwidth()//2-100}+{root.winfo_screenheight()//2-40}")

def start_game():
    """Runs the OpenCV game loop in a separate thread."""
    game_thread = threading.Thread(target=run_game, daemon=True)
    game_thread.start()

def run_game():
    """Runs the OpenCV game logic in a separate thread."""
    global points, game_running, cap  
    try:
        root.withdraw()  # Hide the Tkinter window safely
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Camera could not be opened.")
            return  

        print("Camera opened successfully!")

        # Create single OpenCV window
        cv2.namedWindow("Hand Gesture Game", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand Gesture Game", window_width, window_height)

        previous_number = None  
        game_running = True

        while game_running:
            number_to_match = random.choice(list(labels_dict.values()))
            while number_to_match == previous_number:
                number_to_match = random.choice(list(labels_dict.values()))
            previous_number = number_to_match  

            update_message = f'Match: {number_to_match}'

            start_time = time.time()
            match_found = False
            point_awarded = False

            while time.time() - start_time < 5:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Warning: Failed to read frame. Retrying...")
                    continue  

                frame = cv2.resize(frame, (500, 400))  
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing_styles.get_default_hand_landmarks_style(),
                                                  mp_drawing_styles.get_default_hand_connections_style())

                        data_aux = []
                        x_ = []
                        y_ = []

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            x_.append(x)
                            y_.append(y)

                        for i in range(len(hand_landmarks.landmark)):
                            x = hand_landmarks.landmark[i].x
                            y = hand_landmarks.landmark[i].y
                            data_aux.append(x - min(x_))
                            data_aux.append(y - min(y_))

                        try:
                            prediction = model.predict([np.asarray(data_aux)])  
                            predicted_label = str(prediction[0])

                            # Draw bounding box
                            x1 = int(min(x_) * frame.shape[1]) - 10
                            y1 = int(min(y_) * frame.shape[0]) - 10
                            x2 = int(max(x_) * frame.shape[1]) + 10
                            y2 = int(max(y_) * frame.shape[0]) + 10
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(frame, predicted_label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

                        except Exception as e:
                            print(f"Prediction error: {e}")
                            continue

                        if predicted_label == number_to_match and not point_awarded:
                            match_found = True
                            points += 1
                            point_awarded = True
                            break

                # Create single combined window
                combined_window = np.zeros((window_height, window_width, 3), dtype=np.uint8)
                combined_window[:400, :500] = frame  # Add webcam feed
                combined_window[20:new_height+20, window_width - new_width - 20:window_width - 20] = image_resized  # Reference image
                
                # Add status bar
                status_frame = np.zeros((status_bar_height, window_width, 3), dtype=np.uint8)
                cv2.putText(status_frame, update_message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(status_frame, f'Points: {points}', (window_width - 200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                combined_window[-status_bar_height:, :] = status_frame

                # Display the combined window
                cv2.imshow("Hand Gesture Game", combined_window)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    close_program()
                    return

            if not match_found:
                points -= 1

            if points >= 10 or points <= 0:
                game_running = False

        # 🛑 **Freeze the last frame and display Win/Lose message**
        ret, last_frame = cap.read()
        if ret and last_frame is not None:
            last_frame = cv2.resize(last_frame, (500, 400))
            cv2.putText(last_frame, "You Win!" if points >= 10 else "You Lose!", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(last_frame, "Press Q to Quit", (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        while True:
            cv2.imshow("Hand Gesture Game", last_frame)  

            if cv2.waitKey(1) & 0xFF == ord('q'):
                close_program()
                return

    except Exception as e:
        print(f"Error in run_game(): {e}")

def close_program():
    """Closes all windows, releases resources, and exits the program."""
    global cap
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    root.quit()
    sys.exit()

# Create and place the Start Button
start_button = Button(root, text="Start Game", font=("Arial", 12), command=start_game)
start_button.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()
