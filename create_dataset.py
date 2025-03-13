import os
import pickle
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []
total_img = 0
error_img = 0

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    total_img += 1
    if not os.path.isdir(dir_path):
        continue  # Skip non-directory files

    for img_name in os.listdir(dir_path):
        img_path = os.path.join(dir_path, img_name)
        
        try:
            # Read and Convert Image
            img = cv2.imread(img_path)
            if img is None:
                error_img +=1
                print(f"Warning: Could not read {img_path}, skipping...")
                continue  # Skip unreadable images

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Validate Hands Detection
            if not results.multi_hand_landmarks:
                error_img +=1
                print(f"Skipping {img_path}: No hands detected.")
                continue

            for hand_landmarks in results.multi_hand_landmarks:
                if len(hand_landmarks.landmark) != 21:
                    error_img +=1
                    print(f"Skipping {img_path}: Detected {len(hand_landmarks.landmark)} landmarks instead of 21.")
                    continue

                data_aux = []
                x_ = []
                y_ = []

                # Collect landmark positions
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize coordinates (relative to min x, y)
                for i in range(21):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                data.append(data_aux)
                labels.append(dir_)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            error_img +1
            continue  # Skip images that cause errors

for i, sample in enumerate(data[:5]):  # Display the first 5 samples
    print(f"Sample {i+1}: {sample}")
    
if data:
    for i in range(min(5, len(data))):  # Show up to 5 hand samples
        sample = data[i]
        
        # Extract x and y coordinates
        x_points = sample[0::2]  # X coordinates (even indices)
        y_points = sample[1::2]  # Y coordinates (odd indices)

        plt.figure(figsize=(4, 4))
        plt.scatter(x_points, y_points, c='red', label="Hand Landmark")
        
        # Draw connections like a hand
        connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                       (0, 5), (5, 6), (6, 7), (7, 8), 
                       (5, 9), (9, 10), (10, 11), (11, 12),
                       (9, 13), (13, 14), (14, 15), (15, 16),
                       (13, 17), (17, 18), (18, 19), (19, 20),
                       (0, 17)]
        
        for (a, b) in connections:
            plt.plot([x_points[a], x_points[b]], [y_points[a], y_points[b]], 'b-', linewidth=2)

        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        plt.title(f'Hand Landmark Visualization {i+1}')
        plt.legend()
        plt.show()
else:
    print("No hand landmarks detected to visualize.")
# Save Processed Data

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

procesed=total_img - error_img
print("Data preprocessing completed successfully!")
print("Total images:", total_img)
print("No of noise: " , error_img)
print("No of processed data left for training: ")