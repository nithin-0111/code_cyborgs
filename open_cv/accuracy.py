import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TrafficSystem:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.CONFIG = {
            'DETECTION_THRESHOLD': 0.5,
            'MIN_WIDTH_RECT': 20,
            'MIN_CONTOUR_AREA': 400,
            'MIN_ASPECT_RATIO': 0.5,
            'MAX_ASPECT_RATIO': 4.0
        }

    def calculate_websters_formula(self, count, density):
        lost_time = 4
        cycle_length = (1.5 * lost_time + 5) / (1 - density) if density < 1 else 90
        green_time = (cycle_length * count) / (count + density) if (count + density) > 0 else 10
        return max(10, min(int(green_time), 90))

    def run_single_video(self, video):
        cap = cv2.VideoCapture(video)
        total_count = 0
        frame_count = 0
        max_count = 0
        optimized_times = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame)
            valid_contours = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()

                for box, confidence in zip(boxes, confidences):
                    if confidence >= self.CONFIG['DETECTION_THRESHOLD']:
                        x_min, y_min, x_max, y_max = map(int, box)
                        w, h = x_max - x_min, y_max - y_min
                        aspect_ratio = float(w) / h
                        area = w * h

                        if (self.CONFIG['MIN_WIDTH_RECT'] <= w <= self.CONFIG['MAX_ASPECT_RATIO'] * h and
                                self.CONFIG['MIN_ASPECT_RATIO'] <= aspect_ratio <= self.CONFIG['MAX_ASPECT_RATIO'] and
                                area >= self.CONFIG['MIN_CONTOUR_AREA']):
                            valid_contours.append((x_min, y_min, w, h))

            current_count = len(valid_contours)
            total_count += current_count
            frame_count += 1

            max_count = max(max_count, current_count)

            density = current_count / frame_count if frame_count else 0
            optimized_time = self.calculate_websters_formula(current_count, density)
            optimized_times.append(optimized_time)

        cap.release()
        avg_count = total_count / frame_count if frame_count else 0
        avg_optimized_time = sum(optimized_times) / len(optimized_times) if optimized_times else 0

        return max_count, avg_optimized_time

    def run_and_evaluate(self):
        video_files = ['./testingData/video_01.mp4', './testingData/bangalore.mp4', './testingData/amb1.mp4', './testingData/v2.mp4']
        video_results = {}

        for video_idx, video in enumerate(video_files, start=1):
            print(f"Running evaluation for {video} (First Run)")
            max_count_1, avg_time_1 = self.run_single_video(video)

            print(f"Running evaluation for {video} (Second Run)")
            max_count_2, avg_time_2 = self.run_single_video(video)

            accuracy = 100 - abs(max_count_1 - max_count_2) / max(max_count_1, max_count_2) * 100
            video_results[f"video_{video_idx}"] = accuracy

        self.plot_accuracy(video_results)

    def plot_accuracy(self, results):
        videos = list(results.keys())
        accuracies = list(results.values())

        plt.figure(figsize=(10, 6))
        plt.bar(videos, accuracies, color='skyblue')
        plt.xlabel('Videos')
        plt.ylabel('Accuracy (%)')
        plt.title('Accuracy of Vehicle Detection across Videos')
        plt.ylim(0, 100)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 2, f"{v:.2f}%", ha='center', fontsize=10)
        plt.show()

if __name__ == "__main__":
    traffic_system = TrafficSystem()
    traffic_system.run_and_evaluate()
