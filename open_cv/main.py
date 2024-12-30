import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

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

        self.state_size = 2
        self.action_size = 5
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.batch_size = 32

        self.policy_net = DQNetwork(self.state_size, self.action_size)
        self.target_net = DQNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def calculate_websters_formula(self, count, density):
        lost_time = 4
        cycle_length = (1.5 * lost_time + 5) / (1 - density) if density < 1 else 90
        green_time = (cycle_length * count) / (count + density) if (count + density) > 0 else 10
        return max(10, min(int(green_time), 90))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q_values = self.policy_net(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                with torch.no_grad():
                    target += self.gamma * torch.max(self.target_net(torch.FloatTensor(next_state))).item()
            current_q = self.policy_net(torch.FloatTensor(state))[action]
            loss = nn.MSELoss()(current_q, torch.tensor(target))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def run(self):
        video_files = ['./testingData/video_01.mp4', './testingData/bangalore.mp4', './testingData/amb1.mp4', './testingData/v2.mp4']
        video_results = []

        for video_idx, video in enumerate(video_files, start=1):
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

                for x, y, w, h in valid_contours:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(frame, f"Vehicles Detected: {current_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Optimized Time: {optimized_time}s", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Traffic Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()

            avg_count = total_count / frame_count if frame_count else 0
            avg_optimized_time = sum(optimized_times) / len(optimized_times) if optimized_times else 0

            video_results.append((f"video_{video_idx}", max_count, avg_optimized_time))

        video_results.sort(key=lambda x: x[1], reverse=True)
        for video, count, time in video_results:
            print(f"Vehicle - {video}, Max Vehicles Detected: {count}, Optimized time: {time:.2f}s")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    traffic_system = TrafficSystem()
    traffic_system.run()
