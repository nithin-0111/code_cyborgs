# Intelligent Traffic Simulation and Optimization 

This project aims to analyze and optimize traffic flow at KR Circle in Bangalore using SUMO (Simulation of Urban Mobility) and a custom traffic management system implemented with Python and OpenCV. By combining simulation and real-world video analysis, the project provides insights into traffic patterns and optimizes signal timings for smoother traffic flow.

---

## Folder Structure

### `SUMO/`
Contains the SUMO simulation files for KR Circle traffic in Bangalore. The simulation models traffic flow and congestion, providing a detailed visual representation of traffic behavior at this major intersection.

- **SUMO Features:**
  - Simulation of real-world traffic patterns at KR Circle.
  - Detailed modeling of traffic density and flow.
  - Scenario visualization and analysis.

### `open_cv/`
Contains the Python-based traffic management system using OpenCV, YOLO for object detection, and a Deep Q-Network (DQN) for optimization.

---

## Technical Details

### 1. **SUMO Simulation**
   - **Purpose:** Simulates the traffic flow at KR Circle.
   - **Key Features:**
     - Provides insights into traffic density and congestion.
     - Helps model real-world traffic conditions for KR Circle, a critical junction in Bangalore.
   - **Output:** Traffic simulation scenarios, highlighting bottlenecks and flow patterns.

### 2. **Traffic Management System**
   - **Frameworks Used:** Python, OpenCV, YOLO (Ultralytics), and PyTorch.
   - **Core Components:**
     - **YOLO-based Vehicle Detection:**
       - Uses YOLOv8 for detecting vehicles in video streams.
       - Filters objects based on detection thresholds, aspect ratios, and contour area.
     - **Deep Q-Network (DQN):**
       - Implements reinforcement learning for traffic signal optimization.
       - Adjusts signal timings dynamically using Webster’s formula based on vehicle density.
     - **Video Analysis:**
       - Processes real-world traffic videos.
       - Displays vehicle count and optimized signal timings.
   - **Key Features:**
     - Dynamic signal timing adjustment to minimize congestion.
     - Integration with Webster's formula for calculating optimal green light durations.
     - Real-time traffic density analysis and visualization.

---

## Setup and Usage

### Requirements
- Python 3.7+
- Required Python Libraries:
  - `cv2`
  - `numpy`
  - `torch`
  - `ultralytics`
- SUMO simulation environment.

### Steps to Run the Project
1. **Install SUMO:** Download and install the SUMO environment from [SUMO Download](https://sumo.dlr.de/docs/Downloads.php).
2. **Run SUMO Simulation:**
   - Navigate to the `SUMO/` folder.
   - Execute the SUMO configuration file (`*.sumocfg`) to start the simulation.
3. **Run OpenCV-based Traffic Management:**
   - Navigate to the `open_cv/` folder.
   - Execute the Python script:
     ```bash
     python traffic_system.py
     ```
   - Video files will be processed, and the system will display vehicle counts and optimized timings.

---

## Video Input Details
- The system processes traffic videos stored in the `./testingData/` directory.
- Sample video files:
  - `video_01.mp4`
  - `bangalore.mp4`
  - `amb1.mp4`
  - `v2.mp4`
- Results include:
  - Maximum vehicles detected per video.
  - Average optimized signal timings.

---

## Outputs
- **SUMO Simulation Outputs:** Visualized traffic scenarios.
- **Traffic Management System Outputs:**
  - Vehicle counts in real-time.
  - Optimized signal timings displayed on the video.

---

## Contributions
Contributions to improve the traffic simulation or the reinforcement learning algorithm are welcome. Feel free to create a pull request or report issues.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments
- **SUMO:** For providing a robust platform for traffic simulation.
- **Ultralytics YOLO:** For advanced object detection capabilities.
- **PyTorch:** For enabling the implementation of reinforcement learning.

---
