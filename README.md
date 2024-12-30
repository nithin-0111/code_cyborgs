# Intelligent Traffic Signal Optimization using Deep Q Learning 

This project tackles the ever-growing traffic woes at KR Circle, one of Bangalore’s busiest junctions. We’ve combined the power of SUMO simulations with AI-driven traffic management to analyze and optimize traffic flow like never before.  

---

## What’s Inside  

### `SUMO/`  
This directory contains a SUMO simulation for KR Circle traffic. It’s a digital twin of the real-world traffic at this bustling intersection, helping us visualize and analyze traffic patterns.  

- **What it Does:**  
  - Simulates real-world traffic flow.  
  - Highlights congestion hotspots.  
  - Helps understand traffic density dynamics.  

### `open_cv/`  
This directory is where the real action happens. It contains the Python code for analyzing traffic videos, detecting vehicles using YOLO, and optimizing signal timings with a Deep Q-Network (DQN).  

---

## How It Works  

### 1. **SUMO Simulation**  
- Think of it as a virtual traffic experiment.  
- It models traffic at KR Circle, so we can better understand how vehicles flow through and where they get stuck.  
- The output? A clear picture of how traffic moves (or doesn’t) in this area.  

### 2. **AI-Powered Traffic Management**  
This is the exciting part! The system:  
- **Detects Vehicles:** Uses YOLOv8 to identify vehicles in real-time from video footage.  
- **Optimizes Traffic Lights:** Applies reinforcement learning (using a Deep Q-Network) to dynamically adjust signal timings based on traffic density.  
- **Crunches Numbers with Webster’s Formula:** Ensures green lights are perfectly timed to keep vehicles moving efficiently.  

---

## Getting Started  

### What You Need  
- Python 3.7+  
- These Python libraries:  
  - OpenCV (`cv2`)  
  - NumPy (`numpy`)  
  - PyTorch (`torch`)  
  - Ultralytics YOLO (`ultralytics`)  
- SUMO traffic simulation software ([Download it here](https://sumo.dlr.de/docs/Downloads.php)).  

### Steps to Run  
1. **SUMO Simulation:**  
   - Head to the `SUMO/` directory and run the configuration file (`*.sumocfg`).  
   - This starts the simulation and gives you a visual overview of KR Circle traffic.  

2. **AI Traffic Management:**  
   - Move to the `open_cv/` directory.  
   - Run the Python script:  
     ```bash  
     python traffic_system.py  
     ```  
   - Sit back and watch as the system analyzes videos and optimizes traffic signal timings.  

---

## What It Outputs  

- **From SUMO:**  
  - A detailed simulation of traffic flow at KR Circle.  

- **From AI Traffic Management:**  
  - Real-time vehicle detection.  
  - Optimized green light timings displayed on the video.  

Sample video inputs (stored in `./testingData/`):  
- `video_01.mp4`  
- `bangalore.mp4`  
- `amb1.mp4`  
- `v2.mp4`  

For each video, you’ll see:  
- The maximum number of vehicles detected.  
- The average green light time calculated for optimal traffic flow.  

---

## Why It Matters  

Traffic congestion is a daily challenge in cities like Bangalore. With tools like SUMO and AI, we can not only study traffic but also propose smarter solutions to make daily commutes faster and less stressful.  

---

## Want to Help?  

We’d love your contributions! Whether it’s improving the SUMO model, tweaking the AI code, or just suggesting new ideas, feel free to open a pull request or drop us a message.  

---

## License  
This project is licensed under the MIT License. Check out the [LICENSE](LICENSE) file for details.  

---



Happy traffic-optimizing! 🚦  
