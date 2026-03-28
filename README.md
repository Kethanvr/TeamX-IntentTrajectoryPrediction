# TeamX — Intent & Trajectory Prediction
### MAHE Mobility Challenge 2025 | AI Track | Problem Statement 1

---

## 🚗 What is this project?

Imagine a self-driving car moving through a busy urban street. There are pedestrians crossing roads, cyclists weaving through traffic, and people walking on sidewalks. The car's camera sees all of them — but **seeing is not enough**. By the time the car reacts to where a person *is*, it might already be too late.

This project solves that problem. We built a deep learning model that **predicts where pedestrians and cyclists will be in the next 3 seconds**, giving the autonomous vehicle enough time to plan a safe path around them.

This is called **trajectory prediction** — and it is one of the core problems in building safe Level 4 autonomous vehicles.

---

## 🎯 The Problem We Are Solving

**Problem Statement 1 — Intent & Trajectory Prediction**
MAHE Mobility Challenge, AI Track

In an L4 urban environment:
- The vehicle must predict future positions of pedestrians and cyclists
- Input: 2 seconds of past motion (where they have been)
- Output: 3 seconds of future motion (where they will be)
- The model must understand that people interact with each other — they avoid collisions, walk in groups, and change direction based on who is around them

**Why is this hard?**
- People are unpredictable — they stop, turn, and change speed suddenly
- Multiple people move at the same time and affect each other's paths
- The model must work in real-time on a moving vehicle
- Urban environments are complex — roads, sidewalks, crossings, crowds

---

## 🧠 How We Solved It

We used a **Spatio-Temporal Graph Convolutional Neural Network (ST-GCNN)** — a deep learning architecture specifically designed for multi-agent motion prediction.

### The Core Idea

We model the scene as a **graph**:
- Every pedestrian or cyclist is a **node** in the graph
- Two people who are close to each other are connected by an **edge**
- The strength of the edge represents how much one person influences the other's movement

This lets the model learn **social context** — the fact that people naturally avoid each other, walk together, and react to those around them.

### How the model processes data:

```
Step 1 — INPUT
   8 frames of (x, y) coordinates per person
   = 2 seconds of past movement history
   (nuScenes records at ~4 frames per second)

Step 2 — BUILD GRAPH
   For each frame, build a social graph
   Each person = one node
   People within a threshold distance = connected by edge

Step 3 — SPATIAL GRAPH CONVOLUTION
   The GCN layer passes messages between connected nodes
   Each person's feature gets updated based on their neighbors
   This captures: "Person A is moving towards Person B, so B will move away"

Step 4 — TEMPORAL CONVOLUTION
   A 1D CNN processes the sequence of graph features over time
   This captures: motion patterns like walking speed, acceleration, turning

Step 5 — OUTPUT
   12 frames of predicted (x, y) coordinates per person
   = 3 seconds of future trajectory
```

### Why this architecture works:

| Feature | Benefit |
|---|---|
| Graph structure | Naturally models relationships between people |
| Social pooling | Understands crowd dynamics and collision avoidance |
| Temporal convolution | Captures speed and direction changes over time |
| Lightweight design | Fast enough for real-time autonomous driving use |

---

## 📦 Dataset

We used the **nuScenes v1.0-mini** dataset — a real-world autonomous driving dataset collected on streets in **Boston, USA** and **Singapore**.

| Property | Details |
|---|---|
| Full name | nuScenes v1.0-mini |
| Collected by | Motional (formerly nuTonomy) |
| Cities | Boston, Singapore |
| Total scenes | 1000 (850 train, 150 val in mini split) |
| Sensors | 6 cameras, 1 LiDAR, 5 RADAR, GPS, IMU |
| Agent types | Pedestrians, Cyclists, Cars, Trucks, Buses |
| What we used | Pedestrian and cyclist (x, y) positions per frame |

**Why nuScenes?**
It is one of the most realistic and widely used autonomous driving datasets in the world. It contains real urban traffic scenarios with accurately annotated agent positions, making it perfect for training trajectory prediction models.

### Data Pipeline

The nuScenes dataset stores annotations in a complex JSON format with nested references. We wrote a **custom Python converter** that:
1. Loads the nuScenes SDK
2. Iterates through all scenes and samples
3. Filters for pedestrian and cyclist annotations only
4. Extracts (x, y) world coordinates per agent per frame
5. Assigns consistent IDs across frames to track the same person
6. Outputs clean tab-separated text files ready for model training

---

## 🏗️ Project Structure

```
TeamX-IntentTrajectoryPrediction/
│
├── train.py              ← Main training script
├── test.py               ← Evaluation script (gives ADE/FDE)
├── model.py              ← ST-GCNN model architecture
├── utils.py              ← Data loading and helper functions
├── metrics.py            ← ADE and FDE calculation
│
├── datasets/
│   └── nuscenes/
│       ├── train.txt     ← Converted training trajectories
│       └── val.txt       ← Converted validation trajectories
│
├── checkpoint/
│   └── nuscenes_run/     ← Saved model weights after training
│
└── README.md             ← This file
```

---

## ⚙️ How to Run This Project

### Requirements

- Python 3.8 or above
- GPU recommended (or use Google Colab free tier)
- ~2 GB disk space for dataset

### Step 1 — Clone the repository

```bash
git clone https://github.com/Kethanvr/TeamX-IntentTrajectoryPrediction.git
cd TeamX-IntentTrajectoryPrediction
```

### Step 2 — Install dependencies

```bash
pip install torch torchvision numpy nuscenes-devkit
```

### Step 3 — Download the dataset

Download **nuScenes v1.0-mini** from the official source or from the MAHE Mobility Challenge dataset link and extract it:

```bash
tar -xzf v1.0-mini.tgz -C ./data/
```

### Step 4 — Convert nuScenes data to trajectory format

```python
from nuscenes.nuscenes import NuScenes
import os

nusc = NuScenes(version='v1.0-mini', dataroot='./data/v1.0-mini', verbose=True)

os.makedirs('./datasets/nuscenes', exist_ok=True)

def convert(nusc, out_file):
    rows = []
    frame_id = 0
    for scene in nusc.scene:
        token = scene['first_sample_token']
        while token:
            sample = nusc.get('sample', token)
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                if 'human' not in ann['category_name']:
                    continue
                x, y, _ = ann['translation']
                rows.append(f"{frame_id}\t{ann['instance_token'][:6]}\t{x:.4f}\t{y:.4f}")
            frame_id += 1
            token = sample['next']
    with open(f'./datasets/nuscenes/{out_file}.txt', 'w') as f:
        f.write('\n'.join(rows))
    print(f"Saved {len(rows)} rows to {out_file}.txt")

convert(nusc, 'train')
convert(nusc, 'val')
```

### Step 5 — Train the model

```bash
python train.py \
  --lr 0.01 \
  --n_stgcnn 1 \
  --n_txpcnn 5 \
  --dataset nuscenes \
  --tag nuscenes_run \
  --use_lrschd \
  --num_epochs 50
```

Training takes approximately 20-30 minutes on a GPU. The best model checkpoint is automatically saved to `./checkpoint/nuscenes_run/`.

### Step 6 — Evaluate the model

```bash
python test.py \
  --dataset nuscenes \
  --tag nuscenes_run
```

This will print your **ADE** and **FDE** scores.

---

## 📊 Evaluation Metrics

We use two standard metrics for trajectory prediction:

### ADE — Average Displacement Error
The mean Euclidean distance between every predicted point and the corresponding ground truth point, averaged across all timesteps and all agents.

```
ADE = (1 / N*T) * Σ || predicted(t) - actual(t) ||
```

Lower ADE = the model's predicted paths are closer to where people actually went.

### FDE — Final Displacement Error
The Euclidean distance between only the **final predicted position** and the actual final position of the agent.

```
FDE = || predicted(last frame) - actual(last frame) ||
```

Lower FDE = the model correctly predicts where the person ends up after 3 seconds.

| Metric | Our Score | Interpretation |
|--------|-----------|----------------|
| ADE | TBD | Average path accuracy across 3 seconds |
| FDE | TBD | Accuracy at the 3-second endpoint |

---

## 🔧 Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| obs_seq_len | 8 | Observe 8 frames (2 seconds) of past motion |
| pred_seq_len | 12 | Predict 12 frames (3 seconds) of future motion |
| n_stgcnn | 1 | Number of spatial graph conv layers |
| n_txpcnn | 5 | Number of temporal conv layers |
| kernel_size | 3 | Temporal convolution kernel size |
| learning_rate | 0.01 | Initial learning rate |
| lr_scheduler | True | Reduces LR every 150 epochs |
| batch_size | 128 | Training batch size |
| epochs | 50 | Number of training epochs |

---

## 💻 Running on Google Colab (Free)

If you don't have a local GPU, you can run this entirely free on Google Colab:

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **Runtime → Change Runtime Type → T4 GPU**
3. Mount your Google Drive and upload the dataset
4. Run each step in separate cells as shown above

---

## 🧩 Challenges We Faced and How We Solved Them

**1. nuScenes data format is complex**
The raw nuScenes dataset uses a nested token-based JSON system with no direct trajectory files. We built a custom converter to extract clean (x,y) trajectories.

**2. PyTorch version incompatibility**
The original codebase was written for PyTorch 1.7. Google Colab runs PyTorch 2.x which is stricter about computation graphs. We patched the `loss.backward()` call with `retain_graph=True` to fix the runtime error.

**3. No local GPU available**
We used Google Colab's free T4 GPU tier which gave us sufficient compute to train the model in under 30 minutes.

---

## 👥 Team

**TeamX** — MAHE Mobility Challenge 2025
AI Track | Problem Statement 1: Intent & Trajectory Prediction

---

## 📜 Acknowledgements

- nuScenes dataset by Motional
- CVPR 2020 research on Spatio-Temporal Graph CNNs for trajectory prediction
- MAHE Mobility Challenge for the problem statement and dataset access