# DeepMM: 14-Day Map Matching Challenge 🚗📍

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Completed-green)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 📖 Introduction

**DeepMM** is a project built from scratch over an intensive 14-day coding challenge. The goal was to engineer an industrial-grade **Map Matching Engine** capable of handling real-world GPS noise, complex road topologies, and ambiguous driving scenarios.

This project bridges the gap between **Deep Learning** (for visual trajectory similarity) and **Classic Algorithms** (for logical topology constraints), solving challenging problems such as "GPS drift on overpasses," "flickering on parallel roads," and "reverse driving detection."

## 🛠️ Tech Stack

* **Deep Learning:** PyTorch, LSTM, Siamese Networks (Contrastive Loss)
* **Algorithms:** Hidden Markov Model (HMM), Viterbi Algorithm, Sliding Window Inference
* **Geospatial:** GPX Parsing, Coordinate Projection (Lat/Lon to Meters), AABB (Axis-Aligned Bounding Box) Spatial Indexing
* **Graph Theory:** NetworkX, Topological Shortest Path (Dijkstra)
* **Visualization:** Matplotlib, Seaborn

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/1137725141/DeepMM.git
cd DeepMM
```
### 2. Environment Setup
It is recommended to use Conda for environment management:
```bash
conda create -n deepmm python=3.8
conda activate deepmm
pip install torch numpy matplotlib networkx seaborn scikit-learn
```
