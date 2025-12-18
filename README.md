# GraphSAGE Music Recommendation for Cold-Start 

This repository contains the implementation of a Thesis Project focused on solving the **Cold-Start problem** in music recommendation using **Graph Neural Networks (GNNs)** and **Graph Coarsening** techniques.

## Project Overview
The goal of this project is to recommend new music tracks (released after a specific cutoff date) that have no interaction history. By leveraging a hybrid approach that combines **Topological Structure** (via GraphSAGE) and **Content Features** (Audio Analysis), the system bridges the gap between collaborative filtering and content-based methods.

## Key Features
- **Graph Coarsening:** A hybrid dimensionality reduction pipeline that compresses the graph from **~300k to ~20k nodes**, preserving local community structures while ensuring scalability.
- **Inductive Learning:** Implementation of **GraphSAGE** to generate embeddings for unseen nodes (super-nodes) based on their features and neighborhood.
- **Hybrid Cold-Start Inference:** A novel inference strategy using **KNN** on raw audio features to map new tracks to learned Super-Node embeddings.
- **Reproducible Engineering:** The project uses a `config.yaml` based architecture with dynamic path management, ensuring portability across different environments.

## Project Structure
The pipeline is organized into sequential notebooks:

- `notebooks/`:
  - `01_preprocessing.ipynb`: Data cleaning and feature engineering (Spotify Audio Features).
  - `02_bipartite_graph.ipynb`: Construction of the Playlist-Track bipartite graph.
  - `03_projection.ipynb`: Projection to item-item similarity graph.
  - `04_coarsening.ipynb`: Implementation of the hybrid coarsening algorithm (Head/Tail strategy).
  - `05_super_projection.ipynb`: Projection of the coarsened graph.
  - `06_graphsage_training.ipynb`: Training the GNN via Link Prediction.
  - `07_evaluation.ipynb`: Cold-Start inference and metric calculation (Recall@K, NDCG@K).
- `conf/`: Configuration files (`config.yaml`).
- `src/`: Helper scripts and utilities.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/tcc-music-recsys-gnn.git](https://github.com/YOUR_USERNAME/tcc-music-recsys-gnn.git)
   cd tcc-music-recsys-gnn
