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
  - `S0_init_project.ipynb`
  - `S1_ingest_prep.ipynb`
  - `S2_transform_load.ipynb`
  - `S3_load.ipynb`
  - `S4_coarsening.ipynb`
  - `S5_projection.ipynb`
  - `S7_embeddings_new_tracks.ipynb`
  - `S8_GraphSAGE.ipynbb`
  - `S9_recommender.ipynb`
  - `S10_visualization.ipynb`
- `conf/`: Configuration files (`config.yaml`).
- `src/`: Helper scripts and utilities.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/tcc-music-recsys-gnn.git](https://github.com/YOUR_USERNAME/tcc-music-recsys-gnn.git)
   cd tcc-music-recsys-gnn
