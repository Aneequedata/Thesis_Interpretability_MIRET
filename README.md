# MIRET: Multivariate Interpretable Rebuilt Tree

This repository contains the implementation of **MIRET (Multivariate Interpretable Rebuilt Tree)** as part of my Thesis during my Masters in Management Engineering(Business Intelligence and Analytics) from Sapienza University, a method developed to enhance the interpretability of decision trees while preserving predictive performance. The tool provides a multivariate optimization-based approach to construct interpretable trees with constraints on feature usage.

## Overview

MIRET bridges the gap between the interpretability of traditional decision trees and the complexity of tree ensemble methods like Random Forest and XGBoost. It allows the construction of trees with minimal features per split, making it especially useful for domains where interpretability is critical, such as healthcare and finance.

This implementation supports:
- Rebuilding decision trees to enforce constraints on the number of features used at each node.
- Generating proximity matrices to evaluate the relationships between data points based on the rebuilt tree.
- Analyzing feature importance and the impact of individual splits.

## Key Features
1. **Multivariate Optimization**:
   - Constructs decision trees using multivariate linear splits.
   - Enforces interpretability constraints (e.g., maximum features per split, minimum threshold).

2. **Feature Budgeting**:
   - Allows control over the number of features used across the tree.
   - Supports global and local feature selection strategies.

3. **Proximity Analysis**:
   - Computes proximity matrices to evaluate similarity between data points based on tree splits.
   - Generates clusters for enhanced data understanding.

4. **Performance Comparison**:
   - Benchmarks MIRET against traditional models like Random Forests.
   - Evaluates trade-offs between accuracy and interpretability.

## Repository Structure
- `main_md_xgb.py`: Main script for rebuilding trees and running MIRET experiments.
- `measures.py`: Contains methods to compute evaluation metrics (e.g., accuracy, precision).
- `preprocessing.py`: Includes utilities for dataset preprocessing and split generation.
- `RBT_prove.py`: Implementation of the core MIRET model, including multivariate optimization and tree construction.

## Getting Started

### Prerequisites
1. Install required Python packages:
   ```bash
   pip install numpy pandas scikit-learn gurobipy matplotlib pygraphviz
