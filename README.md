# Black Swan Event Detection in Stock Markets

This project explores anomaly detection in stock markets using deep learning and machine learning techniques with a focus on identifying Black Swan events. It implements and compares three autoencoder-based models:

- **LSTM Autoencoder**
- **1D CNN-LSTM Autoencoder**
- **LSTM-Transformer Autoencoder**

The models are trained on a combined dataset from 20 diverse stocks sourced from Yahoo Finance. In addition, the project investigates the impact of different window sizes (i.e., the number of days per input sequence) on model performance.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Collection](#data-collection)
  - [Training](#training)
  - [Testing and Evaluation](#testing-and-evaluation)
  - [Window Size Experiment](#window-size-experiment)
- [Results](#results)
- [Appendix](#appendix)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Overview

Financial markets can experience extreme events (Black Swan events) that severely impact the economy. This project leverages AI, ML, and DL methods to learn the normal behavior of stock prices through autoencoder reconstruction. When the model is presented with abnormal data (e.g., during market crises), the reconstruction error increases significantly, flagging potential anomalies.

## Features

- **Data Acquisition:** Downloads historical stock data from Yahoo Finance.
- **Data Preprocessing:** Normalizes and segments data into sequences with configurable window sizes.
- **Model Implementation:** Includes three autoencoder models:
  - LSTM Autoencoder
  - 1D CNN-LSTM Autoencoder
  - LSTM-Transformer Autoencoder
- **Anomaly Detection:** Uses reconstruction error analysis to flag abnormal market behavior.
- **Hyperparameter Experimentation:** Evaluates model performance over various window sizes.
- **Visualization:** Plots training/validation loss curves, error distribution histograms, ROC curves, and anomaly detection overlays on stock price graphs.
- **Reproducibility:** All source code files are included in the repository and referenced in the LaTeX report.

## Project Structure
    ├── datasets.py # Code for downloading and processing stock data
    ├── model.py # Implementation of autoencoder-based models 
    ├── train.py # Training script for the models 
    ├── test.py # Testing and evaluation script for anomaly detection 
    ├── experiment_window_size.py # Script to experiment with different window sizes 
    ├── README.md # This file 
    ├── figures/ # Folder containing result figures (loss curves, ROC curves, etc.) 
    └── report/ # LaTeX report (Overleaf project files)


## Installation

Ensure you have Python 3.7 or later installed. Install the required packages using pip:

```bash
pip install torch torchvision torchaudio matplotlib pandas numpy yfinance scikit-learn
```

# Usage
### Data Collection
The datasets.py script downloads historical stock data from Yahoo Finance for 20 stock tickers. You can modify the ticker list and date range as needed.

### Example command:

```bash
python datasets.py
```

# Training
To train the models on the combined dataset, run:

```bash
python train.py
```

This script will: 
- Download and preprocess data for each ticker.
- Combine the datasets.
- Train the three autoencoder models and save their weights (e.g., LSTM_AE.pth, CNN_LSTM_AE.pth, LSTM_Transformer_AE.pth).
- Plot training and validation loss curves.

# Testing and Evaluation
To test the models for anomaly detection (e.g., during a Black Swan period such as the COVID-19 crash), run:

```bash
python test.py
```
This script will:

- Load stock data for both normal and Black Swan periods.
- Evaluate the reconstruction error and generate plots (error distribution histograms, ROC curves, and anomaly detection overlays on price graphs).

# Window Size Experiment
To evaluate how different window sizes affect model performance, run:

```bash
python experiment_window_size.py
```
This script will:
- Train each model using window sizes of 10, 20, 30, 40, and 50 days.
- Record the final validation loss for each model and window size.
- Generate a plot comparing performance across window sizes.

# Results
The project outputs various plots, including:
- Training/Validation Loss Curves: Indicating model convergence.
- Error Distribution Histograms: Showing the separation between normal and anomalous periods.
- Price and Anomaly Detection Overlays: Displaying stock prices with detected anomaly points.
- ROC Curves: Comparing model performance using AUC.
- Window Size Performance Plot: Visualizing the impact of window size on final validation loss.
- Check the figures/ folder for sample images such as:
    - LSTM_AE_loss.png
    - CNN_LSTM_AE_loss.png
    - LSTM_Transformer_AE_loss.png
    - CNN_LSTM_AE_error_distribution.png
    - LSTM_AE_error_distribution.png
    - LSTM_Transformer_AE_error_distribution.png
    - ROC_Comparison.png
    - WindowSize_Performance_Comparison.png