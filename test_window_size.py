# experiment_window_size.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from datasets import download_stock_data, StockDataset
from model import LSTMAE, CNNLSTMAE, LSTMTransformerAE

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 20  # Use a smaller number of epochs for faster experimentation
LEARNING_RATE = 1e-3

# Define list of 20 tickers for training
ticker_list = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NFLX", "NVDA", "JPM", "BAC",
    "WMT", "DIS", "INTC", "CSCO", "PFE", "KO", "MCD", "BA", "IBM", "GE"
]
start_date = "2015-01-01"
end_date   = "2019-12-31"

# Define window sizes to test
window_sizes = [10, 20, 30, 40, 50]

# Define model types (with untrained instances)
model_classes = {
    "LSTM_AE": LSTMAE,
    "CNN_LSTM_AE": CNNLSTMAE,
    "LSTM_Transformer_AE": LSTMTransformerAE
}

# This dictionary will store performance (final validation loss) for each model and window size
performance = {model_name: [] for model_name in model_classes.keys()}

def train_and_evaluate(model, train_loader, val_loader, epochs, model_name="Model"):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    val_losses = []
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * batch_x.size(0)
        epoch_train_loss /= len(train_loader.dataset)
        
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_x)
                epoch_val_loss += loss.item() * batch_x.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        print(f"{model_name} Epoch {epoch+1}/{epochs}: Val Loss = {epoch_val_loss:.4f}")
    return val_losses[-1]  # Return final validation loss

# Loop over window sizes
for w_size in window_sizes:
    print(f"\n=== Experimenting with window size: {w_size} ===")
    # Create dataset for each ticker and combine them
    dataset_list = []
    for ticker in ticker_list:
        print(f"Downloading data for {ticker} ...")
        df = download_stock_data(ticker, start_date, end_date)
        dataset = StockDataset(df, window_size=w_size)
        dataset_list.append(dataset)
    combined_dataset = ConcatDataset(dataset_list)
    print(f"Total samples with window size {w_size}: {len(combined_dataset)}")
    
    # Split into training and validation (80/20)
    train_size = int(0.8 * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # For each model type, initialize a new model, train and evaluate
    for model_name, ModelClass in model_classes.items():
        print(f"\nTraining model: {model_name} with window size {w_size}")
        # Instantiate the model. Input dim is 5 ([Open, High, Low, Close, Volume]), hidden_dim=32.
        model = ModelClass(input_dim=5, hidden_dim=32)
        final_val_loss = train_and_evaluate(model, train_loader, val_loader, EPOCHS, model_name)
        performance[model_name].append(final_val_loss)

# Plot the performance comparison across window sizes
plt.figure(figsize=(10,6))
for model_name, losses in performance.items():
    plt.plot(window_sizes, losses, marker='o', label=model_name)
plt.xlabel("Window Size")
plt.ylabel("Final Validation Loss")
plt.title("Model Performance vs. Window Size")
plt.legend()
plt.grid(True)
plt.savefig("WindowSize_Performance_Comparison.png")
plt.show()

# Print the performance dictionary for reference
print("Performance (Final Validation Loss) for each model and window size:")
for model_name, losses in performance.items():
    print(f"{model_name}: {losses}")
