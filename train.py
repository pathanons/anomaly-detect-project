# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np
from datasets import download_stock_data, StockDataset
from model import LSTMAE, CNNLSTMAE, LSTMTransformerAE

# ตั้งค่า seed
torch.manual_seed(42)

# กำหนด hyperparameters
BATCH_SIZE = 32
EPOCHS = 20
WINDOW_SIZE = 20
LEARNING_RATE = 1e-3

# -----------------------------------------------------------------------------
# 1. กำหนดรายชื่อหุ้นที่ต้องการใช้ในการ train
# -----------------------------------------------------------------------------
ticker_list = [
    "AAPL",    # Apple Inc.
    "MSFT",    # Microsoft Corporation
    "GOOGL",   # Alphabet Inc.
    "AMZN",    # Amazon.com Inc.
    "TSLA",    # Tesla Inc.
    "BRK-B",   # Berkshire Hathaway Inc. (Class B)
    "JPM",     # JPMorgan Chase & Co.
    "JNJ",     # Johnson & Johnson
    "V",       # Visa Inc.
    "PG",      # Procter & Gamble Co.
    "NVDA",    # NVIDIA Corporation
    "HD",      # Home Depot Inc.
    "UNH",     # UnitedHealth Group Inc.
    "DIS",     # Walt Disney Co.
    "MA",      # Mastercard Incorporated
    "PFE",     # Pfizer Inc.
    "BAC",     # Bank of America Corp.
    "KO",      # Coca-Cola Co.
    "CMCSA"    # Comcast Corporation
]
start_date = "2006-01-01"
end_date   = "2019-12-31"

# -----------------------------------------------------------------------------
# 2. ดึงข้อมูลหุ้นจากแต่ละตัวและสร้าง Dataset
# -----------------------------------------------------------------------------
dataset_list = []
for ticker in ticker_list:
    print(f"Downloading data for {ticker} ...")
    df = download_stock_data(ticker, start_date, end_date)
    # สามารถทำ preprocessing หรือ normalization ได้ภายใน StockDataset
    dataset = StockDataset(df, window_size=WINDOW_SIZE)
    dataset_list.append(dataset)

# รวม Dataset จากหุ้นหลายตัวเข้าด้วยกัน
combined_dataset = ConcatDataset(dataset_list)
print(f"Total samples in combined dataset: {len(combined_dataset)}")

# -----------------------------------------------------------------------------
# 3. แบ่งข้อมูลเป็น training และ validation (80/20)
# -----------------------------------------------------------------------------
train_size = int(0.8 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------------------------------------------------
# 4. เลือกโมเดลที่ต้องการ train (ทดลองทั้ง 3 โมเดล)
# -----------------------------------------------------------------------------
models = {
    "LSTM_AE": LSTMAE(input_dim=5, hidden_dim=32),
    "CNN_LSTM_AE": CNNLSTMAE(input_dim=5, hidden_dim=32),
    "LSTM_Transformer_AE": LSTMTransformerAE(input_dim=5, hidden_dim=32)
}

def train_model(model, train_loader, val_loader, epochs, model_name="Model"):
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for batch_x, _ in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            # Gradient clipping เพื่อลดปัญหา gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item() * batch_x.size(0)
        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                output = model(batch_x)
                loss = criterion(output, batch_x)
                epoch_val_loss += loss.item() * batch_x.size(0)
        epoch_val_loss /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"{model_name} Epoch {epoch+1}/{epochs}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

    # Plot train vs validation loss
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss - {model_name}")
    plt.legend()
    plt.savefig(f"{model_name}_loss.png")
    plt.show()

    return model

# -----------------------------------------------------------------------------
# 5. Train โมเดลทั้ง 3
# -----------------------------------------------------------------------------
trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name} ...")
    trained_model = train_model(model, train_loader, val_loader, EPOCHS, model_name=name)
    trained_models[name] = trained_model

# -----------------------------------------------------------------------------
# 6. บันทึกโมเดล (optional)
# -----------------------------------------------------------------------------
for name, model in trained_models.items():
    torch.save(model.state_dict(), f"{name}.pth")
