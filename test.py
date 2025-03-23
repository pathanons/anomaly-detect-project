# test.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

from datasets import download_stock_data, StockDataset
from model import LSTMAE, CNNLSTMAE, LSTMTransformerAE

# ตั้งค่า seed (optional)
torch.manual_seed(42)

# -----------------------------------------------------------------------------
# 1) กำหนดช่วงเวลาสำหรับ Normal และ Black Swan
# -----------------------------------------------------------------------------
# Normal period (ตัวอย่าง: 2019-06-01 ถึง 2019-08-31)
start_normal = "2019-06-01"
end_normal   = "2019-08-31"

# Black Swan period (ตัวอย่าง: ช่วง COVID-19 Crash 2020-02-01 ถึง 2020-04-30)
start_bsw = "2020-02-01"
end_bsw   = "2020-04-30"

# เลือกหุ้นที่ต้องการทดสอบ
ticker = "AAPL"

# Hyperparameters
WINDOW_SIZE = 20
BATCH_SIZE = 32

# -----------------------------------------------------------------------------
# 2) โหลดข้อมูลและสร้าง DataLoader สำหรับ Normal และ Black Swan
# -----------------------------------------------------------------------------
df_normal = download_stock_data(ticker, start_normal, end_normal)
df_black_swan = download_stock_data(ticker, start_bsw, end_bsw)

dataset_normal = StockDataset(df_normal, window_size=WINDOW_SIZE)
dataset_black_swan = StockDataset(df_black_swan, window_size=WINDOW_SIZE)

loader_normal = DataLoader(dataset_normal, batch_size=BATCH_SIZE, shuffle=False)
loader_black_swan = DataLoader(dataset_black_swan, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------------------------------------------------------
# 3) โหลดโมเดลทั้งสามที่ฝึกไว้ และคำนวณ Reconstruction Error
# -----------------------------------------------------------------------------
models = {
    "LSTM_AE": LSTMAE(input_dim=5, hidden_dim=32),
    "CNN_LSTM_AE": CNNLSTMAE(input_dim=5, hidden_dim=32),
    "LSTM_Transformer_AE": LSTMTransformerAE(input_dim=5, hidden_dim=32)
}

# โหลด weights (สมมติว่าไฟล์ .pth อยู่ใน directory เดียวกัน)
for name, model in models.items():
    model.load_state_dict(torch.load(f"{name}.pth", map_location=torch.device('cpu')))
    model.eval()

def get_reconstruction_errors(model, loader):
    """
    คำนวณ reconstruction error (MSE) ของแต่ละ sequence แล้ว return เป็น numpy array
    """
    criterion = torch.nn.L1Loss(reduction='none')
    errors = []
    with torch.no_grad():
        for batch_x, _ in loader:
            output = model(batch_x)
            loss = criterion(output, batch_x)  # shape: [batch_size, seq_len, input_dim]
            loss = loss.mean(dim=(1,2))        # ค่าเฉลี่ยต่อ sample
            errors.extend(loss.cpu().numpy())
    return np.array(errors)

results = {}
for name, model in models.items():
    # คำนวณ error ช่วงปกติ (normal) และช่วง Black Swan
    err_normal = get_reconstruction_errors(model, loader_normal)
    err_bsw = get_reconstruction_errors(model, loader_black_swan)
    results[name] = {
        "normal": err_normal,
        "black": err_bsw
    }

# -----------------------------------------------------------------------------
# 4) Plot Histogram: เปรียบเทียบ distribution ของ Reconstruction Error
# -----------------------------------------------------------------------------
for name, errs in results.items():
    plt.figure(figsize=(10,5))
    plt.hist(errs["normal"], bins=30, alpha=0.5, label="Normal Period", density=True)
    plt.hist(errs["black"], bins=30, alpha=0.5, label="Black Swan Period", density=True)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Density")
    plt.title(f"Distribution of Reconstruction Errors - {name}")
    plt.legend()
    plt.savefig(f"{name}_error_distribution.png")
    plt.show()

# -----------------------------------------------------------------------------
# 5) ROC Curve เปรียบเทียบโมเดลทั้งสาม
# -----------------------------------------------------------------------------
plt.figure(figsize=(10,7))
aucs = {}  # เก็บค่า AUC ของแต่ละโมเดล

for name, errs in results.items():
    # สร้าง label: normal = 0, black swan = 1
    y_true = np.concatenate([
        np.zeros_like(errs["normal"]), 
        np.ones_like(errs["black"])
    ])
    # ใช้ reconstruction error เป็น score
    y_scores = np.concatenate([errs["normal"], errs["black"]])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    aucs[name] = roc_auc
    
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison for Black Swan Detection")
plt.legend(loc="lower right")
plt.savefig("ROC_Comparison.png")
plt.show()

# -----------------------------------------------------------------------------
# 6) Plot กราฟ Close Price และ Anomaly Score เฉพาะช่วง Black Swan
#    (พร้อมไฮไลต์จุด anomaly) สำหรับแต่ละโมเดล
# -----------------------------------------------------------------------------
def map_error_to_dates(df, errors, window_size):
    """
    คืนค่า (date_seq, errors_mapped) สำหรับ mapping error ให้ตรงกับวันที่
    โดยใช้ 'กึ่งกลาง' ของ sequence เป็นตัวแทน
    """
    dates = pd.to_datetime(df.index)
    # กำหนดวันที่เป็นกึ่งกลาง sequence
    date_seq = dates[window_size//2 : len(dates) - window_size//2]
    min_len = min(len(date_seq), len(errors))
    date_seq = date_seq[:min_len]
    errors_mapped = errors[:min_len]
    return date_seq, errors_mapped

for name, model in models.items():
    # ดึง reconstruction error ในช่วง Black Swan
    err_bsw = results[name]["black"]
    
    # map error กับวันที่ (กึ่งกลางของ window)
    date_seq, errors_bsw = map_error_to_dates(df_black_swan, err_bsw, WINDOW_SIZE)
    
    # สร้าง DataFrame สำหรับเก็บข้อมูล
    df_bsw_result = pd.DataFrame({
        "Date": date_seq,
        "Error": errors_bsw
    }).set_index("Date")
    
    # กำหนด threshold จากช่วง normal ของโมเดลเดียวกัน
    err_normal = results[name]["normal"]
    threshold = err_normal.mean() + 2 * err_normal.std()
    
    # หาจุด anomaly
    anomaly_indices = df_bsw_result.index[df_bsw_result["Error"] > threshold]
    
    # สร้างกราฟ 2 แกน
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), sharex=True)
    ax_price = axes[0]
    ax_error = axes[1]
    
    # Plot ราคา (Close Price) ในช่วง Black Swan
    ax_price.plot(df_black_swan.index, df_black_swan["Close"], label="Close Price", color='steelblue')
    ax_price.set_ylabel("Price")
    ax_price.set_title(f"{ticker} Price during Black Swan - {name}")
    
    # Plot Reconstruction Error
    ax_error.plot(df_bsw_result.index, df_bsw_result["Error"], label="Reconstruction Error", color='darkorange')
    ax_error.axhline(threshold, color='red', linestyle='--', label='Threshold')
    
    # ไฮไลต์จุด anomaly
    ax_error.scatter(anomaly_indices,
                     df_bsw_result.loc[anomaly_indices, "Error"],
                     color='red', marker='o', s=50, label='Anomaly Detected')
    
    ax_error.set_ylabel("Reconstruction Error")
    ax_error.set_title("Anomaly Score (Reconstruction Error)")
    ax_error.legend(loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{name}_BlackSwan_Price_Anomaly.png")
    plt.show()

# -----------------------------------------------------------------------------
# 7) สรุปผล: พิมพ์ AUC ของแต่ละโมเดล และบอกโมเดลที่ดีที่สุด
# -----------------------------------------------------------------------------
print("===== MODEL COMPARISON SUMMARY =====")
for name, score in aucs.items():
    print(f"- {name}: AUC = {score:.4f}")

best_model = max(aucs, key=aucs.get)
print(f"\n>>> Best model based on AUC is: {best_model} (AUC = {aucs[best_model]:.4f}) <<<")
