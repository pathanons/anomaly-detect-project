# datasets.py (ปรับปรุงส่วน normalization)
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # หรือ StandardScaler ก็ได้
from torch.utils.data import Dataset

def download_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

def preprocess_data(df):
    scaler = MinMaxScaler()  # หรือ StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled, scaler

def create_sequences(data, window_size):
    sequences = []
    for i in range(len(data) - window_size):
        seq = data[i : i + window_size]
        sequences.append(seq)
    return np.array(sequences)

class StockDataset(Dataset):
    def __init__(self, df, window_size=30):
        # Normalize data ก่อนการสร้าง sequence
        df_scaled, self.scaler = preprocess_data(df)
        self.data = df_scaled.values.astype('float32')
        self.window_size = window_size
        self.sequences = create_sequences(self.data, window_size)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return seq, seq

if __name__ == "__main__":
    ticker = "AAPL"
    df = download_stock_data(ticker, "2015-01-01", "2021-12-31")
    df.to_csv(f"{ticker}_data.csv")
