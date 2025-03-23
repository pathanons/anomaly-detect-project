# model.py
import torch
import torch.nn as nn

# ------------------------
# Model 1: LSTM Autoencoder
# ------------------------
class LSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMAE, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim]
        _, (h, _) = self.encoder(x)
        # ใช้ hidden state เป็น input สำหรับ decoder replication ในทุก timestep
        dec_input = h.repeat(x.size(1), 1, 1).permute(1,0,2)
        reconstructed, _ = self.decoder(dec_input)
        return reconstructed

# ------------------------
# Model 2: 1DCNN-LSTM Autoencoder
# ------------------------
class CNNLSTMAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(CNNLSTMAE, self).__init__()
        # encoder: ใช้ 1D CNN ก่อน LSTM
        self.cnn_encoder = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, padding=kernel_size//2)
        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # decoder
        self.lstm_decoder = nn.LSTM(hidden_dim, input_dim, batch_first=True)
        self.cnn_decoder = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=kernel_size, padding=kernel_size//2)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        # เปลี่ยน shape ให้ CNN: [batch, input_dim, seq_len]
        x_cnn = x.permute(0,2,1)
        x_cnn = self.cnn_encoder(x_cnn)
        x_enc = x_cnn.permute(0,2,1)
        enc_out, (h, c) = self.lstm_encoder(x_enc)
        # ใช้ hidden state สำหรับ decoder
        dec_input = h.repeat(x.size(1), 1, 1).permute(1,0,2)
        dec_out, _ = self.lstm_decoder(dec_input)
        # ผ่าน CNN decoder
        dec_out = dec_out.permute(0,2,1)
        reconstructed = self.cnn_decoder(dec_out)
        reconstructed = reconstructed.permute(0,2,1)
        return reconstructed

# ------------------------
# Model 3: LSTM-Transformer Autoencoder
# ------------------------
class LSTMTransformerAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, nhead=2, num_transformer_layers=1):
        super(LSTMTransformerAE, self).__init__()
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        # LSTM decoder
        self.lstm_decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        lstm_out, (h, c) = self.lstm_encoder(x)  # lstm_out: [batch, seq_len, hidden_dim]
        # Transformer ต้องการ shape [seq_len, batch, hidden_dim]
        transformer_in = lstm_out.permute(1,0,2)
        transformer_out = self.transformer_encoder(transformer_in)
        transformer_out = transformer_out.permute(1,0,2)
        # ใช้ hidden state สำหรับ decoder reconstruction
        dec_input = transformer_out.mean(dim=1).unsqueeze(1).repeat(1, x.size(1), 1)
        reconstructed, _ = self.lstm_decoder(dec_input)
        return reconstructed

if __name__ == "__main__":
    # ทดสอบการสร้างโมเดล
    batch_size = 16
    seq_len = 30
    input_dim = 5  # [Open, High, Low, Close, Volume]
    x = torch.randn(batch_size, seq_len, input_dim)

    model1 = LSTMAE(input_dim, hidden_dim=32)
    model2 = CNNLSTMAE(input_dim, hidden_dim=32)
    model3 = LSTMTransformerAE(input_dim, hidden_dim=32)

    for model in [model1, model2, model3]:
        out = model(x)
        print(model.__class__.__name__, out.shape)
