import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, TimeDistributed, RepeatVector, Lambda, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import pywt
import math
import warnings
warnings.filterwarnings('ignore')

# กำหนดค่าสีสำหรับการแสดงผล
sns.set(style="whitegrid")
COLORS = {
    'normal': 'blue',
    'black_swan': 'red',
    'prediction': 'green',
    'threshold': 'purple',
    'anomaly': 'orange'
}

class BlackSwanDetector:
    """
    ระบบตรวจจับเหตุการณ์ Black Swan ในตลาดหุ้น โดยใช้เทคนิคผสมผสานระหว่าง
    วิธีการทางสถิติและการเรียนรู้เชิงลึก (Deep Learning)
    """
    
    def __init__(self, seq_length=30, latent_dim=10, batch_size=32, epochs=50):
        """
        กำหนดค่าเริ่มต้นสำหรับระบบตรวจจับ Black Swan
        
        Args:
            seq_length (int): ความยาวของลำดับเวลาที่ใช้ในการวิเคราะห์ (จำนวนวันย้อนหลัง)
            latent_dim (int): ขนาดของ latent space สำหรับ VAE
            batch_size (int): ขนาด batch ในการฝึกโมเดล
            epochs (int): จำนวนรอบในการฝึกโมเดล
        """
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.models = {}  # เก็บโมเดลต่างๆ
        self.scalers = {}  # เก็บ scalers สำหรับข้อมูล
        self.data = None  # ข้อมูลดิบ
        self.train_data = None  # ข้อมูลฝึก
        self.test_data = None  # ข้อมูลทดสอบ
        self.features = None  # คุณลักษณะที่สกัดได้
        self.black_swan_events = {}  # เก็บข้อมูลเหตุการณ์ Black Swan
        self.results = {}  # เก็บผลการทำนาย
        self.history = {}  # เก็บประวัติการฝึกโมเดล
    
    def load_data(self, tickers, start_date, end_date, black_swan_periods=None):
        """
        โหลดข้อมูลหุ้นจาก Yahoo Finance และกำหนดช่วงเวลา Black Swan
        
        Args:
            tickers (list): รายชื่อหุ้นที่ต้องการวิเคราะห์
            start_date (str): วันเริ่มต้นในรูปแบบ 'YYYY-MM-DD'
            end_date (str): วันสิ้นสุดในรูปแบบ 'YYYY-MM-DD'
            black_swan_periods (dict): พจนานุกรมที่ระบุช่วงเวลา Black Swan
                เช่น {'2008-crisis': ('2008-09-01', '2008-10-31'), 
                      'covid-crash': ('2020-02-20', '2020-03-23')}
        
        Returns:
            self: คืนค่าตัวเองเพื่อให้สามารถเรียกเมธอดต่อเนื่องได้
        """
        print(f"กำลังโหลดข้อมูลหุ้น {tickers} ระหว่างวันที่ {start_date} ถึง {end_date}...")
        
        # โหลดข้อมูลหุ้น
        self.data = {}
        for ticker in tickers:
            self.data[ticker] = yf.download(ticker, start=start_date, end=end_date)
            print(f"  โหลดข้อมูล {ticker} สำเร็จ: {len(self.data[ticker])} วัน")
        
        # กำหนดช่วงเวลา Black Swan
        if black_swan_periods:
            self.black_swan_events = black_swan_periods
            
            # สร้างคอลัมน์ที่ระบุว่าวันใดเป็นช่วง Black Swan
            for ticker in tickers:
                self.data[ticker]['is_black_swan'] = 0
                
                for event_name, (start, end) in black_swan_periods.items():
                    # แปลงสตริงเป็น Timestamp
                    start_date = pd.Timestamp(start)
                    end_date = pd.Timestamp(end)
                    
                    # เลือกช่วงเวลาและกำหนดค่า
                    mask = (self.data[ticker].index >= start_date) & (self.data[ticker].index <= end_date)
                    self.data[ticker].loc[mask, 'is_black_swan'] = 1
                    
                    event_days = sum(mask)
                    print(f"  เหตุการณ์ {event_name} มี {event_days} วันในชุดข้อมูล {ticker}")
        
        return self
    
    def feature_engineering(self, tickers=None, wavelet_transform=True, technical_indicators=True):
        """
        สกัดคุณลักษณะจากข้อมูลราคาหุ้น
        
        Args:
            tickers (list): รายชื่อหุ้นที่ต้องการสกัดคุณลักษณะ ถ้าเป็น None จะใช้ทุกตัว
            wavelet_transform (bool): ใช้การแปลง wavelet หรือไม่
            technical_indicators (bool): ใช้ technical indicators หรือไม่
        
        Returns:
            self: คืนค่าตัวเองเพื่อให้สามารถเรียกเมธอดต่อเนื่องได้
        """
        print("กำลังสกัดคุณลักษณะจากข้อมูลราคาหุ้น...")
        
        if tickers is None:
            tickers = list(self.data.keys())
        
        self.features = {}
        
        for ticker in tickers:
            df = self.data[ticker].copy()
            features = pd.DataFrame(index=df.index)
            
            # คุณลักษณะพื้นฐาน
            # ผลตอบแทนรายวัน
            features['return'] = df['Close'].pct_change()
            # ผลตอบแทนสะสม 5 วัน
            features['return_5d'] = df['Close'].pct_change(5)
            # ความผันผวนของผลตอบแทน 20 วัน
            features['volatility_20d'] = features['return'].rolling(20).std()
            # ปริมาณการซื้อขายเทียบค่าเฉลี่ย 20 วัน
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # True Range และ Average True Range
            df['high_low'] = df['High'] - df['Low']
            df['high_close'] = np.abs(df['High'] - df['Close'].shift(1))
            df['low_close'] = np.abs(df['Low'] - df['Close'].shift(1))
            df['true_range'] = np.maximum(df['high_low'], np.maximum(df['high_close'], df['low_close']))
            features['atr_14d'] = df['true_range'].rolling(14).mean()
            
            # Z-score ของราคาเทียบค่าเฉลี่ย 20 วัน
            features['price_zscore'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).std()
            
            # การเปลี่ยนแปลงของค่าเฉลี่ยเคลื่อนที่
            features['ma_ratio_fast'] = df['Close'] / df['Close'].rolling(10).mean()
            features['ma_ratio_medium'] = df['Close'] / df['Close'].rolling(30).mean()
            features['ma_ratio_slow'] = df['Close'] / df['Close'].rolling(50).mean()
            
            # ความเบ้และความโด่งของผลตอบแทน 20 วัน
            features['return_skew_20d'] = features['return'].rolling(20).skew()
            features['return_kurt_20d'] = features['return'].rolling(20).kurt()
            
            # เพิ่ม Technical Indicators
            if technical_indicators:
                # RSI - Relative Strength Index
                delta = df['Close'].diff()
                gain = delta.mask(delta < 0, 0)
                loss = -delta.mask(delta > 0, 0)
                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()
                rs = avg_gain / avg_loss
                features['rsi_14d'] = 100 - (100 / (1 + rs))
                
                # MACD - Moving Average Convergence Divergence
                ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
                ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
                features['macd'] = ema_12 - ema_26
                features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
                features['macd_hist'] = features['macd'] - features['macd_signal']
                
                # Bollinger Bands
                # Bollinger Bands - วิธีการที่แตกต่างออกไป
                bb_middle = df['Close'].rolling(20).mean()
                bb_std = df['Close'].rolling(20).std()
                bb_upper = bb_middle + (bb_std * 2)
                bb_lower = bb_middle - (bb_std * 2)

                features['bb_middle'] = bb_middle
                features['bb_std'] = bb_std
                features['bb_upper'] = bb_upper
                features['bb_lower'] = bb_lower
                features['bb_width'] = (bb_upper - bb_lower) / bb_middle
                features['bb_pct'] = (df['Close'] - bb_lower) / (bb_upper - bb_lower)
                
                # Chaikin Money Flow
                mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
                mfv = mfm * df['Volume']
                features['cmf_20d'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()
            
            # การแปลง Wavelet
            if wavelet_transform:
                # ใช้ 'db4' wavelet กับผลตอบแทนรายวัน
                close_returns = features['return'].dropna().values
                
                # ตรวจสอบว่ามีข้อมูลเพียงพอ
                if len(close_returns) >= 16:  # ต้องมีอย่างน้อย 2^4 จุดสำหรับระดับ 4
                    coeffs, levels = self._wavelet_decomposition(close_returns, wavelet='db4', level=4)
                    
                    # สร้างคุณลักษณะจากการแปลง wavelet
                    for i, data in enumerate(levels):
                        # ตัดหรือขยายข้อมูลให้มีความยาวเท่ากับ close_returns
                        if len(data) > len(close_returns):
                            data = data[:len(close_returns)]
                        elif len(data) < len(close_returns):
                            data = np.pad(data, (0, len(close_returns) - len(data)), 
                                        mode='constant', constant_values=np.nan)
                        
                        if i == 0:
                            component_name = 'wavelet_approx'
                        else:
                            component_name = f'wavelet_detail_{i}'
                        
                        # ต้องตรวจสอบให้แน่ใจว่า index ที่เราใช้มีความยาวเท่ากับข้อมูล
                        valid_indices = features.dropna().index[:len(data)]
                        
                        # ทำให้แน่ใจว่าความยาวของข้อมูลและ index ตรงกัน
                        data_length = len(data)
                        index_length = len(valid_indices)
                        
                        if data_length > index_length:
                            data = data[:index_length]  # ตัดข้อมูลให้เท่ากับความยาว index
                        elif data_length < index_length:
                            valid_indices = valid_indices[:data_length]  # ตัด index ให้เท่ากับความยาวข้อมูล
                        
                        # สร้าง DataFrame ชั่วคราว
                        temp_df = pd.DataFrame({component_name: data}, index=valid_indices)
                        
                        # รวมเข้ากับ features โดยใช้ดัชนี
                        features = features.join(temp_df, how='left')
            
            # คำนวณพลังงานของแต่ละองค์ประกอบ (จัดการกับโมเดลแยกเพื่อหลีกเลี่ยงเรื่อง index)
            energy_values = temp_df[component_name].rolling(20).apply(
                lambda x: np.sum(x**2) if not np.isnan(np.sum(x**2)) else np.nan, raw=True)
            
            # สร้าง DataFrame สำหรับค่าพลังงาน
            energy_df = pd.DataFrame({f'{component_name}_energy': energy_values}, index=valid_indices)
            
            # รวมเข้ากับ features
            features = features.join(energy_df, how='left')
            
            # เพิ่มป้ายกำกับ Black Swan
            features['is_black_swan'] = df['is_black_swan']
            
            # ลบแถวที่มีค่า NaN
            features = features.dropna()
            
            # เก็บลงในพจนานุกรม
            self.features[ticker] = features
            print(f"  สกัดคุณลักษณะสำหรับ {ticker} สำเร็จ: {features.shape[1]} คุณลักษณะ, {len(features)} วัน")
        
        return self
    
    def _wavelet_decomposition(self, data, wavelet='db4', level=4):
        """
        แยกองค์ประกอบของอนุกรมเวลาด้วยการแปลง wavelet
        
        Args:
            data (array): ข้อมูลอนุกรมเวลา 1 มิติ
            wavelet (str): ประเภทของ wavelet
            level (int): จำนวนระดับในการแยกองค์ประกอบ
            
        Returns:
            tuple: (coefficients, reconstructed_components)
                โดยที่ coefficients คือสัมประสิทธิ์ wavelet
                และ reconstructed_components คือองค์ประกอบที่สร้างใหม่
        """
        # คำนวณสัมประสิทธิ์ wavelet
        coeffs = pywt.wavedec(data, wavelet, level=level)
        
        # สร้างข้อมูลใหม่จากแต่ละระดับความถี่
        reconstructed = []
        
        # สร้างชุดข้อมูลใหม่โดยแยกตามระดับความถี่
        for i in range(level + 1):
            coeff_copy = [np.zeros_like(c) for c in coeffs]
            coeff_copy[i] = coeffs[i]
            reconstructed.append(pywt.waverec(coeff_copy, wavelet))
        
        return coeffs, reconstructed
    
    def prepare_sequences(self, ticker, test_ratio=0.2, val_ratio=0.1):
        """
        เตรียมข้อมูลเป็นลำดับสำหรับการฝึกโมเดลและการทดสอบ
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการเตรียมข้อมูล
            test_ratio (float): สัดส่วนของข้อมูลทดสอบ
            val_ratio (float): สัดส่วนของข้อมูลตรวจสอบ
        
        Returns:
            tuple: (X_train, y_train, X_val, y_val, X_test, y_test)
                  ข้อมูลสำหรับการฝึก ตรวจสอบ และทดสอบ
        """
        print(f"กำลังเตรียมข้อมูลลำดับสำหรับ {ticker}...")
        
        # ดึงข้อมูลคุณลักษณะ
        features_df = self.features[ticker].copy()
        
        # แยกป้ายกำกับออกมา
        y = features_df['is_black_swan'].values
        
        # ลบคอลัมน์ป้ายกำกับออก
        X = features_df.drop('is_black_swan', axis=1)
        
        # เตรียม scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        
        # เก็บ scaler ไว้ใช้ภายหลัง
        self.scalers[ticker] = scaler
        
        # สร้างลำดับ
        X_seq, y_seq = self._create_sequences(X_scaled, y, self.seq_length)
        
        # แบ่งข้อมูล
        n = len(X_seq)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)
        train_size = n - test_size - val_size
        
        # ข้อมูลฝึก
        X_train, y_train = X_seq[:train_size], y_seq[:train_size]
        
        # ข้อมูลตรวจสอบ
        X_val, y_val = X_seq[train_size:train_size+val_size], y_seq[train_size:train_size+val_size]
        
        # ข้อมูลทดสอบ
        X_test, y_test = X_seq[train_size+val_size:], y_seq[train_size+val_size:]
        
        # เก็บดัชนีเวลาสำหรับการพล็อต
        # เราต้องเลื่อนดัชนีไปตามความยาวของลำดับ
        train_indices = features_df.index[self.seq_length:train_size+self.seq_length]
        val_indices = features_df.index[train_size+self.seq_length:train_size+val_size+self.seq_length]
        test_indices = features_df.index[train_size+val_size+self.seq_length:]
        
        # เก็บข้อมูลทั้งหมด
        self.train_data = {
            'X': X_train,
            'y': y_train,
            'indices': train_indices
        }
        
        self.val_data = {
            'X': X_val,
            'y': y_val,
            'indices': val_indices
        }
        
        self.test_data = {
            'X': X_test,
            'y': y_test,
            'indices': test_indices
        }
        
        print(f"  แบ่งข้อมูลสำเร็จ: ฝึก {len(X_train)} ลำดับ, "
              f"ตรวจสอบ {len(X_val)} ลำดับ, ทดสอบ {len(X_test)} ลำดับ")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def _create_sequences(self, X, y, seq_length):
        """
        สร้างลำดับข้อมูลสำหรับการเรียนรู้เชิงลึก
        
        Args:
            X (array): ข้อมูลคุณลักษณะ
            y (array): ป้ายกำกับ
            seq_length (int): ความยาวของลำดับ
        
        Returns:
            tuple: (X_seq, y_seq) ลำดับข้อมูลและป้ายกำกับ
        """
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - seq_length):
            X_seq.append(X[i:i + seq_length])
            # ใช้ป้ายกำกับของจุดสุดท้ายในลำดับ
            y_seq.append(1 if np.sum(y[i:i + seq_length]) > 0 else 0)
        
        return np.array(X_seq), np.array(y_seq)
    def build_cnn_lstm_autoencoder(self, ticker, cnn_filters=32, lstm_units=64):
        """
        สร้างโมเดล 1D-CNN-LSTM Autoencoder สำหรับการตรวจจับ Black Swan
        
        โมเดลนี้ใช้ 1D Convolutional layers เพื่อสกัดคุณลักษณะในระดับต่ำก่อน
        แล้วส่งต่อไปยัง LSTM layers เพื่อจับความสัมพันธ์เชิงลำดับในระยะยาว
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการสร้างโมเดล
            cnn_filters (int): จำนวนฟิลเตอร์ของ CNN layers
            lstm_units (int): จำนวนหน่วยของ LSTM layers
        
        Returns:
            self: คืนค่าตัวเองเพื่อให้สามารถเรียกเมธอดต่อเนื่องได้
        """
        print(f"กำลังสร้างโมเดล 1D-CNN-LSTM Autoencoder สำหรับ {ticker}...")
        
        # ดึงขนาดของข้อมูล
        input_dim = self.train_data['X'].shape[2]  # จำนวนคุณลักษณะ
        timesteps = self.seq_length  # ความยาวของลำดับเวลา
        
        # สร้าง encoder
        # 1. Input Layer
        encoder_inputs = Input(shape=(timesteps, input_dim))
        
        # 2. CNN Layers (จับรูปแบบในระดับต่ำ)
        # CNN แรกด้วย kernel ขนาดเล็กเพื่อจับรูปแบบระยะสั้น
        x = Conv1D(filters=cnn_filters, kernel_size=3, padding='same', activation='relu')(encoder_inputs)
        x = BatchNormalization()(x)
        
        # CNN ที่สองด้วย kernel ขนาดใหญ่ขึ้นเพื่อจับรูปแบบระยะกลาง
        x = Conv1D(filters=cnn_filters*2, kernel_size=5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)  # ลดความยาวของลำดับลงครึ่งหนึ่ง
        
        # 3. LSTM Layers (จับความสัมพันธ์เชิงลำดับ)
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        x = Dropout(0.3)(x)  # ป้องกัน overfitting
        
        # 4. Bottleneck Layer
        encoded = Bidirectional(LSTM(self.latent_dim, return_sequences=False))(x)
        
        # สร้าง decoder
        # 1. เริ่มจาก bottleneck ไปเป็นลำดับที่มีความยาวเท่าเดิม แต่สั้นลงเนื่องจาก pooling
        x = RepeatVector(timesteps // 2)(encoded)  # ความยาวลดลงเพราะ MaxPooling1D
        
        # 2. LSTM Layers
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        
        # 3. Upsampling เพื่อกลับไปยังความยาวเดิม
        # ใช้ TimeDistributed(UpSampling1D) เพื่อขยายความยาวลำดับกลับมา
        # แต่ keras ไม่มี UpSampling1D ใน TimeDistributed โดยตรง จึงใช้ Conv1D และ stride แทน
        x = Conv1D(filters=cnn_filters*2, kernel_size=3, padding='same', activation='relu')(x)
        x = Lambda(lambda x: K.repeat_elements(x, 2, axis=1))(x)  # ขยายความยาวเป็น 2 เท่า
        
        # 4. CNN Layers เพื่อฟื้นฟูรายละเอียด
        x = Conv1D(filters=cnn_filters, kernel_size=5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        
        # 5. Output Layer
        decoder_outputs = TimeDistributed(Dense(input_dim))(x)
        
        # สร้างโมเดล autoencoder
        autoencoder = Model(encoder_inputs, decoder_outputs)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # สร้างโมเดล encoder แยก
        encoder = Model(encoder_inputs, encoded)
        
        # สร้างโมเดล decoder แยก
        decoder_inputs = Input(shape=(self.latent_dim,))
        x = RepeatVector(timesteps // 2)(decoder_inputs)
        
        # ต้องสร้างชั้นใหม่เพื่อหลีกเลี่ยงปัญหากับโมเดลที่มีอยู่
        x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
        x = Dropout(0.3)(x)
        x = Conv1D(filters=cnn_filters*2, kernel_size=3, padding='same', activation='relu')(x)
        x = Lambda(lambda x: K.repeat_elements(x, 2, axis=1))(x)
        x = Conv1D(filters=cnn_filters, kernel_size=5, padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        decoder_outputs = TimeDistributed(Dense(input_dim))(x)
        
        decoder = Model(decoder_inputs, decoder_outputs)
        
        # เก็บโมเดลไว้
        self.models[f'{ticker}_vae'] = autoencoder  # ยังคงใช้ชื่อเดิมเพื่อไม่ให้ต้องแก้ไขโค้ดส่วนอื่น
        self.models[f'{ticker}_encoder'] = encoder
        self.models[f'{ticker}_decoder'] = decoder
        
        print(f"  สร้างโมเดล 1D-CNN-LSTM Autoencoder สำหรับ {ticker} สำเร็จ")
        
        return self
    def build_lstm_vae(self, ticker, intermediate_dim=64):
        print(f"กำลังสร้างโมเดล LSTM-VAE สำหรับ {ticker}...")
        
        # ดึงขนาดของข้อมูล
        input_dim = self.train_data['X'].shape[2]
        timesteps = self.seq_length
        latent_dim = self.latent_dim
        
        # สร้างโมเดล VAE ด้วย Model Subclassing
        class VAE(tf.keras.Model):
            def __init__(self, timesteps, input_dim, latent_dim, intermediate_dim, **kwargs):
                super(VAE, self).__init__(**kwargs)
                self.timesteps = timesteps
                self.latent_dim = latent_dim
                
                # Encoder
                self.encoder_lstm = Bidirectional(LSTM(intermediate_dim, return_sequences=False))
                self.encoder_dropout = Dropout(0.2)
                self.z_mean_layer = Dense(latent_dim, name='z_mean')
                self.z_log_var_layer = Dense(latent_dim, name='z_log_var')
                
                # Decoder
                self.decoder_repeat = RepeatVector(timesteps)
                self.decoder_lstm = LSTM(intermediate_dim, return_sequences=True)
                self.decoder_dense = TimeDistributed(Dense(input_dim))
            
            def encode(self, x):
                x = self.encoder_lstm(x)
                x = self.encoder_dropout(x)
                z_mean = self.z_mean_layer(x)
                z_log_var = self.z_log_var_layer(x)
                return z_mean, z_log_var
            
            def reparameterize(self, z_mean, z_log_var):
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                return z_mean + tf.exp(0.5 * z_log_var) * epsilon
            
            def decode(self, z):
                x = self.decoder_repeat(z)
                x = self.decoder_lstm(x)
                return self.decoder_dense(x)
            
            def call(self, inputs, training=None):
                z_mean, z_log_var = self.encode(inputs)
                z = self.reparameterize(z_mean, z_log_var)
                reconstructed = self.decode(z)
                
                # Add loss
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(tf.square(inputs - reconstructed), axis=[1, 2])
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
                )
                
                reconstruction_weight = 1.0
                kl_weight = 0.5
                
                self.add_loss(reconstruction_weight * reconstruction_loss + kl_weight * kl_loss)
                return reconstructed
        
        # สร้างโมเดล
        vae = VAE(timesteps, input_dim, latent_dim, intermediate_dim)
        
        # สร้าง dummy input เพื่อให้โมเดลสร้าง weights
        dummy_input = tf.zeros((1, timesteps, input_dim))
        _ = vae(dummy_input)
        
        # คอมไพล์โมเดล
        vae.compile(optimizer='adam')
        
        # สร้าง encoder และ decoder เพื่อเก็บแยก
        class Encoder(tf.keras.Model):
            def __init__(self, vae_model, **kwargs):
                super(Encoder, self).__init__(**kwargs)
                self.encoder_lstm = vae_model.encoder_lstm
                self.encoder_dropout = vae_model.encoder_dropout
                self.z_mean_layer = vae_model.z_mean_layer
                self.z_log_var_layer = vae_model.z_log_var_layer
            
            def call(self, inputs):
                x = self.encoder_lstm(inputs)
                x = self.encoder_dropout(x)
                z_mean = self.z_mean_layer(x)
                z_log_var = self.z_log_var_layer(x)
                
                # Reparameterize
                batch = tf.shape(z_mean)[0]
                dim = tf.shape(z_mean)[1]
                epsilon = tf.random.normal(shape=(batch, dim))
                z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
                
                return [z_mean, z_log_var, z]
        
        class Decoder(tf.keras.Model):
            def __init__(self, vae_model, **kwargs):
                super(Decoder, self).__init__(**kwargs)
                self.decoder_repeat = vae_model.decoder_repeat
                self.decoder_lstm = vae_model.decoder_lstm
                self.decoder_dense = vae_model.decoder_dense
            
            def call(self, inputs):
                x = self.decoder_repeat(inputs)
                x = self.decoder_lstm(x)
                return self.decoder_dense(x)
        
        encoder = Encoder(vae)
        decoder = Decoder(vae)
        
        # เก็บโมเดลไว้
        self.models[f'{ticker}_vae'] = vae
        self.models[f'{ticker}_encoder'] = encoder
        self.models[f'{ticker}_decoder'] = decoder
        
        print(f"  สร้างโมเดล LSTM-VAE สำหรับ {ticker} สำเร็จ")
        
        return self
    
    def build_statistical_models(self, ticker):
        """
        สร้างโมเดลทางสถิติเพื่อเปรียบเทียบ
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการสร้างโมเดล
        
        Returns:
            self: คืนค่าตัวเองเพื่อให้สามารถเรียกเมธอดต่อเนื่องได้
        """
        print(f"กำลังสร้างโมเดลทางสถิติสำหรับ {ticker}...")
        
        # สร้างโมเดล EWM (Exponentially Weighted Moving Average)
        ewm_detector = EWMDetector()
        
        # สร้างโมเดล IQR (Interquartile Range)
        iqr_detector = IQRDetector()
        
        # สร้างโมเดล Z-Score
        zscore_detector = ZScoreDetector()
        
        # เก็บโมเดลไว้
        self.models[f'{ticker}_ewm'] = ewm_detector
        self.models[f'{ticker}_iqr'] = iqr_detector
        self.models[f'{ticker}_zscore'] = zscore_detector
        
        print(f"  สร้างโมเดลทางสถิติสำหรับ {ticker} สำเร็จ")
        
        return self
    
    def train_models(self, ticker):
        """
        ฝึกโมเดลทั้งหมดสำหรับหุ้นที่ระบุ
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการฝึกโมเดล
        
        Returns:
            self: คืนค่าตัวเองเพื่อให้สามารถเรียกเมธอดต่อเนื่องได้
        """
        print(f"กำลังฝึกโมเดลสำหรับ {ticker}...")
        
        # ฝึกโมเดล LSTM-VAE
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f'models/{ticker}_vae.h5', monitor='val_loss', save_best_only=True)
        
        # LSTM-VAE ใช้เฉพาะข้อมูลปกติในการฝึกเท่านั้น
        # ในทางปฏิบัติ เราต้องการให้โมเดลเรียนรู้รูปแบบข้อมูลปกติ
        # และตรวจจับข้อมูลผิดปกติที่ไม่เคยเห็นมาก่อน
        normal_indices = self.train_data['y'] == 0
        X_normal = self.train_data['X'][normal_indices]
        
        # ใช้ข้อมูลตรวจสอบทั้งหมด (ทั้งปกติและผิดปกติ) สำหรับ validation
        X_val = self.val_data['X']
        
        # ฝึกโมเดล LSTM-VAE
        vae = self.models[f'{ticker}_vae']
        history = vae.fit(
            X_normal, 
            epochs=self.epochs, 
            batch_size=self.batch_size,
            validation_data=(X_val, None),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # เก็บประวัติการฝึกไว้
        self.history[f'{ticker}_vae'] = history.history
        
        # ฝึกโมเดลทางสถิติ (ใช้เฉพาะข้อมูลปกติในการฝึก)
        # สำหรับการฝึกโมเดลทางสถิติ เราต้องแปลงข้อมูลกลับเป็นอนุกรมเวลา 1 มิติ
        # โดยใช้ reconstruction error จากโมเดล LSTM-VAE
        X_train_flat = self._calculate_reconstruction_error(vae, self.train_data['X'])
        X_val_flat = self._calculate_reconstruction_error(vae, self.val_data['X'])
        
        # ฝึกโมเดล EWM
        ewm_detector = self.models[f'{ticker}_ewm']
        ewm_detector.fit(X_train_flat[normal_indices])
        
        # ฝึกโมเดล IQR
        iqr_detector = self.models[f'{ticker}_iqr']
        iqr_detector.fit(X_train_flat[normal_indices])
        
        # ฝึกโมเดล Z-Score
        zscore_detector = self.models[f'{ticker}_zscore']
        zscore_detector.fit(X_train_flat[normal_indices])
        
        print(f"  ฝึกโมเดลสำหรับ {ticker} สำเร็จ")
        
        return self
    
    def train_cross_asset_model(self, tickers, combined_model_name='combined'):
        """
        ฝึกโมเดลแบบ cross-asset โดยใช้ข้อมูลจากหุ้นหลายตัวพร้อมกัน
        
        Args:
            tickers (list): รายชื่อหุ้นที่ต้องการใช้ในการฝึก
            combined_model_name (str): ชื่อของโมเดลรวม
        
        Returns:
            self: คืนค่าตัวเองเพื่อให้สามารถเรียกเมธอดต่อเนื่องได้
        """
        print(f"กำลังฝึกโมเดลแบบ cross-asset จากหุ้น {tickers}...")
        
        # รวมข้อมูลฝึกจากทุกหุ้น
        X_train_combined = []
        y_train_combined = []
        X_val_combined = []
        y_val_combined = []
        
        for ticker in tickers:
            # ตรวจสอบว่ามีข้อมูลของหุ้นนี้หรือไม่
            if ticker not in self.train_data:
                self.prepare_sequences(ticker)
            
            # เพิ่มข้อมูลลงในชุดรวม
            X_train_combined.extend(self.train_data[ticker]['X'])
            y_train_combined.extend(self.train_data[ticker]['y'])
            X_val_combined.extend(self.val_data[ticker]['X'])
            y_val_combined.extend(self.val_data[ticker]['y'])
        
        # แปลงเป็น numpy arrays
        X_train_combined = np.array(X_train_combined)
        y_train_combined = np.array(y_train_combined)
        X_val_combined = np.array(X_val_combined)
        y_val_combined = np.array(y_val_combined)
        
        # สร้างโมเดลรวม
        self.build_lstm_vae(combined_model_name, intermediate_dim=128)  # ใช้โมเดลใหญ่ขึ้น
        
        # แยกข้อมูลปกติสำหรับการฝึก VAE
        normal_indices = y_train_combined == 0
        X_normal = X_train_combined[normal_indices]
        
        # ฝึกโมเดล
        vae = self.models[f'{combined_model_name}_vae']
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = vae.fit(
            X_normal, 
            epochs=self.epochs, 
            batch_size=self.batch_size,
            validation_data=(X_val_combined, None),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # เก็บประวัติการฝึก
        self.history[f'{combined_model_name}_vae'] = history.history
        
        print(f"  ฝึกโมเดลแบบ cross-asset สำเร็จ")
        
        return self
    
    def evaluate_cross_asset_model(self, combined_model_name, ticker, threshold_multiplier=3.0):
        """
        ประเมินประสิทธิภาพของโมเดล cross-asset กับข้อมูลทดสอบของหุ้นที่ระบุ
        
        Args:
            combined_model_name (str): ชื่อของโมเดลรวม
            ticker (str): รหัสหุ้นที่ต้องการประเมิน
            threshold_multiplier (float): ตัวคูณสำหรับการกำหนดค่าขีดแบ่ง
        
        Returns:
            dict: ผลการประเมินโมเดล
        """
        print(f"กำลังประเมินโมเดล cross-asset {combined_model_name} กับข้อมูลทดสอบของ {ticker}...")
        
        # ตรวจสอบว่ามีข้อมูลทดสอบหรือไม่
        if ticker not in self.test_data:
            print(f"ไม่พบข้อมูลทดสอบสำหรับ {ticker}")
            return None
            
        # ดึงข้อมูลทดสอบ
        X_test = self.test_data[ticker]['X']
        y_test = self.test_data[ticker]['y']
        test_indices = self.test_data[ticker]['indices']
        
        # เตรียมพจนานุกรมสำหรับเก็บผลลัพธ์
        results = {
            'y_true': y_test,
            'indices': test_indices,
            'anomaly_scores': {},
            'predictions': {},
            'metrics': {}
        }
        
        # ประเมินโมเดล cross-asset
        combined_vae = self.models[f'{combined_model_name}_vae']
        
        # คำนวณ reconstruction error
        rec_errors = self._calculate_reconstruction_error(combined_vae, X_test)
        
        # คำนวณค่าขีดแบ่งแบบไดนามิก
        vae_threshold = self._calculate_dynamic_threshold(rec_errors, threshold_multiplier)
        
        # เก็บคะแนนความผิดปกติและการทำนาย
        model_name = f'{combined_model_name}_vae'
        results['anomaly_scores'][model_name] = rec_errors
        results['predictions'][model_name] = (rec_errors > vae_threshold).astype(int)
        
        # คำนวณเมทริกซ์ความสับสน (confusion matrix) และรายงานการจำแนก
        results['metrics'][model_name] = {
            'confusion_matrix': confusion_matrix(y_test, results['predictions'][model_name]),
            'classification_report': classification_report(y_test, results['predictions'][model_name], output_dict=True),
            'threshold': vae_threshold
        }
        
        # พิมพ์ผลลัพธ์สรุป
        print(f"\nผลการประเมินโมเดล {combined_model_name} กับข้อมูลของ {ticker}:")
        report = results['metrics'][model_name]['classification_report']
        print(f"  Precision: {report['1']['precision']:.4f}")
        print(f"  Recall: {report['1']['recall']:.4f}")
        print(f"  F1-score: {report['1']['f1-score']:.4f}")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        
        # เก็บผลลัพธ์
        if 'cross_asset_results' not in self.__dict__:
            self.cross_asset_results = {}
        self.cross_asset_results[ticker] = results
        
        return results
    
    def plot_cross_asset_comparison(self, combined_model_name, ticker):
        """
        เปรียบเทียบผลการทำนายระหว่างโมเดลเฉพาะหุ้นและโมเดล cross-asset
        
        Args:
            combined_model_name (str): ชื่อของโมเดลรวม
            ticker (str): รหัสหุ้นที่ต้องการเปรียบเทียบ
        """
        # ตรวจสอบว่ามีผลลัพธ์ของทั้งสองโมเดลหรือไม่
        if ticker not in self.results or 'cross_asset_results' not in self.__dict__ or ticker not in self.cross_asset_results:
            print(f"ไม่พบผลการประเมินครบถ้วนสำหรับ {ticker}")
            return
        
        # ดึงผลลัพธ์
        single_results = self.results[ticker]
        cross_results = self.cross_asset_results[ticker]
        test_indices = single_results['indices']
        y_true = single_results['y_true']
        
        # คำนวณ ROC curve
        single_scores = single_results['anomaly_scores']['vae']
        cross_scores = cross_results['anomaly_scores'][f'{combined_model_name}_vae']
        
        # สร้าง ROC curve
        single_fpr, single_tpr, _ = roc_curve(y_true, single_scores)
        single_auc = auc(single_fpr, single_tpr)
        
        cross_fpr, cross_tpr, _ = roc_curve(y_true, cross_scores)
        cross_auc = auc(cross_fpr, cross_tpr)
        
        # แสดงกราฟ ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(single_fpr, single_tpr, color='blue', lw=2, 
                label=f'Single-Asset Model (AUC = {single_auc:.3f})')
        plt.plot(cross_fpr, cross_tpr, color='red', lw=2, 
                label=f'Cross-Asset Model (AUC = {cross_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Baseline')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Comparison for {ticker}')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(f'cross_asset_comparison_{ticker}_roc.png')
        plt.show()
        
        # เปรียบเทียบคะแนนความผิดปกติ
        # หาช่วง Black Swan
        black_swan_mask = y_true == 1
        black_swan_indices = test_indices[black_swan_mask]
        
        if len(black_swan_indices) > 0:
            # เลือกช่วงเวลาสำหรับการแสดงผล
            # (ก่อนและหลัง Black Swan 30 วัน)
            bs_start = black_swan_indices[0]
            bs_end = black_swan_indices[-1]
            
            # หาดัชนีของวันก่อนและหลัง Black Swan
            stock_data = self.data[ticker]
            start_idx = max(0, stock_data.index.get_loc(bs_start) - 30)
            end_idx = min(len(stock_data) - 1, stock_data.index.get_loc(bs_end) + 30)
            
            display_start = stock_data.index[start_idx]
            display_end = stock_data.index[end_idx]
            
            # สร้างกราฟเปรียบเทียบ
            plt.figure(figsize=(15, 10))
            
            # กราฟบน: ราคาหุ้น
            plt.subplot(3, 1, 1)
            plt.plot(stock_data.loc[display_start:display_end].index, 
                    stock_data.loc[display_start:display_end]['Close'], 
                    color='blue', label='Stock Price')
            
            # ไฮไลต์ช่วง Black Swan
            plt.axvspan(bs_start, bs_end, color='red', alpha=0.3, label='Black Swan Period')
            
            plt.title(f'Stock Price and Anomaly Scores Comparison for {ticker}')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)
            
            # กราฟกลาง: คะแนนความผิดปกติของโมเดลเฉพาะหุ้น
            plt.subplot(3, 1, 2)
            
            # หาดัชนีของช่วงที่ต้องการแสดง
            display_mask = np.logical_and(test_indices >= display_start, test_indices <= display_end)
            plot_indices = test_indices[display_mask]
            
            if len(plot_indices) > 0:
                # คะแนนความผิดปกติของโมเดลเฉพาะหุ้น
                single_display_scores = single_scores[display_mask]
                single_predictions = single_results['predictions']['vae'][display_mask]
                single_threshold = single_results['metrics']['vae']['threshold']
                
                plt.plot(plot_indices, single_display_scores, color='blue', label='Single-Asset Model Score')
                plt.axhline(y=single_threshold, color='blue', linestyle='--', label='Threshold')
                
                # แสดงจุดที่ตรวจพบว่าผิดปกติ
                anomaly_indices = plot_indices[single_predictions == 1]
                if len(anomaly_indices) > 0:
                    anomaly_scores = single_display_scores[single_predictions == 1]
                    plt.scatter(anomaly_indices, anomaly_scores, color='blue', s=50, alpha=0.7, 
                            label='Detected Anomalies')
                
                # ไฮไลต์ช่วง Black Swan
                plt.axvspan(bs_start, bs_end, color='red', alpha=0.1)
                
                plt.ylabel('Anomaly Score')
                plt.legend()
                plt.grid(True)
                
                # กราฟล่าง: คะแนนความผิดปกติของโมเดล cross-asset
                plt.subplot(3, 1, 3)
                
                # คะแนนความผิดปกติของโมเดล cross-asset
                cross_display_scores = cross_scores[display_mask]
                cross_predictions = cross_results['predictions'][f'{combined_model_name}_vae'][display_mask]
                cross_threshold = cross_results['metrics'][f'{combined_model_name}_vae']['threshold']
                
                plt.plot(plot_indices, cross_display_scores, color='red', label='Cross-Asset Model Score')
                plt.axhline(y=cross_threshold, color='red', linestyle='--', label='Threshold')
                
                # แสดงจุดที่ตรวจพบว่าผิดปกติ
                cross_anomaly_indices = plot_indices[cross_predictions == 1]
                if len(cross_anomaly_indices) > 0:
                    cross_anomaly_scores = cross_display_scores[cross_predictions == 1]
                    plt.scatter(cross_anomaly_indices, cross_anomaly_scores, color='red', s=50, alpha=0.7, 
                            label='Detected Anomalies')
                
                # ไฮไลต์ช่วง Black Swan
                plt.axvspan(bs_start, bs_end, color='red', alpha=0.1)
                
                plt.xlabel('Date')
                plt.ylabel('Anomaly Score')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(f'cross_asset_comparison_{ticker}_scores.png')
                plt.show()
    def run_cross_asset_pipeline(self, tickers, combined_model_name='combined'):
        """
        ดำเนินการตรวจจับ Black Swan แบบ cross-asset
        
        Args:
            tickers (list): รายชื่อหุ้นที่ต้องการใช้ในการวิเคราะห์
            combined_model_name (str): ชื่อของโมเดลรวม
        
        Returns:
            dict: ผลการวิเคราะห์
        """
        print(f"เริ่มกระบวนการตรวจจับ Black Swan แบบ cross-asset...")
        
        # 1. ฝึกโมเดล cross-asset
        self.train_cross_asset_model(tickers, combined_model_name)
        
        # 2. ประเมินโมเดลกับข้อมูลของแต่ละหุ้น
        cross_results = {}
        for ticker in tickers:
            cross_results[ticker] = self.evaluate_cross_asset_model(combined_model_name, ticker)
        
        # 3. เปรียบเทียบผลการทำนายกับโมเดลเฉพาะหุ้น
        for ticker in tickers:
            if ticker in self.results:
                self.plot_cross_asset_comparison(combined_model_name, ticker)
        
        return cross_results

    def _calculate_reconstruction_error(self, model, data):
        """
        คำนวณ reconstruction error สำหรับข้อมูลที่ให้มา
        
        Args:
            model: โมเดลที่ใช้ในการ reconstruct ข้อมูล
            data: ข้อมูลที่ต้องการคำนวณ error
        
        Returns:
            array: reconstruction error สำหรับแต่ละตัวอย่าง
        """
        # ทำนายและคำนวณ reconstruction error
        reconstructions = model.predict(data)
        # คำนวณ Mean Squared Error สำหรับแต่ละตัวอย่าง
        mse = np.mean(np.square(data - reconstructions), axis=(1, 2))
        return mse
    
    def evaluate_models(self, ticker, threshold_multiplier=3.0):
        """
        ประเมินประสิทธิภาพของโมเดลทั้งหมดบนข้อมูลทดสอบ
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการประเมินโมเดล
            threshold_multiplier (float): ตัวคูณสำหรับการกำหนดค่าขีดแบ่ง (threshold)
        
        Returns:
            dict: ผลการประเมินโมเดล
        """
        print(f"กำลังประเมินโมเดลสำหรับ {ticker}...")
        
        # ดึงข้อมูลทดสอบ
        X_test = self.test_data['X']
        y_test = self.test_data['y']
        test_indices = self.test_data['indices']
        
        # เตรียมพจนานุกรมสำหรับเก็บผลลัพธ์
        results = {
            'y_true': y_test,
            'indices': test_indices,
            'anomaly_scores': {},
            'predictions': {},
            'metrics': {}
        }
        
        # ประเมินโมเดล LSTM-VAE
        vae = self.models[f'{ticker}_vae']
        
        # คำนวณ reconstruction error
        rec_errors = self._calculate_reconstruction_error(vae, X_test)
        
        # คำนวณค่าขีดแบ่งแบบไดนามิก
        vae_threshold = self._calculate_dynamic_threshold(rec_errors, threshold_multiplier)
        
        # เก็บคะแนนความผิดปกติและการทำนาย
        results['anomaly_scores']['vae'] = rec_errors
        results['predictions']['vae'] = (rec_errors > vae_threshold).astype(int)
        
        # คำนวณเมทริกซ์ความสับสน (confusion matrix) และรายงานการจำแนก
        results['metrics']['vae'] = {
            'confusion_matrix': confusion_matrix(y_test, results['predictions']['vae']),
            'classification_report': classification_report(y_test, results['predictions']['vae'], output_dict=True),
            'threshold': vae_threshold
        }
        
        # ประเมินโมเดลทางสถิติ
        # เตรียมข้อมูลสำหรับโมเดลทางสถิติ
        X_test_flat = self._calculate_reconstruction_error(vae, X_test)
        
        # ประเมินโมเดล EWM
        ewm_detector = self.models[f'{ticker}_ewm']
        ewm_scores = ewm_detector.predict(X_test_flat)
        ewm_threshold = self._calculate_dynamic_threshold(ewm_scores, threshold_multiplier)
        results['anomaly_scores']['ewm'] = ewm_scores
        results['predictions']['ewm'] = (ewm_scores > ewm_threshold).astype(int)
        results['metrics']['ewm'] = {
            'confusion_matrix': confusion_matrix(y_test, results['predictions']['ewm']),
            'classification_report': classification_report(y_test, results['predictions']['ewm'], output_dict=True),
            'threshold': ewm_threshold
        }
        
        # ประเมินโมเดล IQR
        iqr_detector = self.models[f'{ticker}_iqr']
        iqr_scores = iqr_detector.predict(X_test_flat)
        iqr_threshold = self._calculate_dynamic_threshold(iqr_scores, threshold_multiplier)
        results['anomaly_scores']['iqr'] = iqr_scores
        results['predictions']['iqr'] = (iqr_scores > iqr_threshold).astype(int)
        results['metrics']['iqr'] = {
            'confusion_matrix': confusion_matrix(y_test, results['predictions']['iqr']),
            'classification_report': classification_report(y_test, results['predictions']['iqr'], output_dict=True),
            'threshold': iqr_threshold
        }
        
        # ประเมินโมเดล Z-Score
        zscore_detector = self.models[f'{ticker}_zscore']
        zscore_scores = zscore_detector.predict(X_test_flat)
        zscore_threshold = self._calculate_dynamic_threshold(zscore_scores, threshold_multiplier)
        results['anomaly_scores']['zscore'] = zscore_scores
        results['predictions']['zscore'] = (zscore_scores > zscore_threshold).astype(int)
        results['metrics']['zscore'] = {
            'confusion_matrix': confusion_matrix(y_test, results['predictions']['zscore']),
            'classification_report': classification_report(y_test, results['predictions']['zscore'], output_dict=True),
            'threshold': zscore_threshold
        }
        
        # เก็บผลลัพธ์
        self.results[ticker] = results
        
        # พิมพ์ผลลัพธ์สรุป
        print(f"\nผลการประเมินโมเดลสำหรับ {ticker}:")
        for model_name, metrics in results['metrics'].items():
            report = metrics['classification_report']
            print(f"  {model_name.upper()}:")
            print(f"    Precision: {report['1']['precision']:.4f}")
            print(f"    Recall: {report['1']['recall']:.4f}")
            print(f"    F1-score: {report['1']['f1-score']:.4f}")
            print(f"    Accuracy: {report['accuracy']:.4f}")
        
        return results
    
    def _calculate_dynamic_threshold(self, scores, multiplier=3.0, window_size=20):
        """
        คำนวณค่าขีดแบ่งแบบไดนามิกจากคะแนนความผิดปกติ
        
        Args:
            scores (array): คะแนนความผิดปกติ
            multiplier (float): ตัวคูณสำหรับการกำหนดค่าขีดแบ่ง
            window_size (int): ขนาดของหน้าต่างเลื่อน
        
        Returns:
            float: ค่าขีดแบ่ง
        """
        # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
        mean = np.mean(scores)
        std = np.std(scores)
        
        # คำนวณค่าขีดแบ่ง
        threshold = mean + multiplier * std
        
        return threshold
    
    def plot_training_history(self, ticker):
        """
        แสดงกราฟประวัติการฝึกโมเดล
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการแสดงประวัติการฝึก
        """
        if f'{ticker}_vae' not in self.history:
            print(f"ไม่พบประวัติการฝึกสำหรับ {ticker}")
            return
        
        history = self.history[f'{ticker}_vae']
        
        plt.figure(figsize=(12, 5))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'Loss ระหว่างการฝึกโมเดล LSTM-VAE สำหรับ {ticker}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_roc_curves(self, ticker):
        """
        แสดงกราฟ ROC Curve สำหรับทุกโมเดล
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการแสดง ROC Curve
        """
        if ticker not in self.results:
            print(f"ไม่พบผลการประเมินสำหรับ {ticker}")
            return
        
        results = self.results[ticker]
        y_true = results['y_true']
        
        plt.figure(figsize=(10, 8))
        
        # สีสำหรับแต่ละโมเดล
        colors = {
            'vae': 'blue',
            'ewm': 'orange',
            'iqr': 'green',
            'zscore': 'red'
        }
        
        # ชื่อที่แสดงในกราฟ
        model_names = {
            'vae': 'LSTM_VAE_Autoencoder',
            'ewm': 'EWM_Detector',
            'iqr': 'IQR_Detector',
            'zscore': 'Zscore_Detector'
        }
        
        # แสดง ROC Curve สำหรับแต่ละโมเดล
        for model_name, scores in results['anomaly_scores'].items():
            fpr, tpr, _ = roc_curve(y_true, scores)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[model_name], lw=2,
                     label=f'{model_names[model_name]} (AUC = {roc_auc:.3f})')
        
        # แสดงเส้น baseline
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Baseline')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()
    
    def plot_stock_with_anomalies(self, ticker, window_size=30):
        """
        แสดงกราฟราคาหุ้นพร้อมกับจุดที่ตรวจพบว่าผิดปกติ
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการแสดงกราฟ
            window_size (int): ขนาดของหน้าต่างที่ใช้แสดง (จำนวนวัน)
        """
        if ticker not in self.results:
            print(f"ไม่พบผลการประเมินสำหรับ {ticker}")
            return
        
        results = self.results[ticker]
        test_indices = results['indices']
        
        # ดึงข้อมูลราคาหุ้น
        stock_data = self.data[ticker]
        
        # เลือกเฉพาะช่วงเวลาทดสอบและรีเซ็ตดัชนี
        test_data = stock_data.loc[test_indices].copy()  # ใช้ .copy() เพื่อหลีกเลี่ยงการเตือน SettingWithCopyWarning
        
        # หาช่วง Black Swan ด้วยวิธีที่แตกต่าง - ใช้ข้อมูลจาก y_true โดยตรง
        # เนื่องจากเรามี y_true ที่เป็น array อยู่แล้วในผลลัพธ์
        y_true = results['y_true']
        
        # แปลง y_true เป็นช่วงของวันที่ Black Swan
        black_swan_periods = []
        is_black_swan = False
        start_idx = None
        
        for i, (idx, is_bs) in enumerate(zip(test_indices, y_true)):
            if is_bs == 1 and not is_black_swan:
                is_black_swan = True
                start_idx = idx
            elif is_bs == 0 and is_black_swan:
                is_black_swan = False
                black_swan_periods.append((start_idx, test_indices[i-1]))
        
        # ถ้าช่วง Black Swan ยังไม่จบ
        if is_black_swan:
            black_swan_periods.append((start_idx, test_indices[-1]))
        
        # กรณีที่ไม่พบช่วง Black Swan เลย
        if len(black_swan_periods) == 0:
            print(f"ไม่พบช่วง Black Swan ในข้อมูลทดสอบของ {ticker}")
            # สร้างช่วงจากทั้งชุดข้อมูลทดสอบ
            black_swan_periods = [(test_indices[0], test_indices[-1])]
        
        # เลือกโมเดลที่ต้องการแสดงผล
        model_names = ['vae', 'ewm', 'iqr', 'zscore']
        model_labels = ['LSTM-VAE', 'EWM', 'IQR', 'Z-Score']
        
        # แสดงกราฟสำหรับแต่ละช่วง Black Swan
        for start, end in black_swan_periods:
            # หาวันเริ่มต้นและสิ้นสุดของช่วงที่ต้องการแสดง
            # แสดงข้อมูล window_size วันก่อนและหลัง Black Swan
            start_loc = max(0, stock_data.index.get_loc(start) - window_size)
            end_loc = min(len(stock_data) - 1, stock_data.index.get_loc(end) + window_size)
            
            display_start = stock_data.index[start_loc]
            display_end = stock_data.index[end_loc]
            
            # เลือกข้อมูลที่ต้องการแสดง
            display_data = stock_data.loc[display_start:display_end]
            
            # สร้างกราฟ
            fig, axes = plt.subplots(len(model_names) + 1, 1, figsize=(15, 4 * (len(model_names) + 1)), sharex=True)
            
            # แสดงราคาหุ้น
            ax = axes[0]
            ax.plot(display_data.index, display_data['Close'], color='blue', label='Stock Price')
            
            # ไฮไลต์ช่วง Black Swan
            ax.axvspan(start, end, color='red', alpha=0.3, label='Black Swan Period')
            
            # กำหนดชื่อแกนและชื่อกราฟ
            ax.set_title(f'Stock Price with Black Swan Period - {ticker}')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            
            # แสดงคะแนนความผิดปกติสำหรับแต่ละโมเดล
            for i, (model_name, model_label) in enumerate(zip(model_names, model_labels)):
                ax = axes[i + 1]
                
                # หาดัชนีของช่วงที่ต้องการแสดง
                mask = np.logical_and(test_indices >= display_start, test_indices <= display_end)
                if not np.any(mask):
                    ax.text(0.5, 0.5, "No test data in this range", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                    continue
                    
                plot_indices = test_indices[mask]
                
                # ดึงคะแนนความผิดปกติ
                if model_name in results['anomaly_scores']:
                    scores = results['anomaly_scores'][model_name][mask]
                    predictions = results['predictions'][model_name][mask]
                    threshold = results['metrics'][model_name]['threshold']
                    
                    # แสดงคะแนนความผิดปกติ
                    ax.plot(plot_indices, scores, color='green', label='Anomaly Score')
                    
                    # แสดงค่าขีดแบ่ง
                    ax.axhline(y=threshold, color='purple', linestyle='--', label='Threshold')
                    
                    # แสดงจุดที่ตรวจพบว่าผิดปกติ
                    anomaly_indices = plot_indices[predictions == 1]
                    if len(anomaly_indices) > 0:
                        anomaly_scores = scores[predictions == 1]
                        ax.scatter(anomaly_indices, anomaly_scores, color='red', s=50, label='Detected Anomalies')
                    
                    # ไฮไลต์ช่วง Black Swan
                    ax.axvspan(start, end, color='red', alpha=0.1)
                    
                    # กำหนดชื่อแกนและชื่อกราฟ
                    ax.set_title(f'Anomaly Scores - {model_label}')
                    ax.set_ylabel('Score')
                    ax.legend()
                    ax.grid(True)
                else:
                    ax.text(0.5, 0.5, f"No data available for {model_label}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
            
            plt.tight_layout()
            plt.savefig(f'black_swan_{ticker}_{start.strftime("%Y%m%d")}_to_{end.strftime("%Y%m%d")}.png')
            plt.show()
    
    def feature_importance_analysis(self, ticker):
        """
        วิเคราะห์ความสำคัญของคุณลักษณะในการตรวจจับ Black Swan
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการวิเคราะห์
        """
        if ticker not in self.results:
            print(f"ไม่พบผลการประเมินสำหรับ {ticker}")
            return
        
        # ดึงข้อมูลคุณลักษณะเฉพาะช่วงทดสอบ
        features_df = self.features[ticker].copy()
        test_indices = self.results[ticker]['indices']
        test_features = features_df.loc[test_indices].drop('is_black_swan', axis=1)
        
        # ดึงป้ายกำกับสำหรับช่วงทดสอบ
        y_test = self.results[ticker]['y_true']
        
        # คำนวณค่าสหสัมพันธ์ระหว่างคุณลักษณะและป้ายกำกับ (Black Swan)
        correlations = {}
        for column in test_features.columns:
            corr = np.corrcoef(test_features[column].values, y_test)[0, 1]
            correlations[column] = abs(corr)  # ใช้ค่าสัมบูรณ์เพื่อดูเฉพาะขนาดของความสัมพันธ์
        
        # เรียงลำดับคุณลักษณะตามความสำคัญ
        sorted_correlations = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        # แสดงกราฟความสำคัญของคุณลักษณะ (แสดง 15 อันดับแรก)
        n_features = min(15, len(sorted_correlations))
        top_features = sorted_correlations[:n_features]
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh([x[0] for x in top_features], [x[1] for x in top_features])
        
        # กำหนดสีตามความสำคัญ
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / n_features))
        
        plt.xlabel('Feature Importance (Absolute Correlation)')
        plt.title(f'Top {n_features} Important Features for Black Swan Detection - {ticker}')
        plt.gca().invert_yaxis()  # แสดงอันดับ 1 ที่ด้านบน
        plt.tight_layout()
        plt.show()
        
        # แสดงกราฟคุณลักษณะสำคัญระหว่างช่วงปกติและช่วง Black Swan
        fig, axes = plt.subplots(5, 1, figsize=(15, 20), sharex=True)
        
        for i, (feature_name, _) in enumerate(top_features[:5]):
            ax = axes[i]
            
            # ดึงค่าคุณลักษณะ
            feature_values = test_features[feature_name].values
            
            # แสดงค่าคุณลักษณะ
            ax.plot(test_indices, feature_values, color='blue', label=feature_name)
            
            # แยกจุดเป็นปกติและ Black Swan
            normal_indices = test_indices[y_test == 0]
            normal_values = feature_values[y_test == 0]
            
            black_swan_indices = test_indices[y_test == 1]
            black_swan_values = feature_values[y_test == 1]
            
            # แสดงจุด
            ax.scatter(normal_indices, normal_values, color='green', alpha=0.5, label='Normal')
            ax.scatter(black_swan_indices, black_swan_values, color='red', alpha=0.5, label='Black Swan')
            
            # ไฮไลต์ช่วง Black Swan
            for start, end in self.black_swan_events.values():
                start_date = pd.Timestamp(start)
                end_date = pd.Timestamp(end)
                if start_date in test_indices or end_date in test_indices:
                    ax.axvspan(start_date, end_date, color='red', alpha=0.1)
            
            # กำหนดชื่อแกนและชื่อกราฟ
            ax.set_title(f'Feature: {feature_name}')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return sorted_correlations
    
    def run_pipeline(self, ticker, test_ratio=0.2, val_ratio=0.1, threshold_multiplier=3.0):
        """
        ดำเนินการตรวจจับ Black Swan ทั้งกระบวนการสำหรับหุ้นหนึ่งตัว
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการวิเคราะห์
            test_ratio (float): สัดส่วนของข้อมูลทดสอบ
            val_ratio (float): สัดส่วนของข้อมูลตรวจสอบ
            threshold_multiplier (float): ตัวคูณสำหรับการกำหนดค่าขีดแบ่ง
        
        Returns:
            dict: ผลการวิเคราะห์
        """
        print(f"เริ่มกระบวนการตรวจจับ Black Swan สำหรับ {ticker}...")
        
        # 1. เตรียมข้อมูลลำดับ
        self.prepare_sequences(ticker, test_ratio=test_ratio, val_ratio=val_ratio)
        
        # 2. สร้างโมเดล
        self.build_lstm_vae(ticker)
        self.build_statistical_models(ticker)
        
        # 3. ฝึกโมเดล
        self.train_models(ticker)
        
        # 4. ประเมินโมเดล
        results = self.evaluate_models(ticker, threshold_multiplier=threshold_multiplier)
        
        # 5. แสดงกราฟประวัติการฝึก
        self.plot_training_history(ticker)
        
        # 6. แสดงกราฟ ROC Curve
        self.plot_roc_curves(ticker)
        
        # 7. แสดงกราฟราคาหุ้นพร้อมกับจุดที่ตรวจพบว่าผิดปกติ
        self.plot_stock_with_anomalies(ticker)
        
        # 8. วิเคราะห์ความสำคัญของคุณลักษณะ
        feature_importance = self.feature_importance_analysis(ticker)
        
        # 9. สรุปผล
        self._summarize_results(ticker)
        
        return self.results[ticker]
    
    def _summarize_results(self, ticker):
        """
        สรุปผลการวิเคราะห์
        
        Args:
            ticker (str): รหัสหุ้นที่ต้องการสรุปผล
        """
        if ticker not in self.results:
            print(f"ไม่พบผลการประเมินสำหรับ {ticker}")
            return
        
        results = self.results[ticker]
        
        print("\n" + "="*80)
        print(f"สรุปผลการตรวจจับ Black Swan สำหรับ {ticker}")
        print("="*80)
        
        # สรุปประสิทธิภาพของแต่ละโมเดล
        print("\nประสิทธิภาพของโมเดล:")
        print("-"*50)
        
        for model_name, metrics in results['metrics'].items():
            report = metrics['classification_report']
            print(f"{model_name.upper()}:")
            print(f"  Precision: {report['1']['precision']:.4f}")
            print(f"  Recall: {report['1']['recall']:.4f}")
            print(f"  F1-score: {report['1']['f1-score']:.4f}")
            print(f"  Accuracy: {report['accuracy']:.4f}")
            print("-"*30)
        
        # สรุปจำนวน Black Swan ที่ตรวจพบ
        y_true = results['y_true']
        total_days = len(y_true)
        black_swan_days = np.sum(y_true)
        
        print(f"\nจำนวนวันทั้งหมดในชุดข้อมูลทดสอบ: {total_days}")
        print(f"จำนวนวันที่เป็น Black Swan: {black_swan_days} ({black_swan_days/total_days*100:.2f}%)")
        
        for model_name, predictions in results['predictions'].items():
            detected = np.sum(predictions)
            true_positives = np.sum((predictions == 1) & (y_true == 1))
            false_positives = np.sum((predictions == 1) & (y_true == 0))
            
            print(f"\n{model_name.upper()}:")
            print(f"  จำนวนวันที่ตรวจพบว่าผิดปกติ: {detected} ({detected/total_days*100:.2f}%)")
            print(f"  True Positives: {true_positives} ({true_positives/black_swan_days*100:.2f}% ของ Black Swan ทั้งหมด)")
            print(f"  False Positives: {false_positives} ({false_positives/detected*100:.2f}% ของการตรวจจับทั้งหมด)")
        
        print("\n" + "="*80)


# โมเดลทางสถิติสำหรับตรวจจับความผิดปกติ

class EWMDetector:
    """
    ตรวจจับความผิดปกติโดยใช้ Exponentially Weighted Moving Average (EWM)
    
    EWM ให้น้ำหนักกับข้อมูลปัจจุบันมากกว่าข้อมูลในอดีต ซึ่งเหมาะสำหรับ
    การตรวจจับความผิดปกติที่เกิดขึ้นอย่างรวดเร็ว
    """
    
    def __init__(self, span=30, alpha=None, adjust=True):
        """
        กำหนดค่าเริ่มต้นสำหรับ EWM Detector
        
        Args:
            span (int): ค่า span สำหรับ EWM (n/(1-alpha) โดยที่ n คือจำนวนข้อมูล)
            alpha (float): ค่า smoothing factor (ถ้าเป็น None จะคำนวณจาก span)
            adjust (bool): ปรับแก้ค่าเริ่มต้นหรือไม่
        """
        self.span = span
        self.alpha = alpha
        self.adjust = adjust
        self.mean = None
        self.std = None
        self.threshold = None
    
    def fit(self, X, threshold_multiplier=3.0):
        """
        ฝึกโมเดลด้วยข้อมูลปกติ
        
        Args:
            X (array): ข้อมูลอนุกรมเวลา 1 มิติที่เป็นข้อมูลปกติ
            threshold_multiplier (float): ตัวคูณสำหรับการกำหนดค่าขีดแบ่ง
        """
        # คำนวณค่าเฉลี่ยเคลื่อนที่แบบถ่วงน้ำหนักเอกซ์โพเนนเชียล
        X_series = pd.Series(X)
        self.mean = X_series.ewm(span=self.span, alpha=self.alpha, adjust=self.adjust).mean().iloc[-1]
        
        # คำนวณส่วนเบี่ยงเบนมาตรฐานแบบถ่วงน้ำหนักเอกซ์โพเนนเชียล
        self.std = X_series.ewm(span=self.span, alpha=self.alpha, adjust=self.adjust).std().iloc[-1]
        
        # กำหนดค่าขีดแบ่ง
        self.threshold = self.mean + threshold_multiplier * self.std
    
    def predict(self, X):
        """
        ทำนายคะแนนความผิดปกติสำหรับข้อมูลที่ให้มา
        
        Args:
            X (array): ข้อมูลอนุกรมเวลา 1 มิติที่ต้องการทำนาย
        
        Returns:
            array: คะแนนความผิดปกติสำหรับแต่ละตัวอย่าง
        """
        # คำนวณค่า Z-score
        z_scores = (X - self.mean) / self.std
        
        # แปลงเป็นคะแนนความผิดปกติ (ใช้ค่าสัมบูรณ์)
        anomaly_scores = np.abs(z_scores)
        
        return anomaly_scores


class IQRDetector:
    """
    ตรวจจับความผิดปกติโดยใช้ Interquartile Range (IQR)
    
    IQR เป็นวิธีที่ทนทานต่อค่าผิดปกติ (robust) โดยพิจารณาช่วงระหว่างควอร์ไทล์ที่ 1 และ 3
    """
    
    def __init__(self, k=1.5):
        """
        กำหนดค่าเริ่มต้นสำหรับ IQR Detector
        
        Args:
            k (float): ตัวคูณสำหรับ IQR (ค่ามาตรฐานคือ 1.5)
        """
        self.k = k
        self.q1 = None
        self.q3 = None
        self.iqr = None
        self.median = None
    
    def fit(self, X, threshold_multiplier=1.0):
        """
        ฝึกโมเดลด้วยข้อมูลปกติ
        
        Args:
            X (array): ข้อมูลอนุกรมเวลา 1 มิติที่เป็นข้อมูลปกติ
            threshold_multiplier (float): ตัวคูณเพิ่มเติมสำหรับการกำหนดค่าขีดแบ่ง
        """
        # คำนวณค่าสถิติ
        self.q1 = np.percentile(X, 25)
        self.q3 = np.percentile(X, 75)
        self.iqr = self.q3 - self.q1
        self.median = np.median(X)
        
        # ไม่ได้ใช้ threshold_multiplier ในที่นี้เนื่องจาก IQR มีค่า k อยู่แล้ว
        # แต่สามารถปรับ k ได้หากต้องการ
        self.k = self.k * threshold_multiplier
    
    def predict(self, X):
        """
        ทำนายคะแนนความผิดปกติสำหรับข้อมูลที่ให้มา
        
        Args:
            X (array): ข้อมูลอนุกรมเวลา 1 มิติที่ต้องการทำนาย
        
        Returns:
            array: คะแนนความผิดปกติสำหรับแต่ละตัวอย่าง
        """
        # คำนวณค่าขีดแบ่งบนและล่าง
        lower_bound = self.q1 - self.k * self.iqr
        upper_bound = self.q3 + self.k * self.iqr
        
        # คำนวณระยะห่างจากค่าขีดแบ่ง
        dist_lower = np.maximum(0, lower_bound - X)
        dist_upper = np.maximum(0, X - upper_bound)
        
        # คะแนนความผิดปกติคือระยะห่างจากค่าขีดแบ่งที่มากกว่า
        anomaly_scores = np.maximum(dist_lower, dist_upper)
        
        # ปรับให้เป็นสัดส่วนกับ IQR เพื่อให้เปรียบเทียบกับโมเดลอื่นได้
        anomaly_scores = anomaly_scores / self.iqr
        
        return anomaly_scores


class ZScoreDetector:
    """
    ตรวจจับความผิดปกติโดยใช้ Z-Score
    
    Z-Score วัดว่าข้อมูลแต่ละตัวอยู่ห่างจากค่าเฉลี่ยกี่ส่วนเบี่ยงเบนมาตรฐาน
    """
    
    def __init__(self, window_size=None):
        """
        กำหนดค่าเริ่มต้นสำหรับ Z-Score Detector
        
        Args:
            window_size (int): ขนาดของหน้าต่างสำหรับการคำนวณแบบเลื่อน (ถ้าเป็น None จะใช้ทั้งชุดข้อมูล)
        """
        self.window_size = window_size
        self.mean = None
        self.std = None
    
    def fit(self, X, threshold_multiplier=3.0):
        """
        ฝึกโมเดลด้วยข้อมูลปกติ
        
        Args:
            X (array): ข้อมูลอนุกรมเวลา 1 มิติที่เป็นข้อมูลปกติ
            threshold_multiplier (float): ตัวคูณสำหรับการกำหนดค่าขีดแบ่ง
        """
        # คำนวณค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน
        self.mean = np.mean(X)
        self.std = np.std(X)
        self.threshold_multiplier = threshold_multiplier
    
    def predict(self, X):
        """
        ทำนายคะแนนความผิดปกติสำหรับข้อมูลที่ให้มา
        
        Args:
            X (array): ข้อมูลอนุกรมเวลา 1 มิติที่ต้องการทำนาย
        
        Returns:
            array: คะแนนความผิดปกติสำหรับแต่ละตัวอย่าง
        """
        if self.window_size is None:
            # คำนวณ Z-score โดยใช้ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานจากการฝึก
            z_scores = np.abs((X - self.mean) / self.std)
        else:
            # ใช้วิธีหน้าต่างเลื่อน
            z_scores = np.zeros_like(X)
            
            for i in range(len(X)):
                # กำหนดช่วงของหน้าต่าง
                start = max(0, i - self.window_size)
                window_data = X[start:i+1]
                
                if len(window_data) > 1:  # ต้องมีข้อมูลอย่างน้อย 2 ตัว
                    window_mean = np.mean(window_data)
                    window_std = np.std(window_data)
                    
                    # ป้องกันการหารด้วยศูนย์
                    if window_std > 0:
                        z_scores[i] = abs((X[i] - window_mean) / window_std)
                    else:
                        z_scores[i] = 0
                else:
                    z_scores[i] = 0
        
        return z_scores


def main():
    """
    ฟังก์ชันหลักสำหรับสาธิตการใช้งาน BlackSwanDetector
    """
    # กำหนดพารามิเตอร์
    tickers = ['SPY', 'QQQ', 'DIA', 'IWM']  # เพิ่มหุ้นหลายตัว
    start_date = '2006-01-01'
    end_date = '2022-12-31'
    
    # กำหนดช่วงเวลา Black Swan
    black_swan_periods = {
        '2008-crisis': ('2008-09-01', '2008-10-31'),  # วิกฤติการเงิน 2008
        'covid-crash': ('2020-02-20', '2020-03-23')   # ตลาดร่วงจาก COVID-19
    }
    
    # สร้างระบบตรวจจับ Black Swan
    detector = BlackSwanDetector(
        seq_length=30,
        latent_dim=10,
        batch_size=32,
        epochs=50
    )
    
    # โหลดข้อมูลหุ้น
    detector.load_data(tickers, start_date, end_date, black_swan_periods)
    
    # สกัดคุณลักษณะ
    detector.feature_engineering(tickers)
    
    # ดำเนินการตรวจจับ Black Swan แบบหุ้นเดี่ยว (single-asset)
    for ticker in tickers:
        detector.run_pipeline(
            ticker, 
            test_ratio=0.2,
            val_ratio=0.1,
            threshold_multiplier=3.0
        )
    
    # ดำเนินการตรวจจับ Black Swan แบบหุ้นหลายตัว (cross-asset)
    detector.run_cross_asset_pipeline(tickers, combined_model_name='market_combined')

if __name__ == "__main__":
    main()