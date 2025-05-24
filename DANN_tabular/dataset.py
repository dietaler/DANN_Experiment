import pandas as pd
import numpy as np
import torch
import re
import multiprocessing
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import params

class CICIDSDataset(Dataset):
    def __init__(self, df):
        self.X = df.drop(columns=['Label']).values  # 特徵
        self.y = df['Label'].values  # 標籤

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(0), torch.tensor(self.y[idx], dtype=torch.long)

def load_data(df, batch_size=params.batch_size):
    # 標準化 & Label Encoding
    df.iloc[:, :-1] = StandardScaler().fit_transform(df.iloc[:, :-1])
    print(f"標準化後的數據形狀: {df.shape}")
    encoder = LabelEncoder()
    fixed_classes = np.array(['Benign', 'DoS', 'Port-scan', 'Botnets', 'Web-attacks', 'Brute-force'])
    encoder.classes_ = fixed_classes
    df['Label'] = encoder.transform(df['Label'])
    
    # 切分數據集
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['Label'])

    # 創建 Dataset & DataLoader
    train_loader = DataLoader(CICIDSDataset(train_df), batch_size=batch_size, shuffle=True, pin_memory=False)
    test_loader = DataLoader(CICIDSDataset(test_df), batch_size=batch_size, shuffle=False, pin_memory=False)

    return train_loader, test_loader

def change_label(df):
    # cicids2018 map to cicids2017
    cicids2018to2017 = {
        'Benign': 'Benign',
        'DDoS attacks-LOIC-HTTP': 'DDoS',
        'DDOS attack-LOIC-UDP': 'DDoS',
        'DDOS attack-HOIC': 'DDoS',
        'DoS attacks-GoldenEye': 'DoS GoldenEye',
        'DoS attacks-Slowloris': 'DoS slowloris',
        'DoS attacks-Hulk': 'DoS Hulk',
        'DoS attacks-SlowHTTPTest': 'DoS Slowhttptest',
        'Bot': 'Bot',
        'Infilteration': 'Infiltration',
        'SSH-Bruteforce': 'SSH-Patator',
        'FTP-BruteForce': 'FTP-Patator',
        'Brute Force -Web': 'Web Attack – Brute Force',
        'Brute Force -XSS': 'Web Attack – XSS',
        'SQL Injection': 'Web Attack – Sql Injection',
    }

    # cicids2017 map to major 6 classes
    cicids2017to6 = {
        'Benign': 'Benign',  # 正常流量
        'DDoS': 'DoS',
        'DoS Hulk': 'DoS',
        'DoS GoldenEye': 'DoS',
        'DoS slowloris': 'DoS',
        'DoS Slowhttptest': 'DoS',
        'PortScan': 'Port-scan',
        'FTP-Patator': 'Brute-force',
        'SSH-Patator': 'Brute-force',
        'Web Attack – Brute Force': 'Web-attacks',
        'Web Attack – XSS': 'Web-attacks',
        'Web Attack – Sql Injection': 'Web-attacks',
        'Bot': 'Botnets',
        'Infiltration': 'Botnets',
        'Heartbleed': 'Botnets'
    }

    def clean_label(label):
        if 'Web Attack' in label:
            # 去除亂碼並保留原始類型
            return re.sub(r'Ã.*Â\x96', '–', label)  # 將亂碼替換為正常符號 "–"
        return label
    
    # 處理label的亂碼問題
    df['Label'] = df['Label'].apply(clean_label)
    # 將所有samples分到六大類
    df['Label'] = df['Label'].map(cicids2018to2017).fillna(df['Label'])
    df['Label'] = df['Label'].map(cicids2017to6).fillna(df['Label'])
    
    return df

def down_sampling(df, max_samples=15000):
    """
    下採樣所有類別，使每個類別的樣本數不超過 max_samples。
    
    - df: pandas DataFrame，包含 'Label' 欄位。
    - max_samples: 每個類別最多保留的樣本數。
    
    回傳：
    - 下採樣後的 DataFrame。
    """
    print(f"下採樣前: {df.shape}")

    # 確保 max_samples 不超過某類別的總樣本數
    df_downsampled = df.groupby("Label").apply(lambda x: x.sample(n=min(len(x), max_samples), random_state=42))
    
    # 重設索引
    df_downsampled = df_downsampled.reset_index(drop=True)

    print(f"下採樣後: {df_downsampled.shape}")
    return df_downsampled