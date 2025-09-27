import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# === Load CSV files ===
df_grasp = pd.read_csv("pcr_data_with_grasping.csv")
df_nograsp = pd.read_csv("pcr_data_without_grasping.csv")

# === Feature Engineering ===
def build_features(df):
    base_cols = [
        'Finger1_Proximal', 'Finger1_Distal',
        'Finger2_Proximal', 'Finger2_Distal',
        'Finger3_Proximal', 'Finger3_Distal',
        'Finger1_Proximal_Voltage', 'Finger1_Distal_Voltage',
        'Finger2_Proximal_Voltage', 'Finger2_Distal_Voltage',
        'Finger3_Proximal_Voltage', 'Finger3_Distal_Voltage',
        'Force'
    ]
    df = df.copy()
    for col in base_cols:
        df[f"{col}_delta"] = df[col].diff().fillna(0)
        df[f"{col}_std"] = df[col].rolling(window=5, min_periods=1).std().fillna(0)
    df["Delta_Force"] = df["Force"].diff().fillna(0)
    feature_cols = base_cols + [f"{col}_delta" for col in base_cols] + \
                   [f"{col}_std" for col in base_cols] + ["Delta_Force"]
    return df[feature_cols].values.astype(np.float32)

Xg = build_features(df_grasp)
Xn = build_features(df_nograsp)
yg = np.ones(len(Xg))
yn = np.zeros(len(Xn))

# === Combine and normalize ===
X = np.vstack([Xg, Xn])
y = np.hstack([yg, yn])
mean, std = X.mean(axis=0), X.std(axis=0) + 1e-6
X = (X - mean) / std

# === Create sequences ===
seq_len = 20
X_seq, y_seq = [], []
for i in range(len(X) - seq_len):
    X_seq.append(X[i:i+seq_len])
    y_seq.append(y[i+seq_len])
X_seq = torch.tensor(np.stack(X_seq), dtype=torch.float32)
y_seq = torch.tensor(y_seq, dtype=torch.float32).unsqueeze(1)

# === Train/test split ===
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# === Define Model ===
class RichLSTMSlipDetector(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

# === Training ===
model = RichLSTMSlipDetector()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(20):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val)
    print(f"Epoch {epoch+1}/20 - Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# === Save Model ===
torch.save(model.state_dict(), "rich_lstm_slip_model.pth")

# === Evaluate ===
model.eval()
with torch.no_grad():
    y_probs = model(X_val).numpy().flatten()
    y_true = y_val.numpy().flatten()

# === ROC and Thresholding ===
fpr, tpr, thresholds = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)
opt_idx = np.argmax(tpr - fpr)
opt_threshold = thresholds[opt_idx]

y_pred_binary = (y_probs >= opt_threshold).astype(int)

# === Metrics ===
acc = accuracy_score(y_true, y_pred_binary)
prec = precision_score(y_true, y_pred_binary)
rec = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
conf_matrix = confusion_matrix(y_true, y_pred_binary)

# === Print Evaluation ===
print("\n--- Evaluation Metrics ---")
print(f"AUC: {roc_auc:.3f}")
print(f"Optimal Threshold: {opt_threshold:.3f}")
print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1 Score: {f1:.3f}")
print("Confusion Matrix:\n", conf_matrix)

# === Plot ROC Curve ===
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.scatter(fpr[opt_idx], tpr[opt_idx], color='red', label='Optimal Threshold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Slip Detection')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Overlay Prediction Curve for Grasping ===
# Just as a visual check using grasping sequence
time_g = df_grasp['Time(ms)'].values[seq_len:] / 1000.0
force_g = df_grasp['Force'].values[seq_len:]
Xg_norm = (Xg - mean) / std
Xg_seq = torch.tensor(np.stack([Xg_norm[i:i+seq_len] for i in range(len(Xg_norm) - seq_len)]), dtype=torch.float32)

with torch.no_grad():
    slip_grasp = model(Xg_seq).numpy().flatten()

plt.figure(figsize=(14, 6))
plt.plot(time_g, slip_grasp, 'r-', label='Slip Probability (Grasp)')
plt.plot(time_g, force_g[:len(slip_grasp)] / max(force_g), 'k--', label='Norm. Force (Grasp)')
plt.xlabel("Time (s)")
plt.ylabel("Value")
plt.title("Slip Detection on Grasping Data (Post-Training)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
