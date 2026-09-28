import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 0) 전처리 및 변환 함수 정의 (필요시 수정)
def load(f):
    # CSV 파일 로드 (예시: pandas 사용)
    return pd.read_csv(f).values

def preprocess(sig):
    # 시그널 전처리 로직 (필터링 등)
    return sig

def make_windows(sig):
    # 시그널을 일정한 크기의 윈도우 단위로 슬라이싱
    window_size = 128
    windows = [sig[i:i + window_size] for i in range(0, len(sig) - window_size, window_size)]
    if not windows:  # 데이터가 너무 짧을 경우 대비
        windows = [sig]
    return windows

def minmax(windows):
    # Min-Max 정규화 등
    return windows

def to_cwt(w):
    # 윈도우 신호를 CWT(연속 웨이블릿 변환) 2차원 이미지 형태로 변환
    return np.zeros((64, 64), dtype=np.float32)

def label_of(f):
    # 파일 경로에서 특정 키워드(피험자 이름 등)를 탐색하여 라벨 추출
    parts = f.split('/')
    for part in parts:
        if part in ['A', 'B', 'C', 'D', 'E']:
            return part
    return 'unknown'

# 1) 파일 탐색 및 라벨링
path = '/Users/yunjongsu/Desktop/Deep_Project/semg-auth/**/*.csv'
files = glob.glob(path, recursive=True)

print(f"찾은 파일 개수: {len(files)}")
if len(files) == 0:
    raise ValueError("지정한 경로에서 CSV 파일을 찾지 못했습니다.")

labels = [label_of(f) for f in files]

# 2) 시행 단위 데이터 분할 (순서 유지)
# 라벨 비율을 유지(층화 추출)하여 학습용과 테스트용 파일 경로 분할
tr_files, te_files = train_test_split(
    files, test_size=0.2, stratify=labels, random_state=42
)
print("데이터 분할 성공! (Train:", len(tr_files), ", Test:", len(te_files), ")")

# 3) 각 집합별 윈도우 생성 및 변환 (build 함수)
def build(flist):
    X, y = [], []
    for f in flist:
        sig = preprocess(load(f))
        for w in minmax(make_windows(sig)):
            X.append(to_cwt(w))
            y.append(label_of(f))
    return np.stack(X), np.array(y)

Xtr, ytr = build(tr_files)
Xte, yte = build(te_files)

# 교차 엔트로피 손실 함수(CrossEntropyLoss) 사용을 위해 문자열 라벨을 정수 인덱스로 매핑
unique_labels = np.unique(ytr)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
ytr_encoded = np.array([label_to_idx[l] for l in ytr])

# 4) PyTorch 데이터로더 및 CNN 모델 설정
# 딥러닝 입력을 위해 NumPy 배열을 PyTorch 텐서로 변환 (채널 차원 추가: N, 1, H, W)
Xtr_tensor = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)
ytr_tensor = torch.tensor(ytr_encoded, dtype=torch.long)

train_dataset = TensorDataset(Xtr_tensor, ytr_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 특징 추출을 위한 합성곱(Conv) 및 풀링(Pooling) 레이어 정의
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes) # 최종 분류기

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = SimpleCNN(num_classes=len(unique_labels))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5) 최소 5에폭 이상 학습 및 Loss 출력 로그 기록
num_epochs = 5
print("\n=== 모델 학습 시작 (최소 5에폭) ===")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # 미니배치 단위로 학습 진행 (순전파 -> 손실 계산 -> 역전파 -> 가중치 최적화)
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {epoch_loss:.4f}")

print("<학습 완료>")