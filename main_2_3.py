import glob
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 전처리 및 변환 함수 정의
def load(f):
    # CSV 파일 로드
    return pd.read_csv(f).values

def preprocess(sig):
    # 시그널 전처리 로직 (필터링 등)
    return sig

def make_windows(sig):
    # 시그널을 일정한 크기의 윈도우 단위로 슬라이싱
    window_size = 128
    windows = [sig[i:i + window_size] for i in range(0, len(sig) - window_size, window_size)]
    if not windows:
        windows = [sig]
    return windows

def minmax(windows):
    # Min-Max 정규화
    return windows

def to_cwt(w):
    # 윈도우 신호를 CWT 2차원 이미지 형태로 변환
    return np.zeros((64, 64), dtype=np.float32)

def label_of(f):
    # 파일 경로에서 특정 키워드를 탐색하여 라벨 추출
    parts = f.split('/')
    for part in parts:
        if part in ['A', 'B', 'C', 'D', 'E']:
            return part
    return 'unknown'

# 파일 탐색 및 라벨링
path = '/Users/yunjongsu/Desktop/Deep_Project/semg-auth/**/*.csv'
files = glob.glob(path, recursive=True)

if len(files) == 0:
    raise ValueError("지정한 경로에서 CSV 파일을 찾지 못했습니다.")

labels = [label_of(f) for f in files]

# 시행 단위 데이터 분할 (순서 유지)
# 라벨 비율을 유지(층화 추출)하여 학습용과 테스트용 파일 경로 분할
tr_files, te_files = train_test_split(
    files, test_size=0.2, stratify=labels, random_state=42
)

# 각 집합별 윈도우 생성 및 변환 (build 함수)
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

# 데이터셋 요약 및 클래스별 분포 표 출력
print("데이터셋 요약")
print(f"• 전체 파일(시행) 수: {len(files)} 개")
print(f"  - 훈련용 파일 수: {len(tr_files)} 개")
print(f"  - 테스트용 파일 수: {len(te_files)} 개")
print(f"• 윈도우 변환 후 샘플 수:")
print(f"  - 훈련 윈도우 데이터(Xtr): {Xtr.shape[0]} 개")
print(f"  - 테스트 윈도우 데이터(Xte): {Xte.shape[0]} 개")

tr_counts = Counter(ytr)
te_counts = Counter(yte)

# 학습 및 테스트 데이터의 클래스별 샘플 수를 표 형태로 구성
summary_df = pd.DataFrame({
    'Train 샘플 수': pd.Series(tr_counts),
    'Test 샘플 수': pd.Series(te_counts)
}).fillna(0).astype(int)

summary_df['총 합계'] = summary_df['Train 샘플 수'] + summary_df['Test 샘플 수']
summary_df = summary_df.sort_index()

print("\n[클래스별 데이터 분포 표]")
print(summary_df.to_markdown())
print("="*50 + "\n")

# 교차 엔트로피 손실 함수(CrossEntropyLoss) 사용을 위해 문자열 라벨을 정수 인덱스로 매핑
unique_labels = np.unique(ytr)
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
ytr_encoded = np.array([label_to_idx[l] for l in ytr])

# 5) PyTorch 데이터로더 및 CNN 모델 설정
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