import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, iirnotch, filtfilt

# macOS 한글 폰트 및 마이너스 기호 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 샘플링 주파수
FS = 1000

def preprocess(x, fs=FS):
    # 1) 60 Hz 전원선 노이즈 제거를 위한 노치 필터 적용
    bn, an = iirnotch(60, 30, fs)
    x = filtfilt(bn, an, x, axis=0)

    # 2) 20~499 Hz 생체 신호 대역만 통과시키는 4차 버터워스 대역통과 필터 적용
    b, a = butter(4, [20 / (fs / 2), 499 / (fs / 2)], btype='band')
    return filtfilt(b, a, x, axis=0)


# 테스트를 위한 가상 신호 생성
np.random.seed(42)
t = np.linspace(0, 2, FS * 2)
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 60 * t) + 0.2 * np.random.randn(len(t))
x = x[:, np.newaxis]

# 정의한 함수로 sEMG 신호 전처리 실행
xf = preprocess(x)

# 필터 효과 확인을 위한 주파수 스펙트럼 계산
N = len(x)
freqs = np.fft.rfftfreq(N, 1 / FS)
fft_orig = np.abs(np.fft.rfft(x[:, 0])) / N
fft_filtered = np.abs(np.fft.rfft(xf[:, 0])) / N

# 필터 전후 주파수 스펙트럼 비교 그래프 시각화 및 저장
plt.figure(figsize=(10, 5))
plt.plot(freqs, fft_orig, label='필터 전 (Original)', color='orange', alpha=0.7)
plt.plot(freqs, fft_filtered, label='필터 후 (Filtered)', color='blue', alpha=0.8)

plt.title('필터 전후 주파수 스펙트럼 비교')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 150)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

# 이미지 파일로 저장
plt.savefig('filter_check.png', dpi=300)
plt.show()

print('전:', x.std(), '후:', xf.std())