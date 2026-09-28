import matplotlib.pyplot as plt
import numpy as np
import glob
import os

# 1. 파일 목록 가져오기 및 서로 다른 피험자 3명의 데이터 경로 선택하기
files = sorted(glob.glob('semg-auth/data/**/*.csv', recursive=True))

selected_files = []
seen_subs = set()
for f in files:
    sub_name = os.path.basename(os.path.dirname(f))
    if sub_name not in seen_subs:
        seen_subs.add(sub_name)
        selected_files.append(f)
    if len(selected_files) >= 3:  # 피험자 3명 추출
        break

# 2. 피험자 3명의 2개 채널 데이터를 한 화면에 비교하기 위한 서브플롯 생성
fig, ax = plt.subplots(3, 2, figsize=(10, 8), sharex=True)

for i, file_path in enumerate(selected_files):
    sub_name = os.path.basename(os.path.dirname(file_path))

    x = np.loadtxt(file_path, delimiter=',', skiprows=1)

    # 데이터 형태를 구조로 통일
    if x.shape[0] < x.shape[1]:
        x = x.T

    t = np.arange(len(x)) / 1000.0  # 샘플링 레이트 1000 Hz 기준 초단위 시간 축 생성

    # 각 피험자별 2개 채널 플롯 그리기
    for ch in range(2):
        ax[i, ch].plot(t, x[:, ch], lw=0.6)

        # y축 레이블 설정
        if ch == 0:
            ax[i, ch].set_ylabel(f'{sub_name}\nch{ch + 1}')
        else:
            ax[i, ch].set_ylabel(f'ch{ch + 1}')

        # 첫 번째 행에만 상단 제목 표시
        if i == 0:
            ax[i, ch].set_title(f'Channel {ch + 1}')

        # 1초, 2초 시점에 동작 구간 경계선 표시
        ax[i, ch].axvline(1.0, color='r', ls='--')
        ax[i, ch].axvline(2.0, color='r', ls='--')

# 3. 마지막 행의 x축에만 time 레이블 표시 및 그래프 파일 저장
ax[2, 0].set_xlabel('time (s)')
ax[2, 1].set_xlabel('time (s)')

plt.tight_layout()
plt.savefig('피험자 3명 그래프', dpi=120)
print("피험자 3명 통합 비교 그래프 저장 완료: 피험자 3명 그래프")