import numpy as np, glob, os

# 1. 지정한 경로 하위의 모든 .csv 파일을 재귀적으로 탐색하여 정렬된 목록으로 반환
files = sorted(glob.glob('semg-auth/data/**/*.csv', recursive=True))
print('파일 개수:', len(files))
print('예시 경로:', files[0])

# 2. 첫 번째 CSV 파일을 NumPy 배열로 로드
x = np.loadtxt(files[0], delimiter=',', skiprows=1) # 형식에 맞게 변경
print('배열 모양:', x.shape) # 예: (3000, 2)
print('값 범위:', x.min(), '~', x.max())

# 3. 파일 경로에서 부모 디렉터리 이름을 추출하여 피험자별 파일 개수 집계
from collections import Counter
subs = [os.path.basename(os.path.dirname(f))
        for f in files]