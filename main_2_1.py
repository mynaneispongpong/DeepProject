import glob
from sklearn.model_selection import train_test_split

# 1) 지정된 절대 경로 하위의 모든 CSV 파일을 재귀적으로 탐색하여 목록화
path = '/Users/yunjongsu/Desktop/Deep_Project/semg-auth/**/*.csv'
files = glob.glob(path, recursive=True)

print(f"찾은 파일 개수: {len(files)}")
print(f"파일 목록 일부: {files[:5]}")

# 2) 파일 경로를 분석하여 폴더명을 기반으로 해당 데이터의 라벨을 추출하는 함수
def label_of(f):
    parts = f.split('/')

    for part in parts:
        if part in ['A', 'B', 'C']:
            return part
    return 'unknown'

labels = [label_of(f) for f in files]

# 3) 추출된 라벨의 비율을 유지하여 학습용과 테스트용 파일 리스트로 분할
if len(files) > 0:
    tr_files, te_files = train_test_split(
        files, test_size=0.2, stratify=labels, random_state=42
    )
    print("데이터 분할 성공!")
else:
    print("여전히 파일을 찾지 못했습니다. 경로를 확인해주세요.")