# Python 3.10 슬림 이미지 사용
FROM python:3.12-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필요 라이브러리 복사
COPY requirements.txt ./

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Streamlit 애플리케이션 실행
CMD ["streamlit", "run", "app.py"]
