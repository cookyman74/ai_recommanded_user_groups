# 유저 프로필 기반 그룹 분석 및 호감도 측정 시스템

## 프로젝트 개요

이 시스템은 **유저 프로필** 데이터를 기반으로 **K-Means 클러스터링**을 통해 유저들을 여러 그룹으로 분류하고, **유저 간의 호감도(유사도)**를 계산하여 각 그룹 내에서 유저들이 얼마나 유사한지(호감도)를 측정합니다. 또한, **OpenAI API**를 사용하여 각 그룹의 특징을 분석하고, 그룹 내 유저들이 서로에게 끌릴만한 **질문 리스트**를 자동으로 생성하는 기능을 포함합니다.

이를 통해, 유저들이 상호작용을 촉진할 수 있는 유용한 도구를 제공합니다.



## 주요 기능

- **데이터 로드 및 전처리**: 엑셀 파일에서 참가자 데이터를 로드하고 전처리합니다.
- **K-means 클러스터링**: 여성 참가자들을 기반으로 초기 그룹을 형성합니다.
- **유사도 기반 매칭**: 남성 참가자들을 여성 그룹에 매칭합니다.
- **그룹 밸런싱**: 각 그룹의 크기와 성비를 조정합니다.
- **AI 분석**: OpenAI API를 사용하여 각 그룹의 특징을 분석하고 추천 질문을 생성합니다.
- **결과 시각화**: Streamlit을 사용하여 결과를 시각적으로 표현합니다.


## 주요 라이브러리 및 API

- **Streamlit**: 웹 인터페이스로 사용자와 상호작용할 수 있는 화면을 제공하는 파이썬 기반의 웹 프레임워크입니다.
- **Pandas**: 유저 데이터를 엑셀 파일로부터 읽어오고, 데이터프레임으로 처리하는 데 사용됩니다.
- **Scikit-learn (K-Means)**: 유저들을 그룹으로 나누기 위해 **K-Means 클러스터링** 알고리즘을 사용합니다.
- **Cosine Similarity**: 각 유저의 프로필 데이터를 기반으로 유사도를 측정하는 데 사용됩니다.
- **OpenAI API**: GPT-4 모델을 사용하여 그룹 분석과 질문 리스트 생성을 자동화합니다.



## 설치 및 실행 방법
### 1. 저장소를 클론합니다 
- 저장소를 클론한후 프로젝트 디렉토리로 이동합니다.

```bash
git clone https://github.com/yourusername/meeting-shuffling-system.git
```

### 2. 필요한 패키지를 설치합니다.
   - 다음과 같이 설치 합니다.

```bash
pip install requirements.txt
```

### 3. `.env` 파일을 생성하고 OpenAI API 키를 추가합니다

```text
OPENAI_API_KEY=your_api_key_here
```

### 4. 실행 방법
   - 터미널에서 다음 명령어를 실행하여 Streamlit 서버를 시작합니다.
```bash
streamlit run app.py
```


## 사용 방법 
### 1. 설정
- `config.yml` 파일에서 다음 설정을 조정할 수 있습니다.
  - 텍스트 컬럼 
  - 태그 옵션 
  - 기타 매개변수
### 2. streamlit을 실행시킵니다.
### 3. 웹 브라우저에서 표시된 URL로 이동합니다. (http://localhost:8051)
### 4. 엑셀 파일을 업로드하고 결과를 확인합니다.
- 웹 인터페이스에서 엑셀 파일을 업로드하여 유저 데이터를 입력할 수 있습니다
- 엑셀 파일에는 **유저 프로필** 데이터가 포함되어 있어야 하며, 필수 컬럼은 다음과 같습니다
    - `User ID`, `Preference` (취향), `MBTI`, `Job` (직업), `Hobby` (취미), `Age` (나이), `Gender` (성별), `Height` (키), `Weight` (몸무게), `Ideal Type` (이상형)
    - 엑셀파일 예제



## 결과 출력
결과는 다음 형식으로 출력됩니다:
- 셔플링 계획서 제목 (날짜 포함)
- 각 그룹별 정보:
- 그룹 번호 및 평균 유사도
- 데이터 분석 내용 (성비, 직업, MBTI, 평균 나이)
- 그룹의 특징
- 추천 질문 리스트
- 전체 그룹 구성 요약
- 배정되지 않은 사용자 정보 (있는 경우)



## 주의사항

- OpenAI API 사용량에 주의하세요.
- 개인정보 보호를 위해 실제 참가자 데이터를 안전하게 관리하세요.


---

## 기능 요구 사항.

### 1. 모임 신청 인원과 성별 비율의 동적 처리
- 전체 신청 인원은 **동적으로 변화**하며, 여성과 남성의 비율도 달라질 수 있습니다.
- 신청 인원에 따라 **여성 비율**은 30% 정도를 가정하지만, 상황에 따라 유동적으로 변화할 수 있습니다.

### 2. 그룹 구성 조건
- **6명에서 8명 사이의 인원**으로 구성된 그룹이 생성됩니다.
- **각 그룹에는 최소 3명의 여성이 포함**되어야 하며, 여성이 3명 이상일 경우도 허용됩니다.
- 각 그룹은 **유사도 기반**으로 구성되며, 남성과 여성이 혼합된 형태입니다.

### 3. 여성 인원 배정 및 재조정
- **K-means 클러스터링**을 사용해 유사도가 높은 여성들끼리 그룹을 나눕니다.
- 클러스터링 결과 **3명 미만의 여성이 포함된 그룹**이 발생하면, 남은 여성 중 일부를 해당 그룹에 추가합니다.
- **모든 여성 인원이 반드시 그룹에 포함**되어야 하며, 탈락 인원이 발생하지 않도록 합니다.
- 여성 인원 재조정 시 **반복적으로 여성을 추가**하여 최소 3명을 보장하며, 필요한 경우 다른 그룹에서 인원을 재배치합니다.

### 4. 남성 인원 배정
- 각 여성 그룹에 **최대 5명까지 남성**을 배정하여 **최대 8명까지 그룹 인원**을 맞춥니다.
- 남성 배정은 **유사도 기반**으로 이루어지며, 그룹의 **유사도 평균이 최대화되도록** 합니다.
- **남은 남성 인원을 균등하게 배치**하여 모든 그룹에 빈 공간이 없도록 합니다.

### 5. 자유 텍스트 기반 태그 생성 및 유사도 측정
- 사용자가 제출한 **자유 텍스트 항목(예: 나만의 맛집)**에 대해 **OpenAI API**를 활용하여 **주제 태그**를 생성합니다.
- **`config.yml` 파일**에 자유 텍스트로 작성된 특정 컬럼과 해당 컬럼의 태그 리스트를 정의합니다.
- **태그화된 항목을 K-means 클러스터링 및 유사도 계산에 반영**합니다.
- **TF-IDF 기반 유사도 측정**을 통해 텍스트 기반 유사도도 계산합니다.

### 6. 유사도 계산 및 출력
- *`cosine_similarity`*를 사용해 각 그룹의 유사도를 계산합니다.
- 그룹별로 **직업, MBTI, 취미, 나이 등의 특징을 요약**해 출력합니다.
- 자유 텍스트 태그와 다른 특징 간의 유사도를 평가하여 의미 있는 그룹 구성이 이루어집니다.

### 7. OpenAI API를 통한 AI 질문 리스트 생성
- 각 그룹의 특징을 바탕으로 **OpenAI API**를 활용해 호감을 가질 만한 **질문 리스트**를 생성합니다.
- 질문 리스트는 **사용자 간 상호작용을 촉진**합니다.

### 8. 균등한 배정 및 인덱스 오류 방지
- **배열 인덱스가 범위를 벗어나지 않도록 검사**합니다. 남은 남성 인원이 부족할 경우 유효한 인덱스만 사용합니다.
- **모든 여성과 남성을 최대한 고르게 배정**하여 불균형한 배치를 방지합니다.
- 인덱스 범위를 벗어나는 경우를 방지하기 위해 **유효성 검사를 수행**합니다. 

### 9. 모든 인원 배정 후 최종 조정
- 각 그룹의 총 인원이 **최소 6명 이상**이 되도록 합니다.
- 남은 인원은 **최종적으로 재배정**하여 모든 그룹의 최소 인원을 맞춥니다.

### 10. 탈락 모임 처리
- 만약 최종 인원이 **6명 이하로 구성된 그룹이 발생**하면 해당 그룹을 탈락 모임으로 처리합니다.
- 탈락 모임이 발생하지 않도록 최소화 합니다.

### 11. 결과 출력 및 AI 호스트 역할 수행
- 모든 그룹의 **구성원을 출력**하고, 각 그룹의 요약 정보를 제공합니다.
- **AI 호스트**가 설문을 주기적으로 실행하여 사용자 간의 상호작용을 촉진합니다.