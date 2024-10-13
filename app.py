import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import openai

# OpenAI API 설정
openai.api_key = 'your-openai-api-key'

# 엑셀 파일 업로드
st.title("유저 프로필 기반 그룹 분석 및 호감도 측정 시스템")
uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요.", type="xlsx")

if uploaded_file is not None:
    # 엑셀 파일 로드
    df = pd.read_excel(uploaded_file)
    st.write("업로드된 유저 데이터:")
    st.write(df)

    # 그룹 개수 설정 (관리자가 설정한 그룹 개수)
    num_groups = st.number_input("원하는 그룹 개수를 입력하세요.", min_value=2, max_value=len(df), value=3)

    # '저장' 버튼 추가
    if st.button('저장'):
        st.write(f"{num_groups}개의 그룹으로 유저를 분류합니다.")

        # K-means 클러스터링 적용 (유저를 그룹 개수에 맞게 나눕니다.)
        features = pd.get_dummies(df[['Preference', 'MBTI', 'Job', 'Hobby', 'Ideal Type', 'Gender']])
        # 수치형 데이터는 그대로 사용
        numerical_features = df[['Age', 'Height', 'Weight']]
        all_features = pd.concat([features, numerical_features], axis=1)

        kmeans = KMeans(n_clusters=num_groups, random_state=42)
        df['Cluster'] = kmeans.fit_predict(all_features)

        # 그룹별로 나눠진 유저를 보여주기
        for cluster in range(num_groups):
            group_df = df[df['Cluster'] == cluster].reset_index(drop=True)
            with st.expander(f"그룹 {cluster + 1} ({len(group_df)}명)"):
                st.write(", ".join(group_df['User ID'].tolist()))

            # 그룹 내 사용자들에 대한 특징 요약
            summary = f"이 그룹은 주로 {', '.join(group_df['Job'].unique())} 직업군을 포함하고, " \
                      f"MBTI는 {', '.join(group_df['MBTI'].unique())} 유형들이 포함되어 있습니다. " \
                      f"취미로는 {', '.join(group_df['Hobby'].unique())}가 있으며, {', '.join(group_df['Preference'].unique())}에 관심들이 있습니다. " \
                      f"평균 나이는 {group_df['Age'].mean():.1f}세이고, 성비는 {group_df['Gender'].value_counts().to_dict()}입니다."
            st.write(f"그룹 {cluster + 1} 데이터 평가: {summary}")

            # 그룹별 유저 간 호감도(유사도) 계산
            if len(group_df) > 1:
                group_features = all_features.iloc[group_df.index]
                similarity_matrix = cosine_similarity(group_features)
                np.fill_diagonal(similarity_matrix, np.nan)  # 자기 자신과의 유사도는 제외

                # 호감도 매트릭스를 데이터프레임으로 변환
                similarity_df = pd.DataFrame(similarity_matrix, index=group_df['User ID'], columns=group_df['User ID'])

                st.write(f"그룹 {cluster + 1}의 유저 간 호감도 매트릭스:")
                st.write(similarity_df)

                # 각 유저별 평균 호감도 계산
                mean_affinity = similarity_df.mean(axis=1)
                affinity_df = pd.DataFrame({'User ID': mean_affinity.index, 'Average Affinity': mean_affinity.values})
                st.write("각 유저별 평균 호감도:")
                st.write(affinity_df)

            else:
                st.write("유저가 한 명이므로 호감도 계산을 할 수 없습니다.")

            # OpenAI에게 그룹 요약 및 질문 생성 요청
            def generate_openai_response(summary):
                prompt = f"""
                다음은 한 그룹의 특징입니다:

                {summary}

                이 그룹의 특징을 바탕으로, 해당 그룹 내의 사람들이 서로에게 호감을 가질만한 이유를 분석하고, 서로에게 궁금해하고 끌릴 만한 질문 리스트를 만들어 주세요.
                """
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes group characteristics and creates engaging questions."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response['choices'][0]['message']['content'].strip()

            # OpenAI를 통해 생성된 그룹 요약 및 질문 리스트 받기
            openai_response = generate_openai_response(summary)
            st.write(f"그룹 {cluster + 1}의 AI 기반 분석 및 질문 리스트:")
            st.write(openai_response)

        # AI 호스트(사회자) 역할 수행
        st.write("AI 호스트를 통해 설문을 주기적으로 실행합니다.")
        st.write("이 기능은 주기적으로 실행되어 유저들 간의 상호작용을 촉진합니다.")
