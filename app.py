import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import yaml
from dotenv import load_dotenv
from datetime import datetime
import logging


# 로깅 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경 변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# **1. Config 파일 로드 함수**
def load_config():
    config_path = os.path.join(os.getcwd(), "config.yml")
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            if "text_columns" not in config or "tag_options" not in config:
                raise ValueError("Config에 'text_columns' 또는 'tag_options' 키가 없습니다.")
            return config
    except Exception as e:
        st.error(f"config.yml 파일 로드 오류: {e}")
        return {}

# 유사도 계산
def calculate_similarity(user1, user2, feature_weights, all_features):
    similarity = 0
    for feature, weight in feature_weights.items():
        if feature in ['Age', 'Height', 'Weight']:
            max_val = all_features[feature].max()
            min_val = all_features[feature].min()
            if max_val != min_val:
                norm_diff = abs(user1[feature] - user2[feature]) / (max_val - min_val)
                similarity += (1 - norm_diff) * weight
            else:
                similarity += weight
        else:
            similarity += (user1[feature] == user2[feature]) * weight
    return similarity / sum(feature_weights.values())

# 그룹 생성 : 클러스터링 그룹 생성 및 재배정.
def create_mixed_groups(users_df, all_features, min_group_size=6, max_group_size=8, min_females=3):
    females_df = users_df[users_df['Gender'] == 'Female']
    males_df = users_df[users_df['Gender'] == 'Male']

    # 여성 수에 따라 클러스터 수 결정
    n_clusters = max(len(females_df) // 3, 1)

    # 여성에 대해 K-means 클러스터링 수행
    female_features = all_features.loc[females_df.index]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    females_df['Cluster'] = kmeans.fit_predict(female_features)

    # 클러스터별로 여성 그룹 생성
    female_groups = [females_df[females_df['Cluster'] == i] for i in range(n_clusters)]

    # 3명 미만 그룹 재조정
    def redistribute_females(groups, min_count):
        while any(len(group) < min_count for group in groups):
            for i, group in enumerate(groups):
                if len(group) < min_count:
                    for j, other_group in enumerate(groups):
                        if len(other_group) > min_count:
                            # 유사도가 가장 낮은 여성 찾기
                            group_center = all_features.loc[group.index].mean()
                            other_group_similarities = all_features.loc[other_group.index].apply(
                                lambda x: cosine_similarity([x], [group_center])[0][0], axis=1
                            )
                            least_similar = other_group.loc[other_group_similarities.idxmin()] # 유사도가 가장 낮은 사용자.

                            # 여성 이동
                            groups[i] = pd.concat([groups[i], least_similar.to_frame().T], ignore_index=True)
                            groups[j] = groups[j].drop(least_similar.name)
                            break
                    break
        return groups

    female_groups = redistribute_females(female_groups, min_females)

    # 각 여성 그룹에 남성 추가
    final_groups = []
    remaining_males = males_df.copy()

    feature_weights = {
        'Job': 0.2, 'MBTI': 0.2, 'Hobby': 0.2, 'Ideal Type': 0.1,
        'Age': 0.1, 'Height': 0.1, 'Weight': 0.1
    }

    for female_group in female_groups:
        group = female_group.copy()

        # 그룹에 남성 추가 (최대 8명까지)
        males_to_add = min(max_group_size - len(group), len(remaining_males))
        if males_to_add > 0:
            male_similarities = remaining_males.apply(
                lambda male: np.mean([calculate_similarity(male, female, feature_weights, all_features) for _, female in
                                      group.iterrows()]),
                axis=1
            )
            top_males = remaining_males.loc[male_similarities.nlargest(males_to_add).index]
            group = pd.concat([group, top_males], ignore_index=True)
            remaining_males = remaining_males.drop(top_males.index)

        final_groups.append(group)

    return final_groups, remaining_males


def calculate_group_similarity(group, all_features):
    feature_weights = {
        'Job': 0.2, 'MBTI': 0.2, 'Hobby': 0.2, 'Ideal Type': 0.1,
        'Age': 0.1, 'Height': 0.1, 'Weight': 0.1
    }

    similarities = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            user1 = group.iloc[i]
            user2 = group.iloc[j]
            similarities.append(calculate_similarity(user1, user2, feature_weights, all_features))

    return np.mean(similarities) if similarities else 0



# **2. 자유 텍스트 태그 생성 함수**
def generate_tags_from_text(column, text, config):
    # if config and column in config['tag_options']:
    #     prompt = f"문장: {text}\n다음 리스트에서 관련된 태그를 선택하여 제시만 해주세요:\n  {', '.join(config['tag_options'][column])}."
    #     try:
    #         response = openai.ChatCompletion.create(
    #             model="gpt-4o-mini",
    #             messages=[{"role": "user", "content": prompt}]
    #         )
    #         return response['choices'][0]['message']['content'].strip()
    #     except Exception as e:
    #         logging.error(f"OpenAI API 호출 실패: {e}")
    #         return "API Error"
    return "No Tags"


# **3. 다중 값 컬럼 처리 함수**
def process_multi_value_column(column, df):
    split_data = df[column].str.split(',', expand=True)
    dummies = pd.get_dummies(split_data, prefix=column)
    return dummies.groupby(level=0, axis=1).max()

# **4. 그룹 재배정 함수 (여성 그룹 보충)**
def reallocate_female_groups(female_groups, insufficient_groups):
    logging.debug(f"재배정 전: {insufficient_groups}")
    while insufficient_groups:
        for donor_group in female_groups:
            if len(donor_group) > 3:
                recipient_group = insufficient_groups.pop(0)
                additional_female = donor_group.iloc[-1:]
                recipient_group = pd.concat([recipient_group, additional_female], ignore_index=True)
                donor_group.drop(additional_female.index, inplace=True)
                if len(recipient_group) >= 3:
                    female_groups.append(recipient_group)
                break
    logging.debug(f"재배정 후: {insufficient_groups}")

# **5. 남성 그룹 생성 및 배정 함수**
def assign_male_to_groups(female_groups, males_df, all_features):
    groups = []
    male_features = all_features.iloc[males_df.index]

    for female_group in female_groups:
        female_group_features = all_features.loc[female_group.index]

        # 남성 인원과 여성 그룹 간의 유사도 계산
        male_similarity = calculate_cosine_similarity(male_features, female_group_features)

        # 필요한 남성 인원 수 계산 (최대 8명까지, 남아있는 남성 수보다 초과하지 않음)
        num_males_needed = min(8 - len(female_group), len(males_df))

        if num_males_needed > 0 and not males_df.empty:
            # 유효한 인덱스만 선택 (유효하지 않은 인덱스 제거)
            top_male_idxs = np.argsort(male_similarity.flatten())[::-1][:num_males_needed]
            valid_idxs = [idx for idx in top_male_idxs if idx < len(males_df)]

            if valid_idxs:
                # 유효한 인덱스를 통해 남성 그룹 생성
                male_group = males_df.iloc[valid_idxs]
                males_df = males_df.drop(male_group.index).reset_index(drop=True)
            else:
                male_group = pd.DataFrame()  # 유효한 인덱스가 없을 때 빈 그룹
        else:
            male_group = pd.DataFrame()  # 필요한 인원이 없을 때 빈 그룹

        # 최종 그룹 생성
        final_group = pd.concat([female_group, male_group], ignore_index=True)
        groups.append(final_group)

    return groups, males_df


# **6. 코사인 유사도 계산 함수**
def calculate_cosine_similarity(male_features, female_group_features):
    return cosine_similarity(
        male_features, female_group_features.mean(axis=0).values.reshape(1, -1)
    )

# **7. 남은 인원 재배정 함수**
def allocate_remaining_users(groups, remaining_males):
    for group in groups:
        while len(group) < 6 and not remaining_males.empty:
            user = remaining_males.iloc[0:1]
            group = pd.concat([group, user], ignore_index=True)
            remaining_males = remaining_males.drop(user.index).reset_index(drop=True)

        if len(group) < 6 and remaining_males.empty:
            st.warning(f"그룹 {groups.index(group) + 1}에 최소 인원을 채우지 못했습니다.")

# **8. AI 질문 생성 함수**
def generate_openai_response(summary):
    prompt = f"""
    다음은 한 그룹의 특징입니다:

    {summary}

    이 그룹의 특징을 바탕으로, 해당 그룹 내의 사람들이 서로에게 호감을 가질만한 이유를 분석하고,
    서로에게 궁금해할 질문 리스트를 만들어 주세요.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        st.error(f"OpenAI API 호출 실패: {e}")
        return ""

# **9. 메인 실행 로직**
def main():
    config = load_config()
    uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요.", type="xlsx")

    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.write("업로드된 유저 데이터:", df)

        for column in config.get("text_columns", []):
            if column in df.columns:
                df[column + "_tags"] = df[column].apply(lambda x: generate_tags_from_text(column, str(x), config))
                tag_dummies = pd.get_dummies(df[column + "_tags"], prefix=column)
                df = pd.concat([df, tag_dummies], axis=1)

        hobby_dummies = process_multi_value_column('Hobby', df)
        ideal_type_dummies = process_multi_value_column('Ideal Type', df)
        features = pd.get_dummies(df[['Preference', 'MBTI', 'Job', 'Gender']])
        numerical_features = df[['Age', 'Height', 'Weight']]
        all_features = pd.concat([features, hobby_dummies, ideal_type_dummies, numerical_features], axis=1)

        # 그룹 생성
        all_groups, remaining_users = create_mixed_groups(df, all_features)

        # 결과 출력 보고서
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        st.markdown(f"## {current_date} 셔플링 계획서")

        # 셔플링 그룹 결과 출력 >>>>
        for idx, group in enumerate(all_groups):
            # 그룹 유사도 계산
            group_similarity = calculate_group_similarity(group, all_features)

            st.markdown(f"## 🌟 셔플링 그룹 {idx + 1}")
            st.markdown(f"#### 📌 기본 정보")
            st.markdown(f" - 그룹인원수: {len(group)}명 ({group['Gender'].value_counts().to_dict()})")
            st.markdown(f" - 그룹내 유사도 평균: {group_similarity:.2f}")
            st.markdown(f" - 평균나이: {group['Age'].mean():.1f}")
            summary = f"직업: {', '.join(group['Job'].unique())}, " \
                      f"MBTI: {', '.join(group['MBTI'].unique())}, " \
                      f"평균 나이: {group['Age'].mean():.1f}, 성비: {group['Gender'].value_counts().to_dict()}"
            st.write(group)  # 그룹 리스트 표

            # AI 질문 생성
            ai_questions = generate_openai_response(summary)
            st.write("AI 생성 질문:", ai_questions)
            st.markdown("---")

        # 전체 그룹 구성 요약
        st.markdown("\n ## 📊 전체 그룹 구성 요약:")
        st.write(f" 🔢 총 그룹 수: {len(all_groups)}")
        st.write(f" 📏 평균 그룹 크기: {sum(len(group) for group in all_groups) / len(all_groups):.2f}")
        st.write(f" ⬇️️ 최소 그룹 크기: {min(len(group) for group in all_groups)}")
        st.write(f" ⬆ 최대 그룹 크기: {max(len(group) for group in all_groups)}")
        st.markdown("---")

        # 남은 사용자 처리
        if not remaining_users.empty:
            st.markdown("\n ## 🚶‍♂️그룹에 배정되지 않은 사용자:")
            st.write(remaining_users)

if __name__ == "__main__":
    main()
