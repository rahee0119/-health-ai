import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(page_title="Bio AI System", page_icon="🩺", layout="centered")
st.title("🩺 AI 기반 8대 만성 질환 예측 및 맞춤 처방 시스템")
st.write("공공데이터 분석 알고리즘을 활용한 개인 맞춤형 바이오 헬스케어 플랫폼입니다.")
st.markdown("---")

glucose_list = [125, 95, 140, 105, 180, 85, 110, 155, 90, 100]
bp_list = [135, 110, 145, 120, 160, 115, 125, 150, 118, 90]
bmi_list = [28.5, 22.0, 24.5, 31.2, 21.5, 26.0, 18.0, 29.0, 23.5, 20.1]
hemo_list = [14.5, 15.0, 13.8, 14.2, 15.2, 14.0, 9.5, 13.5, 10.2, 13.0]
age_list = [45, 23, 55, 34, 62, 28, 41, 58, 31, 16]
outcome_list = [1, 2, 3, 4, 5, 0, 6, 7, 0, 8]

medical_data = {
    "Glucose": glucose_list, "BloodPressure": bp_list, "BMI": bmi_list,
    "Hemoglobin": hemo_list, "Age": age_list, "Outcome": outcome_list
}
df = pd.DataFrame(medical_data)

X = df[["Glucose", "BloodPressure", "BMI", "Hemoglobin", "Age"]]
y = df["Outcome"]

knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn_model.fit(X, y)

st.sidebar.header("📊 입력 방식 선택")
know_data = st.sidebar.radio("자신의 정확한 건강검진 수치를 아시나요?", ("수치를 직접 입력", "생활습관 설문조사로 추정"))

user_glucose, user_bp, user_bmi, user_hemo = 100, 120, 22.0, 14.0

if know_data == "수치를 직접 입력":
    st.header("📝 생체 데이터 직접 입력")
    user_glucose = st.number_input("공복 혈당 수치 (mg/dL)", min_value=50, max_value=300, value=100)
    user_bp = st.number_input("수축기 혈압 수치 (mmHg)", min_value=50, max_value=220, value=120)
    user_bmi = st.slider("BMI 체질량지수 (kg/m²)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    user_hemo = st.slider("헤모글로빈 수치 (g/dL)", min_value=5.0, max_value=20.0, value=14.0, step=0.1)
else:
    st.header("📋 생활 습관 기반 신체 수치 추정 설문")
    
    st.markdown("##### 🩸 파트 1: 혈당 및 대사 성향 분석")
    q1_1 = st.selectbox("Q1. 탕후루, 탄산음료, 정제 탄수화물(빵, 면) 섭취 빈도는?", (1, 2, 3), format_func=lambda x: ["거의 안 먹음", "주 2~3회", "매일 먹음"][x-1])
    q1_2 = st.selectbox("Q2. 식사 후 참기 힘들 정도로 졸음이 쏟아지나요?", (1, 2, 3), format_func=lambda x: ["그렇지 않다", "가끔 졸리다", "항상 졸음이 쏟아진다"][x-1])
    glucose_score = q1_1 + q1_2
    user_glucose = 90 if glucose_score <= 2 else (115 if glucose_score <= 4 else 145)
    
    st.markdown("##### 💓 파트 2: 혈압 및 심혈관 부하 분석")
    q2_1 = st.selectbox("Q3. 평소 국물을 끝까지 마시거나 짜게 드시나요?", (1, 2, 3), format_func=lambda x: ["싱겁게 먹음", "보통", "매우 짜게 먹음"][x-1])
    q2_2 = st.selectbox("Q4. 부모님이나 형제 중 심혈관 질환 가족력이 있나요?", (1, 2, 3), format_func=lambda x: ["없음", "한 분 계심", "두 분 이상 계심"][x-1])
    q2_3 = st.selectbox("Q5. 갑자기 자리에서 일어날 때 눈앞이 핑 돌며 어지러운가요?", (1, 2, 3), format_func=lambda x: ["전혀 없음", "가끔 있음", "매우 자주 있음"][x-1])
    
    bp_score = q2_1 + q2_2
    if q2_3 == 3: 
        user_bp = 90
    else: 
        user_bp = 110 if bp_score <= 2 else (128 if bp_score <= 4 else 150)
        
    st.markdown("##### 🏃‍♂️ 파트 3: 신체 계측 및 대사량 분석")
    height_cm = st.number_input("Q6. 자신의 키 (cm)", min_value=100.0, max_value=220.0, value=165.0)
    weight_kg = st.number_input("Q7. 자신의 몸무게 (kg)", min_value=30.0, max_value=150.0, value=55.0)
    user_bmi = round(weight_kg / ((height_cm/100) ** 2), 1)
    
    st.markdown("##### 🌀 파트 4: 말초 혈액 신호 분석")
    q4_1 = st.selectbox("Q8. 평소 손발이 차갑거나 안색이 창백하다는 말을 듣나요?", (1, 2, 3), format_func=lambda x: ["아니다", "가끔 듣는다", "자주 듣는다"][x-1])
    user_hemo = 15.0 if q4_1 == 1 else (13.2 if q4_1 == 2 else 9.6)

user_age = st.number_input("마지막으로, 나이 (세)를 입력하세요", min_value=1, max_value=120, value=15)

if st.button("🚀 인공지능 종합 건강 처방 리포트 생성"):
    user_profile = np.array([[user_glucose, user_bp, user_bmi, user_hemo, user_age]])
    predicted_class = int(knn_model.predict(user_profile))
    
    st.success("🎉 인공지능 분석이 완벽하게 완료되었습니다!")
    
    st.subheader("📊 추정 생체 수치 대시보드")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("혈당", f"{user_glucose} mg/dL")
    col2.metric("혈압", f"{user_bp} mmHg")
    col3.metric("BMI", f"{user_bmi}")
    col4.metric("헤모글로빈", f"{user_hemo} g/dL")
    
    st.markdown("---")
    
    disease_name_map = {
        0: "만성 질환 안전 지대", 1: "당뇨병", 2: "역류성 식도염", 3: "고혈압", 
        4: "대사성 비만", 5: "심장 질환", 6: "혈액성 빈혈", 7: "뇌혈관 질환(뇌졸중)", 8: "기립성 저혈압"
    }
    
    st.warning(f"⚠️ **[AI 위험도 경고]** 임상 데이터 패턴 분석 결과, 향후 **[{disease_name_map[predicted_class]}]** 발병 우려가 있으니 각별한 주의가 필요합니다.")
    
    st.subheader("📋 의학 통계 기반 개인 맞춤형 행동 처방전")
    
    if predicted_class == 1 or predicted_class == 4:
        st.info("🍏 **식습관:** 정제 탄수화물과 설탕을 제한하고, 식이섬유가 풍부한 거친 통곡물과 채소 위주 식단을 구성하세요.\n\n"
                "💊 **영양제 보충:** [바나바잎 추출물], [크롬], [마그네슘]을 권장합니다.\n\n"
                "🏋️ **운동 습관:** 식후 30분 뒤 하체 스쿼트 및 빠르게 걷기 운동을 30분 이상 매일 수행하세요.")
    elif predicted_class == 3 or predicted_class == 5 or predicted_class == 7:
        st.info("🍏 **식습관:** 나트륨 배출을 위해 저염식을 실천하고, 칼륨이 풍부한 바나나, 아보카도를 드세요.\n\n"
                "💊 **영양제 보충:** [코엔자임Q10], [오메가3], [나토키나제]를 권장합니다.\n\n"
                "🏋️ **운동 습관:** 자전거, 수영, 조깅 같은 유산소 운동을 주 4회 이상 하세요.")
    elif predicted_class == 8:
        st.info("🍏 **식습관:** 하루 2L 이상의 물을 충분히 마시고 적절한 염분을 유지하세요.\n\n"
                "💊 **영양제 보충:** [비타민 B12], [엽산], [철분]을 권장합니다.\n\n"
                "🏋️ **운동 습관:** '실내 자전거'와 '발꿈치 들기 운동'을 매일 하세요.")
    elif predicted_class == 6:
        st.info("🍏 **식습관:** 비타민 C가 풍부한 과일과 함께 붉은 살코기, 시금치를 자주 섭취하세요.\n\n"
                "💊 **영양제 보충:** [철분제]와 [락토페린]을 권장합니다.\n\n"
                "🏋️ **운동 습관:** 가벼운 평지 산책부터 강도를 천천히 늘리세요.")
    else:
        st.info("🍏 **식습관:** 야식과 식후 바로 눕는 습관을 버리고, 규칙적인 식사 시간을 유지하세요.\n\n"
                "💊 **영양제 보충:** [종합비타민]과 [유산균]을 권장합니다.\n\n"
                "🏋️ **운동 습관:** 하루 7,000보 걷기를 기본 목표로 삼고, 스트레칭을 생활화하세요.")
                
    st.caption("※ 본 리포트는 인공지능 알고리즘 분석 결과이며, 의사의 전문적 진단을 대신할 수 없습니다.")
