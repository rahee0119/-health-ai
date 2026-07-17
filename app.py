import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# 웹페이지 스타일 및 타이틀 설정
st.set_page_config(page_title="Bio AI Health Platform", page_icon="🩺", layout="centered")

# 메인 헤더 디자인 꾸미기
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🩺 바이오 AI 헬스케어 플랫폼</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6B7280; font-size: 18px;'>공공 보건의료 빅데이터 기반 만성 질환 예측 및 맞춤형 행동 처방 시스템</p>", unsafe_allow_html=True)
st.markdown("---")

# [백엔드] 고정 데이터셋 구축 (깨짐 방지용 완벽 정수 데이터)
glucose_list = [135, 95, 140, 110, 130, 98, 85, 120, 105, 92]
bp_list = [145, 115, 150, 120, 135, 110, 90, 130, 122, 95]
bmi_list = [28.5, 22.0, 24.5, 31.2, 21.5, 26.0, 18.0, 29.0, 23.5, 20.1]
hemo_list = [14.5, 15.0, 13.8, 14.2, 15.2, 14.0, 9.5, 13.5, 10.2, 13.0]
age_list = [45, 18, 55, 32, 60, 22, 15, 50, 28, 14]
outcome_list = [1, 0, 3, 4, 5, 2, 8, 7, 0, 6]

medical_data = {
    "Glucose": glucose_list, "BloodPressure": bp_list, "BMI": bmi_list,
    "Hemoglobin": hemo_list, "Age": age_list, "Outcome": outcome_list
}
df = pd.DataFrame(medical_data)

X = df[["Glucose", "BloodPressure", "BMI", "Hemoglobin", "Age"]]
y = df["Outcome"]

knn_model = KNeighborsClassifier(n_neighbors=1, weights='uniform')
knn_model.fit(X, y)

# 화면 레이아웃 분리
st.sidebar.header("📊 입력 방식 선택")
know_data = st.sidebar.radio("자신의 정확한 건강검진 수치를 아시나요?", ("생활습관 설문조사로 추정", "수치를 직접 입력"))

# 변수 초기화
user_glucose = 100
user_bp = 120
user_bmi = 22.0
user_hemo = 14.0

if know_data == "수치를 직접 입력":
    st.header("📝 생체 데이터 직접 입력")
    st.success("💡 병원이나 보건소에서 측정한 정확한 수치를 입력하면 데이터 기반 인공지능 진단 정확도가 극대화됩니다.")
    user_glucose = st.number_input("공복 혈당 수치 입력 (mg/dL)", min_value=50, max_value=300, value=100)
    user_bp = st.number_input("수축기 혈압 수치 입력 (mmHg)", min_value=50, max_value=220, value=120)
    user_bmi = st.slider("BMI 체질량지수 조절 (kg/m²)", min_value=10.0, max_value=50.0, value=22.0, step=0.1)
    user_hemo = st.slider("헤모글로빈 수치 조절 (g/dL)", min_value=5.0, max_value=20.0, value=14.0, step=0.1)
else:
    st.header("📋 생활 습관 기반 신체 수치 추정 설문")
    st.info("💡 자신의 정확한 혈압이나 혈당 수치를 모르더라도, 아래 설문을 통해 의학 통계 기반의 수치를 가중 추정합니다.")
    
    st.markdown("#### 🩸 파트 1: 혈당 및 대사 성향 분석")
    q1_1 = st.selectbox("Q1. 당류(탕후루, 탄산음료) 및 정제 탄수화물(빵, 면) 섭취 빈도는 어떤가요?", (1, 2, 3), format_func=lambda x: ["거의 안 먹음 (안정적)", "주 2~3회 (보통)", "매일 먹음 (과다)"][x-1])
    q1_2 = st.selectbox("Q2. 식사 직후 참기 힘들 정도로 졸음이 쏟아지거나 무기력해지시나요?", (1, 2, 3), format_func=lambda x: ["그렇지 않다", "가끔 졸리다", "식후 항상 강한 피로감이 온다"][x-1])
    glucose_score = q1_1 + q1_2
    user_glucose = 90 if glucose_score <= 2 else (115 if glucose_score <= 4 else 145)
    
    st.markdown("#### 💓 파트 2: 혈압 및 심혈관 부하 분석")
    q2_1 = st.selectbox("Q3. 평소 국물 요리의 국물을 완전히 마시거나 음식을 짜게 드시나요?", (1, 2, 3), format_func=lambda x: ["싱겁게 먹음 (저염식)", "보통 소스류 즐김", "매우 짜고 자극적이게 먹음"][x-1])
    q2_2 = st.selectbox("Q4. 부모님이나 직계 가족 중 고혈압, 심장병, 뇌졸중 내력이 있으신가요?", (1, 2, 3), format_func=lambda x: ["없음", "한 분 계심", "두 분 이상 계심"][x-1])
    q2_3 = st.selectbox("Q5. 앉아있거나 누워있다가 갑자기 일어설 때 순간적으로 눈앞이 아찔하며 어지러운가요?", (1, 2, 3), format_func=lambda x: ["전혀 없음", "가끔 핑 돈다", "매우 자주 주저앉을 정도로 어지럽다"][x-1])
    
    bp_score = q2_1 + q2_2
    if q2_3 == 3: user_bp = 90
    else: user_bp = 110 if bp_score <= 2 else (128 if bp_score <= 4 else 150)
        
    st.markdown("#### 🏃‍♂️ 파트 3: 신체 계측 정보 분석")
    height_cm = st.number_input("Q6. 현재 자신의 키를 알려주세요 (cm 단위)", min_value=100.0, max_value=220.0, value=165.0)
    weight_kg = st.number_input("Q7. 현재 자신의 몸무게를 알려주세요 (kg 단위)", min_value=30.0, max_value=150.0, value=55.0)
    user_bmi = round(weight_kg / ((height_cm/100) ** 2), 1)
    
    st.markdown("#### 🌀 파트 4: 말초 혈액 신호 분석")
    q4_1 = st.selectbox("Q8. 평소 손발이 남들에 비해 차갑거나 안색이 창백하다는 지적을 받나요?", (1, 2, 3), format_func=lambda x: ["아니다 (따뜻함)", "겨울철에만 가끔 들음", "사계절 내내 자주 듣는다"][x-1])
    user_hemo = 15.0 if q4_1 == 1 else (13.2 if q4_1 == 2 else 9.6)

st.markdown("---")
user_age = st.number_input("📌 최종 분석을 위해 현재 나이를 입력하세요 (세)", min_value=1, max_value=120, value=15)

# 리포트 출력 
if st.button("🚀 인공지능 종합 맞춤형 분석 리포트 발행"):
    user_profile = np.array([[float(user_glucose), float(user_bp), float(user_bmi), float(user_hemo), float(user_age)]])
    predicted_res = knn_model.predict(user_profile)
    predicted_class = int(predicted_res[0])
    
    st.balloons() 
    st.markdown("<h3 style='text-align: center; color: #10B981;'>📊 생체 데이터 예측 대시보드</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🩸 추정 혈당", f"{int(user_glucose)} mg/dL")
    col2.metric("💓 추정 혈압", f"{int(user_bp)} mmHg")
    col3.metric("🏃‍♂️ BMI 지수", f"{user_bmi}")
    col4.metric("🌀 헤모글로빈", f"{user_hemo} g/dL")
    
    st.markdown("---")
    
    disease_name_map = {
        0: "만성 질환 안전 지대", 1: "당뇨병", 2: "역류성 식도염", 3: "고혈압", 
        4: "대사성 비만", 5: "심장 질환", 6: "혈액성 빈혈", 7: "뇌혈관 질환(뇌졸중)", 8: "기립성 저혈압"
    }
    
    st.markdown(f"<div style='background-color: #FEF3C7; padding: 15px; border-left: 5px solid #F59E0B; border-radius: 4px;'>"
                f"<span style='color: #92400E; font-weight: bold;'>⚠️ [인공지능 생체 데이터 패턴 분류 결과]</span><br>"
                f"<span style='font-size: 16px; color: #78350F;'>사용자의 현재 라이프스타일 패턴은 향후 <b>[{disease_name_map[predicted_class]}]</b> 유발 위험군 데이터와 가장 유사합니다.</span>"
                f"</div>", unsafe_allow_html=True)
    
    st.markdown("<br><h4 style='color: #1E3A8A;'>📋 맞춤형 예방 및 정밀 처방전</h4>", unsafe_allow_html=True)
    
    if predicted_class == 1 or predicted_class == 4:
        st.info("🍏 **의학적 식습관 조절:** 정제 탄수화물과 가공 설탕(당류) 섭취를 엄격히 제한하고, 식이섬유가 배출을 돕는 통곡물과 채소 식단을 도입하세요.\n\n"
                "💊 **바이오 영양제 권장:** 세포의 포도당 대사 능력 및 인슐린 감수성 개선을 유도하기 위해 [바나바잎 추출물], [크롬], [마그네슘] 배합을 추천합니다.\n\n"
                "🏋️ **과학적 운동 요법:** 혈당 스파이크를 방지하기 위해 식후 30분 시점에 허벅지 대근육을 소모하는 스쿼트 및 30분 인터벌 걷기를 실천하세요.")
    elif predicted_class == 3 or predicted_class == 5 or predicted_class == 7:
        st.info("🍏 **의학적 식습관 조절:** 삼투압성 혈관 스트레스를 낮추기 위해 국물 염분을 제한하는 저염식을 수행하고, 나트륨을 빼주는 칼륨 식품(바나나)을 자주 섭취하세요.\n\n"
                "💊 **바이오 영양제 권장:** 혈관 벽 내피세포 탄성도 보호와 혈행 개선을 고려해 [코엔자임Q10], [오메가3 고함량], [나토키나제] 보충을 권장합니다.\n\n"
                "🏋️ **과학적 운동 요법:** 말초 혈관 유연성을 증가시키기 위해 급격한 웨이트 트레이닝보다 수영, 고정 자전거 등 전신 유산소 운동을 주 4회 규칙적으로 수행하세요.")
    elif predicted_class == 8:
        st.info("🍏 **의학적 식습관 조절:** 기립 시 일시적인 탈수성 뇌혈류량 감소를 예방하도록 수시로 종이컵 10잔 이상의 청정 수분과 적절한 미네랄 염분을 공급해 주세요.\n\n"
                "💊 **바이오 영양제 권장:** 말초 신경계 자율 조절 및 혈액 합성 효율 증대를 위해 활성형 [비타민 B12], [체내 흡수율이 높은 철분], [엽산] 조합을 권장합니다.\n\n"
                "🏋️ **과학적 운동 요법:** 하체 정맥의 혈류가 아래로 정체되는 현상을 방지하기 위해 매일 종아리 비복근을 자극하는 '발꿈치 들기 운동' 30회 3세트를 생활화하세요.")
    elif predicted_class == 6:
        st.info("🍏 **의학적 식습관 조절:** 적혈구내 산소 운반 결합력을 정상화하기 위해 헴철 구조가 풍부한 붉은 살코기, 동물성 단백질 및 시금치를 비타민 C와 융합해 섭취하세요.\n\n"
                "💊 **바이오 영양제 권장:** 조혈 작용 및 혈액 면역 인자 보충을 목표로 소화 방해율이 낮은 [가용성 철분제]와 [락토페린 영양 균형제]를 추천합니다.\n\n"
                "🏋️ **과학적 운동 요법:** 전신 산소 포화도가 쉽게 떨어질 수 있는 상태이므로 숨이 가쁜 무리한 고강도 운동은 절대 금하며, 가벼운 평지 산책부터 시작해 점진적으로 심폐 능력을 강화하세요.")
    else:
        st.info("🍏 **의학적 식습관 조절:** 소화 효소의 역류 및 소화기관 내 압력을 증가시키는 야식 섭취 및 식후 즉시 와식(눕는 행동) 생활 방식을 즉시 지양하세요.\n\n"
                "💊 **바이오 영양제 권장:** 전반적인 생체 에너지 대사 흐름 원활화와 소화 장벽 장내 유익균 보충을 위해 고함량 [종합비타민 B군]과 [4세대 유산균] 배합을 추천합니다.\n\n"
                "🏋️ **과학적 운동 요법:** 신체 항상성 지표 균형을 유도하기 위해 하루 최소 7,000보 보행을 기본 루틴으로 설정하고 척추 균형 스트레칭을 정기적으로 수행하세요.")
                
    st.markdown("<p style='font-size:12px; color:gray; text-align:center;'>※ 본 정보과학 임상 분석 리포트는 KNN 거리 연산 모델 기반 참고 수치이며, 정밀한 의료적 확진은 반드시 전문 의사의 진단서에 준하여 결정되어야 합니다.</p>", unsafe_allow_html=True)
