import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# ==================== 페이지 설정 ====================
st.set_page_config(
    page_title="SARIMA 수요예측 대시보드",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CSS 스타일 ====================
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem !important;
    }
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem !important;
    }
    header[data-testid="stHeader"] {
        height: 2.5rem !important;
        min-height: 2.5rem !important;
    }
    #MainMenu {
        visibility: visible !important;
    }
    footer {
        visibility: hidden;
    }

    .appViewContainer, .main {
        background-color: #ffffff;
    }
    
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e8ebed;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #1a1a1a !important;
    }
    
    body, p, span, label, div {
        color: #1a1a1a !important;
    }
    
    h1, h2, h3 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    h2 {
        border-bottom: none !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
    }
    
    .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1.5px solid #000000 !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 18px !important;
        padding: 12px 24px !important;
        width: auto !important;
    }
    
    .stButton > button:hover {
        background-color: #f3f4f6 !important;
        border-color: #000000 !important;
    }

    [data-testid="stSpinner"] > div {
        border: 4px solid #e0eaff !important;
        border-top: 4px solid #3b82f6 !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        animation: spin 0.8s linear infinite !important;
        margin: 20px auto !important;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    [data-testid="stSpinner"] p {
        text-align: center !important;
        color: #3b82f6 !important;
        font-weight: 600 !important;
    }
    
    [data-testid="metric-container"] {
        background-color: #f8f9fa !important;
        border: 1px solid #e8ebed !important;
        border-left: 3px solid #000000 !important;
        padding: 16px !important;
        border-radius: 6px !important;
        box-shadow: none !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-size: 28px !important;
        font-weight: 800 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricLabel"] {
        color: #666666 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }
    
    [data-testid="metric-container"] [data-testid="stMetricDelta"] {
        color: #000000 !important;
        font-weight: 600 !important;
    }
    
    input[type="number"], .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #d0d3d8 !important;
        border-radius: 4px !important;
        padding: 8px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    [data-testid="stRadio"] {
        padding: 10px;
    }
    
    [data-testid="stRadio"] label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    [data-testid="stFileUploader"] {
        padding: 10px;
    }
    
    [data-testid="dataframe"] {
        background-color: #ffffff !important;
        border: 1px solid #e8ebed !important;
    }
    
    th {
        background-color: #f3f4f6 !important;
        color: #666666 !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #e8ebed !important;
    }
    
    td {
        color: #1a1a1a !important;
        border-color: #f3f4f6 !important;
    }
    
    .stSuccess {
        background-color: #f0fdf4 !important;
        border-left: 3px solid #22c55e !important;
        color: #166534 !important;
    }
    
    .stWarning {
        background-color: #fffbeb !important;
        border-left: 3px solid #f59e0b !important;
        color: #92400e !important;
    }
    
    .stError {
        background-color: #fef2f2 !important;
        border-left: 3px solid #ef4444 !important;
        color: #991b1b !important;
    }
    
    .stInfo {
        background-color: #f0f9ff !important;
        border-left: 3px solid #3b82f6 !important;
        color: #082f49 !important;
    }
    
    hr {
        border: none;
        height: 1px;
        background-color: #e8ebed;
        margin: 20px 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 기본 데이터 로드 ====================
@st.cache_data
def load_data():
    """실제 데이터 로드 (새로운 데이터로 업데이트)"""
    import_values = [84577963, 79079212, 90110217, 91329654, 91411413, 70842890, 71252529, 66935324, 74712550, 82986383, 75211247, 96283067, 81363841, 60361822, 68541968, 67263477, 63593310, 69400022, 66127566, 56851103, 57037998, 63994160, 68521087, 85309687, 68927916, 61526774, 68847408, 69528489, 68644728, 69033836, 70549268, 75918817, 78646276, 86510835, 91136492, 105902527, 85414614, 89206341, 103679572, 93570487, 90453153, 98472620, 93558613, 96669728, 109404473, 97541163, 129608744, 148158413, 137365700, 110723411, 140516444, 125895009, 126038178, 122127936, 118631326, 116653116, 124974944, 147251082, 158291308, 159589922, 163140781, 135472024, 163553891, 157596701, 151169827, 139254079, 145147505, 132676766, 143500339, 159347236, 182073499, 186939741, 175149164, 170212373, 181751604, 144447236, 149593293, 156120113, 168179471, 161731970, 184321215, 201562486, 231749557, 258865649, 226960991, 193731270, 226717870, 221583285, 219320276, 207210225, 214992398, 206021089, 216899563, 233999616, 247921745, 273180787, 242522850, 223398253, 240251470, 224940090, 217135463, 213804627, 211693721, 209707271, 204695111, 197987749, 215456875, 217713453, 206480247, 206968620, 223261339, 209907544, 203355219, 203778683, 209223686, 196976979, 200364322, 218265814, 236402476, 233761255, 225824787, 187425311, 237932401, 210512005, 212300303, 188070559, 203437620, 195851698, 207802843, 229261612, 229242952, 227924794, 182953068, 185912798, 216114315, 212156075, 219241461, 205471148, 226603023, 216264741, 236324573, 231630515, 241898188, 231557272, 192746635, 231876209]
    
    export_values = [2204019, 2678639, 2918878, 3035281, 2830370, 2712353, 3016501, 3072664, 2833542, 2859013, 3660280, 5001969, 4825637, 4161753, 8132315, 8407941, 6251696, 8410830, 7015234, 6509103, 7535325, 8457352, 19768925, 11154088, 7902306, 7330252, 14676919, 10619592, 9737012, 9882426, 8936727, 9275483, 10642349, 9254586, 17176770, 13427479, 8305490, 9365131, 9026563, 6685667, 8413060, 6957343, 5235655, 6525329, 6698787, 7012063, 18029222, 10792935, 9238535, 5338626, 11029980, 10668863, 14274431, 13036994, 11923795, 10798375, 8056116, 11428989, 17021997, 12452030, 11845784, 10201181, 14819685, 15558309, 16208901, 14533110, 13858597, 13276314, 15086411, 18806544, 27120312, 22069765, 14075414, 16503507, 21081484, 24862161, 27536285, 32062710, 35624108, 38671538, 48645680, 46612428, 54147581, 49226981, 41373770, 36825881, 78769189, 72368323, 67539430, 92721505, 67418020, 66550882, 91242484, 71090586, 87689380, 84822220, 66931806, 58635735, 89999067, 63711188, 76703390, 89095816, 68423153, 87499253, 85786275, 76040804, 75106363, 73814001, 57276053, 77504044, 78238254, 85694253, 92369043, 87992679, 71396593, 84330848, 103945519, 74737202, 90859619, 90314065, 92099213, 87565377, 105357102, 119108220, 104377779, 98419689, 117548820, 138893362, 138866651, 124382517, 123439907, 122377933, 86651770, 111395400, 106782940, 126907285, 118650156, 118114804, 133034154, 124170313, 164370041, 126656847, 116481381, 128576598, 117149873, 140314112]
    
    # 날짜 범위 생성 (2014-01 ~ 2026-02)
    dates = pd.date_range('2014-01-01', periods=146, freq='MS')
    
    # 데이터프레임 생성
    import_data = pd.DataFrame({
        '날짜': dates,
        '전자상거래 수입 금액': import_values
    }).set_index('날짜')
    
    export_data = pd.DataFrame({
        '날짜': dates,
        '전자상거래 수출 금액': export_values
    }).set_index('날짜')
    
    return import_data, export_data

# ==================== 세션 상태 초기화 ====================
if 'run_forecast' not in st.session_state:
    st.session_state.run_forecast = False

if 'forecast_months' not in st.session_state:
    st.session_state.forecast_months = 12

if 'import_data' not in st.session_state:
    st.session_state.import_data = None

if 'export_data' not in st.session_state:
    st.session_state.export_data = None

try:
    import_data, export_data = load_data()
    st.session_state.import_data = import_data
    st.session_state.export_data = export_data
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"데이터 로드 오류: {str(e)}")

# ==================== 사이드바 ====================
if data_loaded:
    with st.sidebar:
        st.title("📊 수요예측")
        
        st.markdown("---")
        
        # 데이터 소스 선택
        st.markdown("### 📁 데이터 소스")
        data_source = st.radio(
            "데이터 선택",
            options=["기본 데이터", "파일 업로드"],
            label_visibility="collapsed"
        )
        
        # 수입/수출 데이터 선택
        st.markdown("### 📊 데이터 유형")
        selected_data_type = st.radio(
            "데이터 유형",
            options=["📥 수입 데이터", "📤 수출 데이터"],
            label_visibility="collapsed",
            key="data_type_select"
        )
        
        if data_source == "파일 업로드":
            st.markdown("#### 템플릿 다운로드")
            
            # 선택된 데이터 유형에 따라 템플릿 생성
            if selected_data_type == "📥 수입 데이터":
                template_df = pd.DataFrame({
                    '연도': [2026, 2026, 2026],
                    '월': [3, 4, 5],
                    '전자상거래 수입 금액': [250000000, 260000000, 270000000],
                    '전자상거래 수입 건수': [0, 0, 0],
                    '전체 수입 금액': [0, 0, 0],
                    '전체 수입 건수': [0, 0, 0]
                })
                template_filename = "SARIMA_수입데이터_템플릿.xlsx"
            else:
                template_df = pd.DataFrame({
                    '연도': [2026, 2026, 2026],
                    '월': [3, 4, 5],
                    '전자상거래 수출 금액': [250000000, 260000000, 270000000],
                    '전자상거래 수출 건수': [0, 0, 0],
                    '전체 수출 금액': [0, 0, 0],
                    '전체 수출 건수': [0, 0, 0]
                })
                template_filename = "SARIMA_수출데이터_템플릿.xlsx"
            
            template_buffer = BytesIO()
            with pd.ExcelWriter(template_buffer, engine='openpyxl') as writer:
                template_df.to_excel(writer, index=False)
            template_buffer.seek(0)
            
            st.download_button(
                label="📥 템플릿 다운로드",
                data=template_buffer.getvalue(),
                file_name=template_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.markdown("#### 파일 업로드")
            
            uploaded_file = st.file_uploader("엑셀 파일 선택", type=['xlsx', 'xls'])
            
            if uploaded_file:
                try:
                    new_data = pd.read_excel(uploaded_file)
                    
                    if selected_data_type == "📥 수입 데이터":
                        required_cols = ['연도', '월', '전자상거래 수입 금액']
                        if not all(col in new_data.columns for col in required_cols):
                            st.error(f"❌ 파일에 필요한 열이 없습니다: {required_cols}")
                        else:
                            try:
                                # 날짜 생성 - 명시적 방식
                                new_data['연도'] = new_data['연도'].astype(int)
                                new_data['월'] = new_data['월'].astype(int)
                                new_data['날짜'] = pd.to_datetime(
                                    new_data['연도'].astype(str) + '-' + 
                                    new_data['월'].astype(str).str.zfill(2) + '-01'
                                )
                                new_data_processed = new_data.set_index('날짜').sort_index()
                                
                                st.markdown("#### 📊 미리보기")
                                st.dataframe(new_data_processed[['전자상거래 수입 금액']], use_container_width=True)
                                
                                if st.button("📤 데이터 병합", use_container_width=True, key="merge_import"):
                                    # 수입 데이터 업데이트
                                    existing_import = st.session_state.import_data.copy()
                                    new_import = new_data_processed[['전자상거래 수입 금액']]
                                    st.session_state.import_data = pd.concat([
                                        existing_import,
                                        new_import
                                    ]).drop_duplicates(keep='last').sort_index()
                                    st.success("✅ 수입 데이터에 새로운 데이터가 추가되었습니다!")
                                    st.info(f"📈 현재 총 {len(st.session_state.import_data)}개월의 데이터가 있습니다.")
                            except Exception as date_error:
                                st.error(f"❌ 날짜 변환 오류: {str(date_error)}\n연도와 월이 숫자 형식인지 확인하세요.\n예: 연도=2026, 월=3")
                    
                    else:  # 수출 데이터
                        required_cols = ['연도', '월', '전자상거래 수출 금액']
                        if not all(col in new_data.columns for col in required_cols):
                            st.error(f"❌ 파일에 필요한 열이 없습니다: {required_cols}")
                        else:
                            try:
                                # 날짜 생성 - 명시적 방식
                                new_data['연도'] = new_data['연도'].astype(int)
                                new_data['월'] = new_data['월'].astype(int)
                                new_data['날짜'] = pd.to_datetime(
                                    new_data['연도'].astype(str) + '-' + 
                                    new_data['월'].astype(str).str.zfill(2) + '-01'
                                )
                                new_data_processed = new_data.set_index('날짜').sort_index()
                                
                                st.markdown("#### 📊 미리보기")
                                st.dataframe(new_data_processed[['전자상거래 수출 금액']], use_container_width=True)
                                
                                if st.button("📤 데이터 병합", use_container_width=True, key="merge_export"):
                                    # 수출 데이터 업데이트
                                    existing_export = st.session_state.export_data.copy()
                                    new_export = new_data_processed[['전자상거래 수출 금액']]
                                    st.session_state.export_data = pd.concat([
                                        existing_export,
                                        new_export
                                    ]).drop_duplicates(keep='last').sort_index()
                                    st.success("✅ 수출 데이터에 새로운 데이터가 추가되었습니다!")
                                    st.info(f"📈 현재 총 {len(st.session_state.export_data)}개월의 데이터가 있습니다.")
                            except Exception as date_error:
                                st.error(f"❌ 날짜 변환 오류: {str(date_error)}\n연도와 월이 숫자 형식인지 확인하세요.\n예: 연도=2026, 월=3")
                
                except Exception as e:
                    st.error(f"❌ 파일 처리 오류: {str(e)}")
        
        st.markdown("---")
        
        # SARIMA 파라미터 설정
        st.markdown("### 🔧 SARIMA 파라미터")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**비계절성**")
            p = st.number_input("p (AR 차수)", min_value=0, max_value=5, value=1, key="p_param", step=1)
            d = st.number_input("d (차분 횟수)", min_value=0, max_value=2, value=1, key="d_param", step=1)
            q = st.number_input("q (MA 차수)", min_value=0, max_value=5, value=1, key="q_param", step=1)
        
        with col2:
            st.markdown("**계절성**")
            P = st.number_input("P (SAR 차수)", min_value=0, max_value=2, value=1, key="P_param", step=1)
            D = st.number_input("D (계절 차분)", min_value=0, max_value=2, value=1, key="D_param", step=1)
            Q = st.number_input("Q (SMA 차수)", min_value=0, max_value=2, value=1, key="Q_param", step=1)
        
        st.markdown("---")
        
        # 미래예측기간 설정 (통합)
        st.markdown("### 📅 미래예측기간")
        forecast_months = st.slider(
            "개월 선택",
            min_value=1,
            max_value=36,
            value=st.session_state.forecast_months,
            key="slider_forecast",
            label_visibility="collapsed"
        )
        st.session_state.forecast_months = forecast_months
        
        st.markdown("---")
        
        # 사용가이드 다운로드
        st.markdown("### 📖 사용가이드")
        
        guide_text = """================================================================================
                  SARIMA 수요예측 대시보드 사용 가이드
================================================================================

1. 개요
SARIMA 수요예측 대시보드는 한국의 전자상거래 수입/수출 데이터를 기반으로 
미래 수요를 예측하는 대시보드입니다.

주요 특징
- 146개월(2014.01 ~ 2026.02)의 기본 데이터 내장
- 실시간 데이터 업로드 및 누적 업데이트 기능
- SARIMA 시계열 분석 모델 적용
- 직관적인 시각화 및 통계 분석
- 95% 신뢰구간 포함 예측


2. SARIMA 파라미터 설명

비계절성 (Non-Seasonal)
- p (AR 차수): 과거 값의 영향 정도 (추천값: 0~3)
- d (차분 횟수): 추세 제거 (추천값: 0~2)
- q (MA 차수): 오차항의 영향 (추천값: 0~3)

계절성 (Seasonal)
- P, D, Q: 계절성 주기(12개월)에서의 비계절성 파라미터
- 월별 데이터의 계절 패턴 반영


3. 사용 방법

Step 1: 데이터 유형 선택
- 좌측 사이드바에서 수입 또는 수출 데이터 선택

Step 2: SARIMA 파라미터 설정
- 기본값(1,1,1)x(1,1,1,12)부터 시작
- MAPE 값이 낮을수록 더 좋은 모델

Step 3: 예측 기간 설정
- 1~36개월 범위에서 선택

Step 4: 예측 실행
- 우측 상단의 "예측 실행" 버튼 클릭


문의: SCMmanager@inu.ac.kr
마지막 업데이트: 2026년
================================================================================"""
        
        st.download_button(
            label="📥 사용가이드 다운로드",
            data=guide_text,
            file_name="SARIMA_수요예측_사용가이드.txt",
            mime="text/plain"
        )

    # ==================== 메인 컨텐츠 ====================
    st.markdown("# 📊 SARIMA 수요예측 대시보드")
    st.markdown("한국의 전자상거래 수입/수출 데이터를 기반으로 미래 수요를 예측합니다.")
    
    # 우측 상단에 예측실행 버튼 배치
    col_space1, col_space2, col_btn = st.columns([2, 0.5, 0.5])
    with col_btn:
        forecast_button = st.button("🔮 예측 실행", key="btn_forecast_main", use_container_width=True)
        if forecast_button:
            st.session_state.run_forecast = True
    
    st.markdown("---")
    
    # 선택된 데이터 유형에 따라 데이터 선택
    if selected_data_type == "📥 수입 데이터":
        data_display = st.session_state.import_data['전자상거래 수입 금액']
        data_label = "수입"
    else:
        data_display = st.session_state.export_data['전자상거래 수출 금액']
        data_label = "수출"
    
    # 데이터 분석 섹션
    st.markdown(f"### 📊 {data_label} 데이터 분석 ({len(data_display)}개월)")
    
    # 통계 표시
    latest_val = float(data_display.iloc[-1])
    prev_year_val = float(data_display.iloc[-13]) if len(data_display) >= 13 else latest_val
    yoy_change = ((latest_val - prev_year_val) / prev_year_val * 100) if prev_year_val != 0 else 0
    avg_val = float(data_display.mean())
    max_val = float(data_display.max())
    min_val = float(data_display.min())
    max_date = data_display.idxmax().strftime('%Y.%m')
    min_date = data_display.idxmin().strftime('%Y.%m')
    
    # 통계 메트릭
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("최근 값", f"${latest_val:,.0f}", f"{yoy_change:+.1f}% YoY")
    
    with metric_col2:
        st.metric("평균", f"${avg_val:,.0f}", "전체 기간")
    
    with metric_col3:
        st.metric("최고", f"${max_val:,.0f}", max_date)
    
    with metric_col4:
        st.metric("최저", f"${min_val:,.0f}", min_date)
    
    st.markdown("---")
    
    # 기본 그래프
    st.markdown("### 📊 과거 데이터 추세")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data_display.index,
        y=data_display.values,
        name='실제 데이터',
        mode='lines',
        line=dict(color='#1f2937', width=2),
        hovertemplate='<b>%{x|%Y.%m}</b><br>$%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'<b>{data_label} 데이터 추세</b>',
        xaxis_title='날짜',
        yaxis_title='금액 (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        margin=dict(l=80, r=80, t=80, b=80),
        font=dict(size=11, color='#1a1a1a'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor='#e8ebed', borderwidth=1),
        dragmode='pan'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
    
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
    
    st.markdown("---")
    
    # ==================== 예측 결과 ====================
    if st.session_state.run_forecast:
        with st.spinner("모델 학습 중..."):
            try:
                order = (int(p), int(d), int(q))
                seasonal_order = (int(P), int(D), int(Q), 12)
                
                model = SARIMAX(
                    data_display,
                    order=order,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                results = model.fit(disp=False)
                
                fitted_values = results.fittedvalues
                forecast_result = results.get_forecast(steps=forecast_months)
                forecast_values = forecast_result.predicted_mean
                confidence_intervals = forecast_result.conf_int()
                
                last_date = data_display.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='MS'
                )
                
                # 평가지표
                actual_values = data_display.values
                fitted_for_eval = fitted_values.dropna().values
                actual_for_eval = data_display.iloc[len(data_display) - len(fitted_for_eval):].values
                
                mape = np.mean(np.abs((actual_for_eval - fitted_for_eval) / actual_for_eval)) * 100
                mad = np.mean(np.abs(actual_for_eval - fitted_for_eval))
                mse = np.mean((actual_for_eval - fitted_for_eval) ** 2)
                rmse = np.sqrt(mse)
                
                st.success("🔮 Forecasting Insights !")
                
                # 성능 지표
                st.markdown("### 📈 성능 지표")
                
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("MAPE", f"{mape:.2f}%", "낮을수록 좋음")
                
                with metric_col2:
                    st.metric("MAD", f"${mad:,.0f}", "평균 오차")
                
                with metric_col3:
                    st.metric("RMSE", f"${rmse:,.0f}", "제곱근 오차")
                
                with metric_col4:
                    if mape < 10:
                        status = "✅ GOOD"
                    elif mape < 20:
                        status = "⚠️ WARNING"
                    else:
                        status = "❌ CRITICAL"
                    st.metric("상태", status, f"{mape:.1f}%")
                
                st.markdown("---")
                
                # 예측 차트
                st.markdown("### 📊 예측 결과")
                
                fig = go.Figure()
                
                # 실제 데이터
                fig.add_trace(go.Scatter(
                    x=data_display.index,
                    y=data_display.values,
                    name='실제 데이터',
                    mode='lines',
                    line=dict(color='#1f2937', width=2),
                    hovertemplate='<b>실제</b><br>%{x|%Y.%m}<br>$%{y:,.0f}<extra></extra>'
                ))
                
                # 적합 데이터
                fig.add_trace(go.Scatter(
                    x=fitted_values.index,
                    y=fitted_values.values,
                    name='적합 데이터',
                    mode='lines',
                    line=dict(color='#6366f1', width=1.8, dash='dot'),
                    opacity=0.75,
                    hovertemplate='<b>적합</b><br>%{x|%Y.%m}<br>$%{y:,.0f}<extra></extra>'
                ))
                
                # 브릿지 데이터
                last_actual_date = data_display.index[-1]
                last_actual_val = float(data_display.iloc[-1])
                first_forecast_val = float(forecast_values.values[0])
                bridge_x = [last_actual_date, forecast_dates[0]]
                bridge_y = [last_actual_val, first_forecast_val]
                
                # 신뢰구간
                ci_upper = confidence_intervals.iloc[:, 1].values
                ci_lower = confidence_intervals.iloc[:, 0].values
                bridge_ci_x = [last_actual_date] + list(forecast_dates)
                bridge_ci_upper = [last_actual_val] + list(ci_upper)
                bridge_ci_lower = [last_actual_val] + list(ci_lower)
                
                fig.add_trace(go.Scatter(
                    x=bridge_ci_x,
                    y=bridge_ci_upper,
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=bridge_ci_x,
                    y=bridge_ci_lower,
                    name='신뢰구간 (95%)',
                    mode='lines',
                    line=dict(width=0),
                    fillcolor='rgba(239, 68, 68, 0.18)',
                    fill='tonexty',
                    hoverinfo='skip'
                ))
                
                # 예측 데이터
                fig.add_trace(go.Scatter(
                    x=bridge_x + list(forecast_dates[1:]),
                    y=bridge_y + list(forecast_values.values[1:]),
                    name='예측 데이터',
                    mode='lines+markers',
                    line=dict(color='#ef4444', width=2.5),
                    marker=dict(size=6, color='#ef4444', line=dict(width=1, color='#ffffff')),
                    hovertemplate='<b>예측</b><br>%{x|%Y.%m}<br>$%{y:,.0f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'<b>{data_label} 수요 예측 (SARIMA{order}×{seasonal_order})</b>',
                    xaxis_title='날짜',
                    yaxis_title='금액 (USD)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    margin=dict(l=80, r=80, t=80, b=80),
                    font=dict(size=11, color='#1a1a1a'),
                    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor='#e8ebed', borderwidth=1),
                    dragmode='pan'
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})
                
                st.markdown("---")
                
                # 예측값 테이블
                st.markdown("### 📋 예측값")
                
                forecast_table = pd.DataFrame({
                    '날짜': forecast_dates.strftime('%Y.%m'),
                    '예측값': [f'${x:,.0f}' for x in forecast_values.values.astype(int)],
                    '하한': [f'${x:,.0f}' for x in confidence_intervals.iloc[:, 0].values.astype(int)],
                    '상한': [f'${x:,.0f}' for x in confidence_intervals.iloc[:, 1].values.astype(int)],
                })
                
                st.dataframe(forecast_table, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                
                # 모델 정보
                st.markdown("### 🔍 모델 정보")
                
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                with info_col1:
                    st.metric("파라미터", f"{order}×{seasonal_order}")
                with info_col2:
                    st.metric("AIC", f"{results.aic:.2f}")
                with info_col3:
                    st.metric("BIC", f"{results.bic:.2f}")
                with info_col4:
                    st.metric("데이터 기간", f"{len(data_display)}개월")
                
            except Exception as e:
                st.error(f"❌ 오류: {str(e)}")
                st.info("💡 파라미터를 조정해보세요.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999999; font-size: 11px; padding: 8px 0;'>
        <p style='margin: 2px 0;'>SARIMA (Seasonal Autoregressive Integrated Moving Average) 기반 수요예측</p>
        <p style='margin: 2px 0;'>데이터 출처: 한국무역통계진흥원</p>
        <p style='margin: 2px 0;'>All Rights Reserved 인천대 동북아물류대학원 SCM Class</p>
    </div>
    """, unsafe_allow_html=True)
