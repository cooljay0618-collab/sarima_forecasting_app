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
    /* 상단 여백 축소 - 메인 컨텐츠 */
    .block-container {
        padding-top: 1rem !important;
    }
    /* 사이드바 상단 여백 축소 */
    [data-testid="stSidebar"] .block-container,
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem !important;
    }
    /* 헤더(툴바/Deploy)는 유지, 높이만 줄임 */
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

    /* 전체 배경 */
    .appViewContainer, .main {
        background-color: #ffffff;
    }
    
    /* 사이드바 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e8ebed;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #f8f9fa;
    }
    
    /* 사이드바 텍스트 */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #1a1a1a !important;
    }
    
    /* 텍스트 */
    body, p, span, label, div {
        color: #1a1a1a !important;
    }
    
    /* 헤더 */
    h1, h2, h3 {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    h2 {
        border-bottom: none !important;
        margin-top: 20px !important;
        margin-bottom: 15px !important;
    }
    
    /* 기본 버튼 */
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

    /* 예측 실행 버튼 (중앙 배치) - 흰 배경 유지, 글자만 크고 굵게 */
    [data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1.5px solid #000000 !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        font-size: 20px !important;
        padding: 14px 0 !important;
        width: 100% !important;
        letter-spacing: 0.3px !important;
    }
    [data-testid="stHorizontalBlock"] > div:nth-child(2) .stButton > button:hover {
        background-color: #f3f4f6 !important;
    }

    /* 로딩 스피너 - 파란 원형 점선 스타일 */
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
    
    /* 메트릭 카드 */
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
    
    /* 입력 필드 */
    input[type="number"], .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #d0d3d8 !important;
        border-radius: 4px !important;
        padding: 8px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
    }
    
    /* 라디오 */
    [data-testid="stRadio"] {
        padding: 10px;
    }
    
    [data-testid="stRadio"] label {
        color: #000000 !important;
        font-weight: 500 !important;
    }
    
    /* 파일 업로더 */
    [data-testid="stFileUploader"] {
        padding: 10px;
    }
    
    /* 데이터프레임 */
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
    
    /* 메시지 */
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
    
    /* 구분선 */
    hr {
        border: none;
        height: 1px;
        background-color: #e8ebed;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 데이터 로딩 ====================
@st.cache_data
def load_default_data():
    try:
        import_df = pd.read_excel('전자상거래무역_수입.xlsx', sheet_name='전자상거래 수입')
        export_df = pd.read_excel('전자상거래무역_수출.xlsx', sheet_name='전자상거래 수출')
        
        def preprocess(df, amount_col):
            df_clean = df.copy()
            df_clean['date'] = pd.to_datetime(
                df_clean['연도'].astype(str) + '-' + 
                df_clean['월'].astype(str).str.zfill(2) + '-01'
            )
            
            if df_clean[amount_col].dtype == 'object':
                df_clean[amount_col] = df_clean[amount_col].str.replace(',', '').astype('Int64')
            else:
                df_clean[amount_col] = df_clean[amount_col].astype('Int64')
            
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
            df_clean = df_clean.set_index('date')
            
            return df_clean[[amount_col]]
        
        import_data = preprocess(import_df, '전자상거래 수입 금액')
        export_data = preprocess(export_df, '전자상거래 수출 금액')
        
        return import_data, export_data, True
    
    except FileNotFoundError:
        return None, None, False

def create_template():
    """템플릿 엑셀 파일 생성"""
    template_df = pd.DataFrame({
        '연도': [2024, 2024, 2024, 2024, 2024],
        '월': [1, 2, 3, 4, 5],
        '전자상거래 수입 금액': [84577963, 79079212, 90110217, 91329654, 91411413],
        '전자상거래 수입 건수': [869808, 796398, 882972, 895439, 908412],
        '전체 수입 금액': [44746013067, 42061553951, 45558674670, 45873332924, 42607482640],
        '전체 수입 건수': [1426765, 1314508, 1522225, 1553287, 1517623]
    })
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        template_df.to_excel(writer, index=False, sheet_name='데이터')
    output.seek(0)
    return output

def process_uploaded_file(uploaded_file):
    """업로드된 파일 처리"""
    try:
        df = pd.read_excel(uploaded_file)
        
        # 필수 칼럼 확인
        required_cols = ['연도', '월']
        if not all(col in df.columns for col in required_cols):
            st.error("❌ 필수 칼럼이 없습니다: 연도, 월")
            return None, None
        
        df['date'] = pd.to_datetime(
            df['연도'].astype(str) + '-' + 
            df['월'].astype(str).str.zfill(2) + '-01'
        )
        
        # 전자상거래 금액 칼럼 찾기
        amount_col = None
        for col in df.columns:
            if '전자상거래' in col and '금액' in col:
                amount_col = col
                break
        
        if not amount_col:
            st.error("❌ '전자상거래 XXX 금액' 칼럼이 필요합니다")
            return None, None
        
        # 데이터 정제
        if df[amount_col].dtype == 'object':
            df[amount_col] = df[amount_col].str.replace(',', '').astype('Int64')
        else:
            df[amount_col] = df[amount_col].astype('Int64')
        
        df = df.sort_values('date').reset_index(drop=True)
        df = df.set_index('date')
        
        return df[[amount_col]], amount_col
    
    except Exception as e:
        st.error(f"❌ 파일 처리 오류: {str(e)}")
        return None, None

# ==================== 초기화 ====================
import_data, export_data, has_default = load_default_data()

# ==================== 사이드바 ====================
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0; text-align: center; border-bottom: 1px solid #e8ebed; margin-bottom: 20px;'>
        <h2 style='margin: 0; font-size: 24px; color: #000000;'>📊 DASHBOARD</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # 데이터 소스 선택
    st.markdown("**📂 데이터 소스**")
    data_source = st.radio(
        "데이터 선택",
        options=["기본 데이터", "파일 업로드"],
        label_visibility="collapsed",
        key="data_source_radio"
    )
    
    if data_source == "기본 데이터":
        st.markdown("**내장된 수출입 데이터 사용**")
        analysis_mode = st.radio(
            "데이터 선택",
            options=["수입 데이터", "수출 데이터"],
            label_visibility="collapsed",
            key="analysis_radio"
        )
        
        if analysis_mode == "수입 데이터":
            data = import_data.copy() if has_default else None
            column_name = '전자상거래 수입 금액'
            data_label = "수입"
        else:
            data = export_data.copy() if has_default else None
            column_name = '전자상거래 수출 금액'
            data_label = "수출"
    
    else:
        st.markdown("**📥 데이터 업로드**")
        
        # 템플릿 다운로드
        template_file = create_template()
        st.download_button(
            label="📋 템플릿 다운로드",
            data=template_file,
            file_name="SARIMA_데이터_템플릿.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
            key="download_template"
        )
        
        st.markdown("---")
        
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "엑셀 파일 업로드",
            type=['xlsx', 'xls'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            data, column_name = process_uploaded_file(uploaded_file)
            if data is not None:
                st.success(f"✅ 파일 로드 완료: {len(data)}개월 데이터")
                data_label = "커스텀"
            else:
                data = None
        else:
            data = None
    
    st.markdown("---")
    
    # SARIMA 파라미터
    st.markdown("**📈 SARIMA 파라미터**")
    
    st.markdown("""
    <div style='background-color: #f3f4f6; padding: 10px 12px; border-radius: 6px; margin-bottom: 8px;'>
        <p style='margin: 0; font-weight: 600; font-size: 13px; color: #000000;'>비계절성 (Non-Seasonal)</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.markdown("<p style='text-align:center; font-size:12px; font-weight:600; color:#555; margin-bottom:2px;'>p (AR 차수)</p>", unsafe_allow_html=True)
        p = st.number_input(
            "p", min_value=0, max_value=10, value=1,
            label_visibility="collapsed", key="param_p"
        )
    with col2:
        st.markdown("<p style='text-align:center; font-size:12px; font-weight:600; color:#555; margin-bottom:2px;'>d (차분 횟수)</p>", unsafe_allow_html=True)
        d = st.number_input(
            "d", min_value=0, max_value=2, value=1,
            label_visibility="collapsed", key="param_d"
        )
    with col3:
        st.markdown("<p style='text-align:center; font-size:12px; font-weight:600; color:#555; margin-bottom:2px;'>q (MA 차수)</p>", unsafe_allow_html=True)
        q = st.number_input(
            "q", min_value=0, max_value=10, value=1,
            label_visibility="collapsed", key="param_q"
        )
    
    st.markdown(f"<p style='text-align: center; font-weight: 700; color: #000000; margin: 6px 0 12px 0;'>order = ({p}, {d}, {q})</p>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background-color: #f3f4f6; padding: 10px 12px; border-radius: 6px; margin-bottom: 8px;'>
        <p style='margin: 0; font-weight: 600; font-size: 13px; color: #000000;'>계절성 (Seasonal)</p>
    </div>
    """, unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3, gap="small")
    with col4:
        st.markdown("<p style='text-align:center; font-size:12px; font-weight:600; color:#555; margin-bottom:2px;'>P (SAR 차수)</p>", unsafe_allow_html=True)
        P = st.number_input(
            "P", min_value=0, max_value=5, value=1,
            label_visibility="collapsed", key="param_P"
        )
    with col5:
        st.markdown("<p style='text-align:center; font-size:12px; font-weight:600; color:#555; margin-bottom:2px;'>D (계절 차분)</p>", unsafe_allow_html=True)
        D = st.number_input(
            "D", min_value=0, max_value=2, value=1,
            label_visibility="collapsed", key="param_D"
        )
    with col6:
        st.markdown("<p style='text-align:center; font-size:12px; font-weight:600; color:#555; margin-bottom:2px;'>Q (SMA 차수)</p>", unsafe_allow_html=True)
        Q = st.number_input(
            "Q", min_value=0, max_value=5, value=1,
            label_visibility="collapsed", key="param_Q"
        )
    
    st.markdown(f"<p style='text-align: center; font-weight: 700; color: #000000; margin: 6px 0 12px 0;'>seasonal_order = ({P}, {D}, {Q}, 12)</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 예측 기간 - 완전 연동
    st.markdown("**📅 미래 예측 기간**")
    
    if "fm_val" not in st.session_state:
        st.session_state["fm_val"] = 12
    
    def on_slider():
        st.session_state["fm_val"] = st.session_state["fm_slider"]
    
    def on_input():
        st.session_state["fm_val"] = st.session_state["fm_input"]
    
    fcol1, fcol2 = st.columns([2, 1], gap="small")
    with fcol1:
        st.slider(
            "개월 수 슬라이더",
            min_value=1, max_value=36, step=1,
            value=st.session_state["fm_val"],
            label_visibility="collapsed",
            key="fm_slider",
            on_change=on_slider
        )
    with fcol2:
        st.number_input(
            "개월 직접 입력",
            min_value=1, max_value=36, step=1,
            value=st.session_state["fm_val"],
            label_visibility="collapsed",
            key="fm_input",
            on_change=on_input
        )
    
    forecast_months = st.session_state["fm_val"]
    st.markdown(f"<p style='text-align: center; font-weight: 700; color: #000000; margin: 4px 0;'>📆 {forecast_months}개월 예측</p>", unsafe_allow_html=True)

# ==================== 메인 콘텐츠 ====================
st.markdown("<h1 style='margin: 0 0 4px 0; font-size: 32px;'>📊 수요예측 대시보드</h1>", unsafe_allow_html=True)

# 버튼 스타일 CSS
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"] > div:nth-child(2) > div {
    display: flex;
    justify-content: center;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(2) button {
    height: 44px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    padding: 8px 24px !important;
    min-height: 44px !important;
}
</style>
""", unsafe_allow_html=True)

_, mid_btn, _ = st.columns([2, 1, 2])
with mid_btn:
    run_clicked = st.button("📈 예측 실행", key="forecast_btn", use_container_width=True)
    if run_clicked:
        st.session_state.run_forecast = True

if data is not None:
    st.markdown(f"**{data_label} 데이터 분석** • {len(data)}개월 데이터")
    
    st.markdown("---")
    
    # 기본 통계 + 최근 데이터 나란히
    st.markdown("### 📈 데이터 통계")
    
    left_col, right_col = st.columns([1, 1], gap="medium")
    
    with left_col:
        s1, s2 = st.columns(2, gap="small")
        with s1:
            st.metric(
                "최근 값",
                f"₩{data[column_name].iloc[-1]/1e6:.2f}M",
                f"{((data[column_name].iloc[-1] / data[column_name].iloc[-12] - 1) * 100):+.1f}% (YoY)" if len(data) >= 12 else None
            )
        with s2:
            st.metric(
                "평균",
                f"₩{data[column_name].mean()/1e6:.2f}M",
                "월별 평균"
            )
        # 최고/최저 간격 추가
        st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)
        s3, s4 = st.columns(2, gap="small")
        with s3:
            max_idx = data[column_name].idxmax()
            st.metric(
                "최고",
                f"₩{data[column_name].max()/1e6:.2f}M",
                f"{max_idx.strftime('%Y.%m')}"
            )
        with s4:
            min_idx = data[column_name].idxmin()
            st.metric(
                "최저",
                f"₩{data[column_name].min()/1e6:.2f}M",
                f"{min_idx.strftime('%Y.%m')}"
            )
    
    with right_col:
        st.markdown("""
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;'>
            <span style='font-size:15px; font-weight:700;'>📋 최근 데이터</span>
            <span style='font-size:11px; color:#888; font-weight:500;'>단위: 원 (KRW)</span>
        </div>
        """, unsafe_allow_html=True)
        recent_df = data.tail(12).iloc[::-1].copy()
        recent_df.index = recent_df.index.strftime('%Y.%m')
        recent_df.index.name = '날짜'
        st.dataframe(
            recent_df.style.format('{:,.0f}'),
            use_container_width=True,
            height=213,
            hide_index=False
        )
    
    st.markdown("---")
    
    # 기본 그래프
    st.markdown("### 📊 과거 데이터 추세")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[column_name],
        name='실제 데이터',
        mode='lines',
        line=dict(color='#1f2937', width=2),
        hovertemplate='<b>%{x|%Y.%m}</b><br>₩%{y:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'<b>{data_label} 데이터 추세</b>',
        xaxis_title='날짜',
        yaxis_title='금액 (원)',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',
        margin=dict(l=80, r=80, t=80, b=80),
        font=dict(size=11, color='#1a1a1a'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor='#e8ebed', borderwidth=1)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ==================== 예측 결과 ====================
    if getattr(st.session_state, 'run_forecast', False):
        with st.spinner("모델 학습 중..."):
            try:
                order = (p, d, q)
                seasonal_order = (P, D, Q, 12)
                
                model = SARIMAX(
                    data[column_name],
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
                
                last_date = data.index[-1]
                forecast_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=forecast_months,
                    freq='MS'
                )
                
                # 평가지표
                actual_values = data[column_name].values
                fitted_for_eval = fitted_values.dropna().values
                actual_for_eval = data[column_name].iloc[len(data) - len(fitted_for_eval):].values
                
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
                    st.metric("MAD", f"₩{mad/1e6:.2f}M", "평균 오차")
                
                with metric_col3:
                    st.metric("RMSE", f"₩{rmse/1e6:.2f}M", "제곱근 오차")
                
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
                
                # 실제 데이터 (기존 유지)
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[column_name],
                    name='실제 데이터',
                    mode='lines',
                    line=dict(color='#1f2937', width=2),
                    hovertemplate='<b>실제</b><br>%{x|%Y.%m}<br>₩%{y:,.0f}<extra></extra>'
                ))
                
                # 적합 데이터 - 더 촘촘한 dash, 음영 강화
                fig.add_trace(go.Scatter(
                    x=fitted_values.index,
                    y=fitted_values.values,
                    name='적합 데이터',
                    mode='lines',
                    line=dict(color='#6366f1', width=1.8, dash='dot'),
                    opacity=0.75,
                    hovertemplate='<b>적합</b><br>%{x|%Y.%m}<br>₩%{y:,.0f}<extra></extra>'
                ))
                
                # 실제 마지막 값 → 예측 첫 값 연결용 브릿지 데이터
                last_actual_date = data.index[-1]
                last_actual_val = float(data[column_name].iloc[-1])
                first_forecast_val = float(forecast_values.values[0])
                bridge_x = [last_actual_date, forecast_dates[0]]
                bridge_y = [last_actual_val, first_forecast_val]
                
                # 신뢰구간 상단 (브릿지 포함)
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
                
                # 신뢰구간 하단 + fill (음영 강화, 브릿지 포함)
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
                
                # 예측 데이터 - 빨간색 (브릿지 포함하여 연결)
                fig.add_trace(go.Scatter(
                    x=bridge_x + list(forecast_dates[1:]),
                    y=bridge_y + list(forecast_values.values[1:]),
                    name='예측 데이터',
                    mode='lines+markers',
                    line=dict(color='#ef4444', width=2.5),
                    marker=dict(size=6, color='#ef4444', line=dict(width=1, color='#ffffff')),
                    hovertemplate='<b>예측</b><br>%{x|%Y.%m}<br>₩%{y:,.0f}<extra></extra>'
                ))
                
                fig.update_layout(
                    title=f'<b>{data_label} 수요 예측 (SARIMA{order}×{seasonal_order})</b>',
                    xaxis_title='날짜',
                    yaxis_title='금액 (원)',
                    hovermode='x unified',
                    template='plotly_white',
                    height=500,
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff',
                    margin=dict(l=80, r=80, t=80, b=80),
                    font=dict(size=11, color='#1a1a1a'),
                    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor='#e8ebed', borderwidth=1)
                )
                
                fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f3f4f6')
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # 예측값 테이블
                st.markdown("### 📋 예측값")
                
                forecast_table = pd.DataFrame({
                    '날짜': forecast_dates.strftime('%Y.%m'),
                    '예측값': [f'₩{x:,.0f}' for x in forecast_values.values.astype(int)],
                    '하한': [f'₩{x:,.0f}' for x in confidence_intervals.iloc[:, 0].values.astype(int)],
                    '상한': [f'₩{x:,.0f}' for x in confidence_intervals.iloc[:, 1].values.astype(int)],
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
                    st.metric("데이터 기간", f"{len(data)}개월")
                
            except Exception as e:
                st.error(f"❌ 오류: {str(e)}")
                st.info("💡 파라미터를 조정해보세요.")

else:
    st.warning("⚠️ 데이터를 로드할 수 없습니다. 왼쪽 사이드바에서 데이터를 선택하세요.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #999999; font-size: 11px; padding: 8px 0;'>
    <p style='margin: 2px 0;'>SARIMA (Seasonal Autoregressive Integrated Moving Average) 기반 수요예측</p>
    <p style='margin: 2px 0;'>데이터 출처: 한국무역통계진흥원</p>
    <p style='margin: 2px 0;'>All Rights Reserved 인천대 동북아물류대학원 SCM Class</p>
</div>
""", unsafe_allow_html=True)
