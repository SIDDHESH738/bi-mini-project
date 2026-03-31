import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Impact on Jobs Dashboard", layout="wide")

# Custom styling
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}
.css-1d391kg {
    background: rgba(15, 23, 42, 0.75);
    border-radius: 12px;
    box-shadow: 0 16px 35px rgba(0,0,0,0.35);
}
.stButton>button, .stSelectbox>div>div, .stSlider>div>div {
    border-radius: 12px;
}
h1 {
    color: #e2e8f0;
    background: linear-gradient(90deg,#7c3aed,#4f46e5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
h2, h3 {
    color: #f1f5f9;
}
.stMarkdown p, .stMarkdown li {
    color: #e2e8f0;
}
div[data-testid="stHorizontalBlock"] {
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(147, 197, 253, 0.25);
    border-radius: 12px;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown("""
# 🤖 AI Impact on Jobs Dashboard
**Comprehensive Business Intelligence Analysis (2010-2025)**

This interactive dashboard reconstructs the HTML implementation in Streamlit, including:
- dataset loading + overview metrics
- EDA charts (temporal, industry, region, salary, correlations)
- simulated classification/clustering insights
- advanced analytics and interactive filter behavior
- BI insights & recommendations
""", unsafe_allow_html=True)

# Data loading panel
with st.sidebar.expander("Load Dataset (CSV)", expanded=True):
    csv_file = st.file_uploader("Upload dataset (CSV)", type="csv")
    use_sample = st.checkbox("Use sample file ai_impact_jobs_2010_2025 (2).csv if available", value=True)


def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as ex:
        st.error(f"Unable to read CSV: {ex}")
        return None


df = None
if csv_file is not None:
    df = load_data(csv_file)
elif use_sample:
    try:
        df = pd.read_csv("ai_impact_jobs_2010_2025 (2).csv")
    except Exception:
        st.warning("Sample file not found in workspace. Please upload CSV.")

if df is None:
    df = pd.DataFrame(columns=["job_id", "posting_year", "country", "industry", "ai_mentioned", "ai_intensity_score", "automation_risk_score", "salary_usd", "ai_job_displacement_risk", "ai_skills"])

# Normalize dataset columns if present
if "posting_year" in df.columns:
    df["posting_year"] = pd.to_numeric(df["posting_year"], errors="coerce")
if "salary_usd" in df.columns:
    df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")

# derive numeric stats
records = len(df)
features = len(df.columns)
year_min = int(df["posting_year"].min()) if "posting_year" in df.columns and not df["posting_year"].dropna().empty else "N/A"
year_max = int(df["posting_year"].max()) if "posting_year" in df.columns and not df["posting_year"].dropna().empty else "N/A"
country_count = int(df["country"].nunique()) if "country" in df.columns else 0

with st.container():
    st.markdown(f"""
    <div style='display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px;'>
        <div style='background:linear-gradient(135deg, #4f46e5, #0ea5e9); border-radius:16px; padding:18px; width:240px; box-shadow:0 12px 20px rgba(15,23,42,0.35);'>
            <h4 style='margin:0;color:#e2e8f0;'>Total Records</h4>
            <p style='font-size:2rem; margin:6px 0; color:#ffffff;'>{records:,}</p>
        </div>
        <div style='background:linear-gradient(135deg, #10b981, #06b6d4); border-radius:16px; padding:18px; width:240px; box-shadow:0 12px 20px rgba(15,23,42,0.35);'>
            <h4 style='margin:0;color:#e2e8f0;'>Features</h4>
            <p style='font-size:2rem; margin:6px 0; color:#ffffff;'>{features}</p>
        </div>
        <div style='background:linear-gradient(135deg, #f59e0b, #f97316); border-radius:16px; padding:18px; width:240px; box-shadow:0 12px 20px rgba(15,23,42,0.35);'>
            <h4 style='margin:0;color:#e2e8f0;'>Time Period</h4>
            <p style='font-size:2rem; margin:6px 0; color:#ffffff;'>{year_min} - {year_max}</p>
        </div>
        <div style='background:linear-gradient(135deg, #ef4444, #f43f5e); border-radius:16px; padding:18px; width:240px; box-shadow:0 12px 20px rgba(15,23,42,0.35);'>
            <h4 style='margin:0;color:#e2e8f0;'>Countries</h4>
            <p style='font-size:2rem; margin:6px 0; color:#ffffff;'>{country_count}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Initialize tabs replicating nav sections
tabs = st.tabs(["Project Overview", "Dataset Overview", "Preprocessing & EDA", "Classification", "Clustering", "Advanced Analytics", "Interactive Dashboard", "BI Insights"])

# Tab 1: Project Overview
with tabs[0]:
    st.subheader("Project Overview")
    st.markdown("""
    **AI Impact on Jobs: A Comprehensive Data Mining and Business Intelligence Analysis (2010-2025)**

    This project analyzes the influence of AI on the job market using classification and clustering methods, and provides actionable insights for workforce planning.
    """)
    st.markdown("**Key Objectives:**")
    st.markdown("- Trends across industries/regions\n- Displacement risk classification\n- Cluster job market segments\n- Salary & AI skill analytics\n- Strategic BI recommendations")

# Tab 2: Dataset
with tabs[1]:
    st.subheader("Dataset Overview")
    st.write("Rows:", records)
    st.write("Columns:", features)
    st.write("Year range:", f"{year_min} - {year_max}")
    st.write("Countries:", country_count)

    st.write("### Dataset Head")
    st.dataframe(df.head(10), use_container_width=True)

    if records > 0:
        st.write("### Key Profile")
        st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

# Helper chart defaults
industry_names_default = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Education', 'Consulting']
industry_jobs_default = [3200, 2100, 1800, 1600, 1400, 1200, 1000]
region_names_default = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East & Africa']
region_jobs_default = [35, 28, 22, 10, 5]

# Helper chart data (static fallback if real data missing)
if records == 0:
    year_labels = list(range(2010, 2026))
    posts = [850, 920, 1050, 1180, 1320, 1480, 1650, 1850, 2100, 2400, 2800, 3200, 3800, 4200, 4800, 5200]
    ai_rate = [5, 7, 9, 12, 16, 21, 28, 35, 42, 48, 55, 62, 68, 73, 78, 82]
    industry_names = industry_names_default
    industry_jobs = industry_jobs_default
    region_names = region_names_default
    region_jobs = region_jobs_default
    salary_bins = ['$20K-30K', '$30K-40K', '$40K-50K', '$50K-60K', '$60K-70K', '$70K-80K', '$80K-90K', '$90K-100K', '$100K-120K', '$120K-150K', '$150K+']
    salary_count = [850, 1200, 1800, 2200, 2800, 2400, 1900, 1500, 1200, 800, 450]
    corr_labels = ['AI Intensity vs Salary', 'Automation Risk vs Salary', 'AI Intensity vs Risk', 'Experience vs Salary']
    corr_vals = [0.65, -0.45, 0.72, 0.58]
else:
    # derive from data
    year_labels = sorted(df['posting_year'].dropna().astype(int).unique().tolist()) if 'posting_year' in df.columns else list(range(2010, 2026))
    posts_group = df.groupby('posting_year').size() if 'posting_year' in df.columns else None
    posts = [int(posts_group.loc[y]) if posts_group is not None and y in posts_group.index else 0 for y in year_labels]
    ai_rate = []
    if 'ai_intensity_score' in df.columns:
        yearly_ai = df.groupby('posting_year')['ai_intensity_score'].mean().reindex(year_labels, fill_value=np.nan)
        ai_rate = [float(round(v*100,1)) if not np.isnan(v) else 0 for v in yearly_ai]
    else:
        ai_rate = [5, 7, 9, 12, 16, 21, 28, 35, 42, 48, 55, 62, 68, 73, 78, 82][:len(year_labels)]
    industry_names = df['industry'].value_counts().index[:7].tolist() if 'industry' in df.columns and not df['industry'].isna().all() else industry_names_default
    industry_jobs = df['industry'].value_counts().iloc[:7].tolist() if 'industry' in df.columns and not df['industry'].isna().all() else industry_jobs_default
    region_names = df['country'].value_counts().index[:5].tolist() if 'country' in df.columns and not df['country'].isna().all() else region_names_default
    region_jobs = df['country'].value_counts().iloc[:5].tolist() if 'country' in df.columns and not df['country'].isna().all() else region_jobs_default
    salary_bins = ['$20K-30K', '$30K-40K', '$40K-50K', '$50K-60K', '$60K-70K', '$70K-80K', '$80K-90K', '$90K-100K', '$100K-120K', '$120K-150K', '$150K+']
    salary_count = []
    if 'salary_usd' in df.columns and not df['salary_usd'].isna().all():
        salary_count = [
            int(((df['salary_usd'] >= 20000) & (df['salary_usd'] < 30000)).sum()),
            int(((df['salary_usd'] >= 30000) & (df['salary_usd'] < 40000)).sum()),
            int(((df['salary_usd'] >= 40000) & (df['salary_usd'] < 50000)).sum()),
            int(((df['salary_usd'] >= 50000) & (df['salary_usd'] < 60000)).sum()),
            int(((df['salary_usd'] >= 60000) & (df['salary_usd'] < 70000)).sum()),
            int(((df['salary_usd'] >= 70000) & (df['salary_usd'] < 80000)).sum()),
            int(((df['salary_usd'] >= 80000) & (df['salary_usd'] < 90000)).sum()),
            int(((df['salary_usd'] >= 90000) & (df['salary_usd'] < 100000)).sum()),
            int(((df['salary_usd'] >= 100000) & (df['salary_usd'] < 120000)).sum()),
            int(((df['salary_usd'] >= 120000) & (df['salary_usd'] < 150000)).sum()),
            int((df['salary_usd'] >= 150000).sum())
        ]
    else:
        salary_count = [850,1200,1800,2200,2800,2400,1900,1500,1200,800,450]
    corr_labels = ['AI Intensity vs Salary', 'Automation Risk vs Salary', 'AI Intensity vs Risk', 'Experience vs Salary']
    corr_vals = [0.65, -0.45, 0.72, 0.58]

# tab 3: EDA
with tabs[2]:
    st.subheader("Preprocessing & Exploratory Data Analysis")
    col1, col2 = st.columns(2)
    fig_temporal = px.line(x=year_labels, y=posts, labels={"x":"Year","y":"Job Postings"}, title="Job Postings Over Time")
    col1.plotly_chart(fig_temporal, use_container_width=True)

    fig_ai_adopt = px.line(x=year_labels, y=ai_rate, labels={"x":"Year","y":"AI Adoption Rate (%)"}, title="AI Adoption Rate Trend")
    col2.plotly_chart(fig_ai_adopt, use_container_width=True)

    col3, col4 = st.columns(2)
    fig_industry = px.bar(x=industry_names, y=industry_jobs, labels={"x":"Industry","y":"Job Count"}, title="Top Industries by Job Count")
    col3.plotly_chart(fig_industry, use_container_width=True)

    fig_region = px.pie(names=region_names, values=region_jobs, title="Job Distribution by Region")
    col4.plotly_chart(fig_region, use_container_width=True)

    st.write("### Salary Distribution")
    fig_salary = px.bar(x=salary_bins, y=salary_count, labels={"x":"Salary Range","y":"Job Count"}, title="Salary Distribution Across All Jobs")
    st.plotly_chart(fig_salary, use_container_width=True)

    st.write("### Correlation Analysis")
    fig_corr = px.bar(x=corr_labels, y=corr_vals, labels={"x":"Variable Pair","y":"Correlation"}, title="Key Feature Correlations")
    st.plotly_chart(fig_corr, use_container_width=True)

# Tab 4: Classification (simulated metrics)
with tabs[3]:
    st.subheader("Classification Analysis")
    st.metric("Accuracy", "87.5%", delta=None)
    st.metric("Precision", "86.2%", delta=None)
    st.metric("Recall", "87.8%", delta=None)
    st.metric("F1-Score", "87.0%", delta=None)

    # Confusion Matrix bar chart
    cm = np.array([[2450,95,65],[120,2100,180],[80,150,1760]])
    fig_cm = go.Figure(data=[
        go.Bar(name='Low (Actual)', x=['Low Pred','Medium Pred','High Pred'], y=cm[0]),
        go.Bar(name='Medium (Actual)', x=['Low Pred','Medium Pred','High Pred'], y=cm[1]),
        go.Bar(name='High (Actual)', x=['Low Pred','Medium Pred','High Pred'], y=cm[2])
    ])
    fig_cm.update_layout(barmode='group', title='Confusion Matrix - Random Forest (simulated)')
    st.plotly_chart(fig_cm, use_container_width=True)

    fig_fi = go.Figure(go.Bar(x=[35.2,28.7,18.4,12.1,4.8,0.8], y=['Automation Risk Score','AI Intensity Score','Annual Salary (USD)','Industry Type','Years of Experience','Company Size'], orientation='h'))
    fig_fi.update_layout(title='Feature Importance (simulated)')
    st.plotly_chart(fig_fi, use_container_width=True)

# Tab 5: Clustering
with tabs[4]:
    st.subheader("Clustering Analysis")
    st.metric("Optimal Clusters", "4")
    st.metric("Silhouette Score", "0.72")
    st.metric("Davies-Bouldin Index", "0.28")

    elbow_x = list(range(2,11))
    elbow_y = [45000,32000,25000,22000,20000,18500,17500,17000,16800]
    fig_elbow = px.line(x=elbow_x, y=elbow_y, labels={'x':'k','y':'WCSS'}, title='Elbow Method - Optimal Clusters')
    st.plotly_chart(fig_elbow, use_container_width=True)

    # scatter cluster sample
    np.random.seed(42)
    cluster_points = pd.DataFrame({
        'x': np.concatenate([np.random.rand(50)*0.3, np.random.rand(50)*0.4+0.3, np.random.rand(50)*0.3+0.6, np.random.rand(50)*0.3+0.7]),
        'y': np.concatenate([np.random.rand(50)*0.4, np.random.rand(50)*0.4+0.2, np.random.rand(50)*0.4+0.4, np.random.rand(50)*0.4+0.6]),
        'cluster': ['Cluster 0']*50 + ['Cluster 1']*50 + ['Cluster 2']*50 + ['Cluster 3']*50
    })
    fig_cluster = px.scatter(cluster_points, x='x', y='y', color='cluster', title='Job Clusters: AI Intensity vs Automation Risk')
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.write("### Cluster Profiles")
    st.table(
        pd.DataFrame({
            'Cluster': ['Cluster 0','Cluster 1','Cluster 2','Cluster 3'],
            'Size': [3250,4100,3800,3850],
            'Avg AI Intensity': [0.15,0.45,0.75,0.9],
            'Avg Automation Risk': [0.25,0.35,0.55,0.75],
            'Avg Salary': [65000,85000,110000,135000]
        })
    )

# Tab 6: Advanced Analytics
with tabs[5]:
    st.subheader("Advanced Analytics")
    seniority = pd.DataFrame({
        'Level':['Junior','Mid','Senior','Lead','Executive'],
        'Avg Salary':[45000,65000,85000,110000,150000],
        'AI Intensity':[0.2,0.35,0.55,0.75,0.85]
    })
    fig1 = px.bar(seniority, x='Level', y='Avg Salary', title='Average Salary by Seniority Level')
    fig2 = px.bar(seniority, x='Level', y='AI Intensity', title='AI Intensity by Seniority Level', range_y=[0,1])
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig2, use_container_width=True)

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=['Startup','Small','Medium','Large','Enterprise'], y=[25,35,45,55,65], name='AI Adoption Rate (%)'))
    fig_comp.add_trace(go.Bar(x=['Startup','Small','Medium','Large','Enterprise'], y=[55000,65000,75000,85000,95000], name='Average Salary ($)'))
    fig_comp.update_layout(barmode='group', title='AI Adoption & Salary by Company Size')
    st.plotly_chart(fig_comp, use_container_width=True)

    fig_adopt = px.pie(names=['Emerging','Growing','Mature'], values=[35,45,20], title='Industry AI Adoption Stages')
    fig_stage = px.bar(x=['Emerging','Growing','Mature'], y=[65000,85000,105000], title='Average Salary by AI Adoption Stage')
    st.plotly_chart(fig_adopt, use_container_width=True)
    st.plotly_chart(fig_stage, use_container_width=True)

    ai_skill_names = ['Machine Learning','Python Programming','Data Analysis','NLP','Computer Vision','Deep Learning','TensorFlow','PyTorch','Cloud Computing','Statistics']
    ai_skill_freq = [1250,1180,980,750,680,620,580,520,480,420]
    fig_skill = px.bar(x=ai_skill_freq, y=ai_skill_names, orientation='h', title='Top 10 AI Skills in Job Postings')
    st.plotly_chart(fig_skill, use_container_width=True)

# Tab 7: Interactive Dashboard
with tabs[6]:
    st.subheader("Interactive Dashboard")
    col1, col2, col3, col4 = st.columns(4)

    year_select = st.slider("Year (start)", 2010, 2025, 2010)
    industry_filter = st.selectbox("Industry", ['All', 'Technology', 'Finance', 'Healthcare', 'Retail'])
    risk_filter = st.selectbox("Risk Level", ['All', 'Low', 'Medium', 'High'])

    # Simulated filter metrics
    filtered_jobs = 15000
    ai_rate_metric = 42.5
    avg_salary_metric = 78500
    high_risk_metric = 28.5

    if industry_filter == 'Technology':
        ai_rate_metric, avg_salary_metric, high_risk_metric = 68.5, 95000, 35.2
    elif industry_filter == 'Finance':
        ai_rate_metric, avg_salary_metric, high_risk_metric = 55.8, 88000, 32.1

    if risk_filter == 'High':
        filtered_jobs = int(filtered_jobs * 0.285)
        ai_rate_metric = 75.2

    col1.metric("Filtered Jobs", f"{filtered_jobs:,}")
    col2.metric("AI Adoption Rate", f"{ai_rate_metric}%")
    col3.metric("Average Salary", f"${avg_salary_metric:,.0f}")
    col4.metric("High Risk Jobs", f"{high_risk_metric}%")

    trend = [int(v * (filtered_jobs / 15000)) for v in posts] if posts else [0]*len(year_labels)
    fig_trend = px.line(x=year_labels, y=trend, title='Filtered Job Posting Trends', labels={'x':'Year','y':'Job Postings'})
    st.plotly_chart(fig_trend, use_container_width=True)

    low_risk = max(0, 100 - high_risk_metric - 25)
    risk_data = pd.DataFrame({'Risk':['Low','Medium','High'], 'Pct':[low_risk, 25, high_risk_metric]})
    fig_risk = px.pie(risk_data, names='Risk', values='Pct', title='Displacement Risk Distribution')
    st.plotly_chart(fig_risk, use_container_width=True)

# Tab 8: BI Insights
with tabs[7]:
    st.subheader("Business Intelligence Insights & Recommendations")
    st.write("Based on the analysis (15,000+ jobs), high-level impact metrics:")
    st.metric("Jobs at High Risk", "28.5%")
    st.metric("AI Adoption Growth", "+156%")
    st.metric("Jobs Need Reskilling", "35.2%")

    st.markdown("### Strategic Recommendations")
    st.markdown("**For Organizations & HR Leaders**")
    st.markdown("- Prioritize reskilling programs\n- Strategic hiring for AI talent\n- Focus risk mitigation roles >0.7\n- Employee transition support")

    st.markdown("**For Policymakers & Government**")
    st.markdown("- Government-funded AI training initiatives\n- Education curriculum reform\n- Transition safety nets for workers\n- Industry-academic partnerships")

    st.markdown("**For Job Seekers**")
    st.markdown("- Build AI/ML/data skills\n- Explore AI-intensive industries\n- Do continuous learning\n- Network with AI professionals")
