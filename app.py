import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from google import genai
import dotenv
import os
dotenv.load_dotenv()  # Corrected function syntax

# Setup page layout
st.set_page_config(page_title="Agentic CPG Marketing Optimizer", layout="wide")
st.title("🎯 Agentic CPG Marketing Mix Optimizer")
st.subheader("Autonomous Media Allocation & Executive Strategy Agent")

# 1. Core Mathematical Modeling Engine (from our modeling script)
def apply_adstock(series, retention_rate=0.5):
    adstock = np.zeros(len(series))
    for t in range(len(series)):
        adstock[t] = series.iloc[t] if t == 0 else series.iloc[t] + retention_rate * adstock[t-1]
    return adstock

@st.cache_data
def load_and_model_data():
    df = pd.read_csv("synthetic_cpg_mmm_data.csv")
    df['tv_adstock'] = apply_adstock(df['tv_spend'], retention_rate=0.6)
    df['digital_adstock'] = apply_adstock(df['digital_spend'], retention_rate=0.2)
    df['social_adstock'] = apply_adstock(df['social_spend'], retention_rate=0.3)
    
    features = ['tv_adstock', 'digital_adstock', 'social_adstock']
    X = df[features]
    y = df['revenue']
    
    # We use Ridge directly on unscaled features here to preserve 
    # the realistic linear dollar-for-dollar relationship we generated
    model = Ridge(alpha=0.1)
    model.fit(X, y)
    
    unscaled_coefs = model.coef_
    return df, unscaled_coefs

df, unscaled_coefs = load_and_model_data()

# Calculate baseline metrics
channels = ['TV Advertising', 'Digital Search', 'Paid Social']
spends = [df['tv_spend'].sum(), df['digital_spend'].sum(), df['social_spend'].sum()]
rois = [(coef * spend) / spend for coef, spend in zip(unscaled_coefs, spends)]

# 2. Render UI Dashboard Cards (Fixed with exact array index numbers)
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="📺 TV Advertising ROI", value=f"{rois[0]:.2f}x", delta=f"${unscaled_coefs[0]:.2f} / $1")
with col2:
    st.metric(label="💻 Digital Search ROI", value=f"{rois[1]:.2f}x", delta=f"${unscaled_coefs[1]:.2f} / $1")
with col3:
    st.metric(label="📱 Paid Social ROI", value=f"{rois[2]:.2f}x", delta=f"${unscaled_coefs[2]:.2f} / $1")

st.divider()

# 3. Agentic AI Strategy Generator Window
st.write("### 🤖 Autonomous Strategy Agent")
st.write("Click below to invoke the AI Agent. The agent will read your ROI math and write an executive recommendation report.")

if st.button("Invoke Strategy Agent"):
    # Securely initialize client using standard local API key variables
    try:
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        
        # Build strict prompt containing data inputs
        prompt = f"""
        You are an elite enterprise Chief Marketing Officer (CMO) and Data Strategy Consultant. 
        Analyze these historical Marketing Mix Modeling (MMM) metrics for our CPG brand:
        
        - TV Advertising ROI: {rois[0]:.2f}x (Marginal return: ${unscaled_coefs[0]:.2f} per $1 spent)
        - Digital Search ROI: {rois[1]:.2f}x (Marginal return: ${unscaled_coefs[1]:.2f} per $1 spent)
        - Paid Social ROI: {rois[2]:.2f}x (Marginal return: ${unscaled_coefs[2]:.2f} per $1 spent)
        
        Draft a formal, concise Executive Briefing Memo (max 300 words). 
        Identify the highest and lowest performing channels based on diminishing returns. 
        Provide an explicit, data-driven recommendation on how to shift budgets across these channels 
        to maximize overall organizational revenue next quarter. Keep the tone executive and sharp.
        """
        
        with st.spinner("Agent parsing mathematical arrays and drafting memo..."):
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            st.success("Report Drafted Successfully!")
            st.markdown(response.text)
            
    except Exception as e:
        st.error("API Connection Error: Make sure your GEMINI_API_KEY environment variable is set up in your terminal environment.")
        st.caption(f"Error details: {e}")
