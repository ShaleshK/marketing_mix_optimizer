# 🎯 Agentic CPG Marketing Mix Optimizer

An end-to-end data engineering and generative AI application that builds an enterprise Marketing Mix Model (MMM) and leverages an autonomous LLM agent to deliver executive budget reallocation strategies.

## 📊 Business Case & Core Features
Corporate marketing teams often struggle to isolate the true return on investment (ROI) across multi-million dollar ad flights due to lag effects and diminishing returns. This project builds a realistic data framework to solve that problem:
- **Data Pipeline**: Simulates 3 years of weekly CPG media data incorporating adstock carryover, saturation decay curves, and holiday spikes.
- **Statistical Modeling Engine**: Fits a Regularized Ridge Regression model to attribute marginal revenue gains accurately across TV, Digital, and Paid Social spend.
- **Agentic Strategy Framework**: Integrates the Google GenAI SDK (`gemini-2.5-flash`) to parse raw mathematical coefficients and auto-generate executive briefing memos.
- **Interactive UI Panel**: Features a full-stack browser application built entirely via Streamlit.

## 🛠️ Local Execution Checklist

1. Clone this repository to your machine.
2. Install the modern project dependencies inside a Python 3.11 environment:
   ```bash
   pip install -r requirements.txt
   ```
3. Establish your hidden security credentials inside a root `.env` file:
   ```text
   GEMINI_API_KEY=your_actual_developer_api_key_here
   ```
4. Boot up the local application server stream:
   ```bash
   streamlit run app.py