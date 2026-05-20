import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(weeks=i) for i in range(156)]

weeks = np.arange(156)
trend = 10000 + (weeks * 25)  
seasonality = 1 + 0.15 * np.sin(2 * np.pi * weeks / 52)  
holiday_spike = np.where((weeks % 52 >= 48) | (weeks % 52 <= 2), 1.25, 1.0) 

tv_spend = np.random.gamma(shape=5, scale=2000, size=156)      
digital_spend = np.random.normal(loc=5000, scale=1200, size=156) 
social_spend = np.random.exponential(scale=3000, size=156)    
social_spend = np.where(social_spend < 1000, social_spend * 2, social_spend) 

def apply_adstock(spend, retention_rate=0.5):
    adstock = np.zeros_like(spend)
    for t in range(len(spend)):
        if t == 0:
            adstock[t] = spend[t]
        else:
            adstock[t] = spend[t] + retention_rate * adstock[t-1]
    return adstock

tv_adstock = apply_adstock(tv_spend, retention_rate=0.6)  
digital_adstock = apply_adstock(digital_spend, retention_rate=0.2)
social_adstock = apply_adstock(social_spend, retention_rate=0.3)

# AMPLIFIED MULTIPLIERS: Boost linear scale factors to simulate high-return environments
tv_impact = 1.8 * tv_adstock
digital_impact = 3.4 * digital_adstock
social_impact = 1.2 * social_adstock

base_revenue = trend * seasonality * holiday_spike
noise = np.random.normal(loc=0, scale=1500, size=156)
total_revenue = base_revenue + tv_impact + digital_impact + social_impact + noise

df = pd.DataFrame({
    'week_start': dates,
    'tv_spend': np.round(tv_spend, 2),
    'digital_spend': np.round(digital_spend, 2),
    'social_spend': np.round(social_spend, 2),
    'revenue': np.round(total_revenue, 2)
})

df.to_csv('synthetic_cpg_mmm_data.csv', index=False)
print("✨ High-ROI dataset regenerated successfully!")