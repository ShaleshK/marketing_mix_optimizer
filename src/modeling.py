import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def apply_adstock(series, retention_rate=0.5):
    """Computes advertising carryover/decay effect across weeks."""
    adstock = np.zeros(len(series))
    for t in range(len(series)):
        if t == 0:
            adstock[t] = series.iloc[t]
        else:
            adstock[t] = series.iloc[t] + retention_rate * adstock[t-1]
    return adstock

def run_marketing_mix_model():
    # 1. Load the data
    print("🚀 Loading synthetic enterprise data...")
    df = pd.read_csv("synthetic_cpg_mmm_data.csv")
    
    # 2. Feature Engineering: Apply realistic Adstock decay
    print("📊 Engineering media carryover (Adstock) features...")
    df['tv_adstock'] = apply_adstock(df['tv_spend'], retention_rate=0.6)
    df['digital_adstock'] = apply_adstock(df['digital_spend'], retention_rate=0.2)
    df['social_adstock'] = apply_adstock(df['social_spend'], retention_rate=0.3)
    
    # 3. Separate Features (X) and Target (y)
    features = ['tv_adstock', 'digital_adstock', 'social_adstock']
    X = df[features]
    y = df['revenue']
    
    # 4. Scale features (Critical for Ridge Regression stability)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. Fit Ridge Regression (Handles multicollinearity elegantly)
    model = Ridge(alpha=1.0)
    model.fit(X_scaled, y)
     # Unscale coefficients to make them interpretable as actual dollar returns
    unscaled_coefs = model.coef_ / scaler.scale_
    
    # 6. Calculate total estimated ROI per channel
    print("\n=============================================")
    print("🎯 MARKETING MIX MODEL RESULTS (ESTIMATED ROI)")
    print("=============================================")
    channels = ['TV Advertising', 'Digital Search', 'Paid Social']
    spends = [df['tv_spend'].sum(), df['digital_spend'].sum(), df['social_spend'].sum()]
    
    for channel, coef, spend in zip(channels, unscaled_coefs, spends):
        # Rough calculation of revenue driven vs money spent
        estimated_revenue_driven = coef * spend
        roi = estimated_revenue_driven / spend
        print(f"📈 {channel}:")
        print(f"   - Marginal Attribution Coef: ${coef:.2f} revenue per $1 spent")
        print(f"   - Estimated Channel ROI:     {roi:.2f}x")
    print("=============================================\n")

if __name__ == "__main__":
    run_marketing_mix_model()