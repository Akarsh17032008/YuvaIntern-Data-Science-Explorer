# ==============================================================================
# PROJECT: Data Science Report & Insights Presentation
# STAGE: Task 4 - Visualization Generation for Non-Technical Stakeholders
# ==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_mock_insight_visuals():
    """Generates mock visualizations for the executive presentation."""
    sns.set_theme(style="whitegrid")
    
    # Mock Insight 1: Delivery Time vs Churn
    plt.figure(figsize=(8, 5))
    mock_delivery_data = pd.DataFrame({
        'Delivery Time (Days)': ['1-2 Days', '3-5 Days', '5+ Days'],
        'Churn Rate (%)': [5, 12, 42]
    })
    sns.barplot(x='Delivery Time (Days)', y='Churn Rate (%)', data=mock_delivery_data, palette='Reds')
    plt.title("Insight 1: Late Deliveries Drastically Increase Churn Rate", fontsize=14)
    plt.ylabel("Percentage of Customers Churned")
    plt.show()

    print("Executive Visualizations Generated Successfully.")
    print("Actionable Recommendation: Target users in the '5+ Days' delivery bucket with retention offers.")

if __name__ == "__main__":
    generate_mock_insight_visuals()
