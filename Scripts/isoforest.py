from sklearn.preprocessing import MinMaxScaler  # For feature normalization
from sklearn.ensemble import IsolationForest   # For anomaly detection
import seaborn as sns                # For data visualization (pairplot)
import matplotlib.pyplot as plt      # For saving plots

def load_category():
    try:
        with open("chosen_category.txt", "r") as f:
            category = f.read().strip()  # Strip any extra whitespace
        return category
    except FileNotFoundError:
        print("Error: 'chosen_category.txt' not found. Ensure the category is selected first.")
        return None
category = load_category()

product_df = product_df[product_df['review_count'] >= 15]
print(f"Filtered to products w/ reviews > 15. Number of products in the filtered DataFrame: {len(df)}")
# Step 1: Prepare features for the model
features = [
    'avg_rating', 'review_count', 'avg_sentiment_score',
    'safety_term_density', 'incident_review_match_rate']
X = product_df[features]
# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train the Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,  # Number of trees
    contamination=0.0005,  # Proportion of outliers (adjust as needed)
    random_state=42)
product_df['anomaly_score'] = iso_forest.fit_predict(X_scaled)
# Flag anomalies
# Isolation Forest predicts -1 for anomalies and 1 for inliers
product_df['is_outlier'] = product_df['anomaly_score'] == -1

# Count flagged outliers
num_outliers = product_df['is_outlier'].sum()
print(f"Number of outliers detected: {num_outliers}")

# Filter flagged anomalies for inspection
outliers = product_df[product_df['is_outlier']]
# Save each category’s output CSV
isolation_file_name = f'isolationforest_unsafe_products_{category.replace(" ", "_")}.csv'
outliers.to_csv(isolation_file_name, index=False)
print(f'Completed and saved IsolationForest Outliers for category: {category}')
# Features to plot
features_to_plot = [
    'avg_rating', 'review_count', 'safety_term_count',
    'safety_term_density', 'incident_review_match_rate']
# Ensure no NaN values in 'is_outlier'
product_df['is_outlier'] = product_df['is_outlier'].fillna(False)
# Replace inf values with NaN
product_df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Drop rows with NaN in the features_to_plot
product_df.dropna(subset=features_to_plot, inplace=True)
# Pairplot with hue for outliers
pairplot = sns.pairplot(
    product_df,
    vars=features_to_plot,
    hue='is_outlier',  # Highlight outliers
    palette={True: 'red', False: 'blue'},
    diag_kind='kde',
    plot_kws={'alpha': 0.6})
pairplot.fig.suptitle(f'Pairwise Feature Comparison for {category} with IsolationForest Outliers Highlighted', size=16, y=1.02)
# Save the plot as a JPEG file
isolation_pairplot = f'isolationforest_pairplot_products_{category.replace(" ", "_")}.jpeg'
pairplot.fig.savefig(isolation_pairplot, format='jpeg', dpi=300)
print(f'Completed and saved IsolationForest Outliers for category: {category} as {category_file_name}')