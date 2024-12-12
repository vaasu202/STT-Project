import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import networkx as nx
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from scipy.stats import entropy, chi2_contingency, spearmanr
from sklearn.ensemble import IsolationForest

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Advanced Statistical Logging Decorator
def statistical_logger(func):
    def wrapper(*args, **kwargs):
        print(f"\n--- Statistical Analysis for {func.__name__} ---")
        result = func(*args, **kwargs)
        return result
    return wrapper

# Step 1: Load the dataset (Enhanced with Statistical Validation)
@statistical_logger
def load_data(parquet_file_path):
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        
        # Statistical Distribution Analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Handle missing values before statistical analysis
        # Fill missing numeric values with median
        df_cleaned = df.copy()
        for column in numeric_columns:
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
        
        distribution_report = {}
        
        for column in numeric_columns:
            # Normality Test
            _, p_value = stats.normaltest(df_cleaned[column])
            distribution_report[column] = {
                'mean': df_cleaned[column].mean(),
                'median': df_cleaned[column].median(),
                'std': df_cleaned[column].std(),
                'is_normal_distribution': p_value < 0.05,
                'p_value': p_value,
                'missing_original': df[column].isna().sum(),
                'missing_after_cleaning': df_cleaned[column].isna().sum()
            }
        
        # Print Distribution Report
        print("\nStatistical Distribution Report:")
        for col, stats_dict in distribution_report.items():
            print(f"{col}:")
            for key, value in stats_dict.items():
                print(f"  {key}: {value}")
        
        # Outlier Detection using Isolation Forest
        clf = IsolationForest(contamination=0.1, random_state=42)
        outliers = clf.fit_predict(df_cleaned[numeric_columns])
        outlier_percentage = np.mean(outliers == -1) * 100
        print(f"\nOutlier Detection: {outlier_percentage:.2f}% potential outliers")
        
        return df_cleaned
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Step 2: Filter for Lansing city and winter season, then create target variable
@statistical_logger
def preprocess_data(df):
    # Filter for Lansing and winter season
    df = df[df['season'].str.lower() == 'winter'].copy()
    
    print(f"Filtered DataFrame shape: {df.shape}")
    
    # Create target variable: 1 if snow_depth_mm > 0, else 0
    df['snow'] = df['snow_depth_mm'].apply(lambda x: 1 if x > 0 else 0)
    
    # Drop original snow_depth_mm and irrelevant columns (do NOT drop precipitation_mm)
    df = df.drop(['city_name', 'date', 'snow_depth_mm', 'season', 'station_id'], axis=1)
    
    # Remove rows with any missing values
    initial_shape = df.shape
    df = df.dropna()
    final_shape = df.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} rows due to missing values.")
    
    # Class Distribution Analysis
    class_counts = df['snow'].value_counts()
    class_proportions = df['snow'].value_counts(normalize=True)
    
    print("Original Class Distribution Analysis:")
    print("Raw Counts:\n", class_counts)
    print("\nProportions:\n", class_proportions)
    
    # Balance classes by reducing majority class
    minority_class_count = class_counts.min()
    majority_class = class_counts.idxmax()
    
    # Filter dataframe to balance classes
    balanced_df = pd.concat([
        df[df['snow'] == majority_class].sample(n=minority_class_count, random_state=42),
        df[df['snow'] != majority_class]
    ])
    
    # Verify balanced distribution
    balanced_class_counts = balanced_df['snow'].value_counts()
    balanced_class_proportions = balanced_df['snow'].value_counts(normalize=True)
    
    print("\nBalanced Class Distribution Analysis:")
    print("Raw Counts:\n", balanced_class_counts)
    print("\nProportions:\n", balanced_class_proportions)
    
    # Chi-Square Test of Independence
    feature_columns = balanced_df.columns.drop('snow')
    chi_square_results = {}
    
    for col in feature_columns:
        contingency_table = pd.crosstab(balanced_df[col], balanced_df['snow'])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        chi_square_results[col] = {
            'chi2_statistic': chi2,
            'p_value': p_value
        }
    
    print("\nChi-Square Feature Significance:")
    for feature, result in chi_square_results.items():
        print(f"{feature}: Chi2 = {result['chi2_statistic']:.4f}")
    
    return balanced_df

# Step 3: Feature Engineering - Discretization (Enhanced with Mutual Information)
@statistical_logger
def feature_engineering_discrete(df, n_bins=5):
    features = df.drop('snow', axis=1)
    target = df['snow']
    
    # Mutual Information Feature Selection
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector.fit(features, target)
    
    # Print Mutual Information Scores
    mi_scores = pd.Series(selector.scores_, index=features.columns)
    print("Mutual Information Scores:")
    print(mi_scores.sort_values(ascending=False))
    
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    discretized = discretizer.fit_transform(features)
    df_discrete = pd.DataFrame(discretized, columns=features.columns)
    df_discrete['snow'] = target.values
    
    print(f"Discretized features into {n_bins} bins each.")
    return df_discrete

# Optional: Visualize Discretization
def visualize_discretization(df_discrete):
    plt.figure(figsize=(12, 8))
    sns.histplot(data=df_discrete, x='snow', hue='snow', multiple='stack')
    plt.title('Distribution of Snow Classes')
    plt.show()

# Step 4: Split the data
def split_data(df, test_size=0.2, random_state=42):
    X = df.drop('snow', axis=1)
    y = df['snow']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test

# Step 5: Build Discrete Bayesian Network
def build_bayesian_network_discrete(X_train, y_train):
    # Combine X and y for structure learning
    train_data = X_train.copy()
    train_data['snow'] = y_train
    
    # Initialize HillClimbSearch
    hc = HillClimbSearch(train_data)
    # Estimate the best model using BicScore as the scoring method
    try:
        best_model = hc.estimate(scoring_method=BicScore(train_data))
        print("Learned Bayesian Network Structure:")
        print(best_model.edges())
        
        # Ensure the model is a BayesianNetwork
        if not isinstance(best_model, BayesianNetwork):
            print("Converting to BayesianNetwork.")
            best_model = BayesianNetwork(best_model.edges())
        else:
            print("best_model is a BayesianNetwork.")
            # Visualize the Bayesian Network structure using networkx

        plt.figure(figsize=(10, 8))
        G = nx.DiGraph(best_model.edges())
        pos = nx.spring_layout(G)  # positions for all nodes
        nx.draw(G, pos, with_labels=True, node_color='lightblue', font_size=10, node_size=3000, arrowsize=20)
        plt.title('Bayesian Network Structure')
        plt.show()

    except Exception as e:
        print(f"Error during structure learning: {e}")
        raise
    
    return best_model

# Step 6: Fit the Bayesian Network
def fit_bayesian_network(model, data):
    try:
        model.fit(data, estimator=MaximumLikelihoodEstimator)
        print("Bayesian Network parameters estimated using MLE.")
        
        # Displaying conditional probability distributions (CPDs)
        for cpd in model.get_cpds():
            print(f"CPD for {cpd.variable}:\n{cpd}\n")
    except Exception as e:
        print(f"Error during model fitting: {e}")
        raise
    return model

# Step 7: Inference and Prediction
def predict_bayesian_network(model, X_test):
    # Prepare inference
    infer = VariableElimination(model)
    
    predictions = []
    y_pred_proba = []
    
    for index, row in X_test.iterrows():
        evidence = row.to_dict()
        try:
            q = infer.query(variables=['snow'], evidence=evidence, show_progress=False)
            prob_snow = q.values[1]  # Probability of snow=1
            prediction = 1 if prob_snow > 0.5 else 0
            predictions.append(prediction)
            y_pred_proba.append(prob_snow)
        except Exception as e:
            print(f"Error during inference for index {index}: {e}")
            # Default prediction in case of error
            predictions.append(0)
            y_pred_proba.append(0.0)
    
    return predictions, y_pred_proba

# Step 8: Evaluate the model
def evaluate_model(y_test, y_pred, y_pred_proba):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Main function to execute the steps
def main():
    # Replace 'daily_weather.parquet' with your actual parquet file path
    parquet_file_path = 'daily_weather.parquet'
    
    # Load data
    df = load_data(parquet_file_path)
    
    # Preprocess data
    df_discrete = preprocess_data(df)
    
    # Feature Engineering - Discretization
    df_discrete = feature_engineering_discrete(df_discrete, n_bins=5)
    
    # Optional: Visualize Discretization
    visualize_discretization(df_discrete)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df_discrete)
    
    # Build Bayesian Network
    model = build_bayesian_network_discrete(X_train, y_train)
    
    # Fit the model with train_data
    train_data = X_train.copy()
    train_data['snow'] = y_train
    model = fit_bayesian_network(model, train_data)
    
    # Predict
    y_pred, y_pred_proba = predict_bayesian_network(model, X_test)
    
    # Evaluate
    evaluate_model(y_test, y_pred, y_pred_proba)

if __name__ == "__main__":
    main()
