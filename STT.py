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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Step 1: Load the dataset
def load_data(parquet_file_path):
    try:
        df = pd.read_parquet(parquet_file_path)
        print(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# Step 2: Filter for Lansing city and winter season, then create target variable
def preprocess_data(df):
    # Filter for Lansing and winter season
    df = df[df['season'].str.lower() == 'winter'].copy()
    
    print(f"Filtered DataFrame shape: {df.shape}")
    
    # Create target variable: 1 if snow_depth_mm > 0, else 0
    df['snow'] = df['snow_depth_mm'].apply(lambda x: 1 if x > 0 else 0)
    
    # Drop original snow_depth_mm and irrelevant columns (do NOT drop precipitation_mm)
    df = df.drop(['city_name', 'date', 'snow_depth_mm', 'season', 'station_id', 'precipitation_mm'], axis=1)
    
    # Remove rows with any missing values
    initial_shape = df.shape
    df = df.dropna()
    final_shape = df.shape
    print(f"Removed {initial_shape[0] - final_shape[0]} rows due to missing values.")
    
    # Check class distribution
    class_counts = df['snow'].value_counts()
    print("Class distribution after cleaning:")
    print(class_counts)
    
    return df

# Step 3: Feature Engineering - Discretization
def feature_engineering_discrete(df, n_bins=5):
    features = df.drop('snow', axis=1)
    target = df['snow']
    
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
