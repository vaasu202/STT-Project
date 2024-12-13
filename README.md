# STT-Project

This repository contains a Bayesian Network-based classifier for predicting snowfall in the city of Lansing using weather data. The project uses Python libraries for data preprocessing, feature engineering, structure learning, and evaluation. 

## Features
- Data preprocessing with statistical distribution and outlier analysis
- Feature engineering with mutual information and discretization
- Bayesian Network structure learning using `HillClimbSearch` and `BicScore`
- Model fitting with Maximum Likelihood Estimation
- Inference and evaluation with various metrics (accuracy, precision, recall)
---
## Installation
To run this project, you need the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `pgmpy`
- `matplotlib`
- `seaborn`
- `networkx`
- `scipy`
---
## Dataset Overview
We make use of the this dataset - https://drive.google.com/file/d/1z5pfD2vkX7N6JoVUCyytq_utVs85GlI-/view?usp=sharing. The dataset is provided in Parquet format (`daily_weather.parquet`) and contains daily weather observations. Key features include:
- **`season`**: The season during which the observation was recorded (e.g., "winter").
- **`snow_depth_mm`**: The depth of snow measured in millimeters.
- **`city_name`**, **`date`**, and **`station_id`**: Metadata associated with the observations.
- Additional numerical weather-related variables such as temperature, humidity, wind speed, and precipitation.

The target variable `snow` is derived as:
- `1`: If `snow_depth_mm > 0`.
- `0`: Otherwise.
---
## Code Overview
The main script is divided into the following steps:
1. **Data Loading**: Reads the dataset and analyzes statistical properties.
2. **Data Preprocessing**: Filters for relevant records and balances classes.
3. **Feature Engineering**: Discretizes continuous features for Bayesian modeling.
4. **Bayesian Network Construction**:
   - Learns structure.
   - Fits parameters.
5. **Inference and Evaluation**:
   - Predicts snow probabilities.
   - Evaluates model performance with metrics and visualizations.

---
## Key Functions
1. **`load_data(parquet_file_path)`**  
   - Loads the Parquet dataset and performs statistical analysis:
     - Normality tests for numerical features.
     - Missing value handling.
     - Outlier detection using `IsolationForest`.

2. **`preprocess_data(df)`**  
   - Filters the dataset for winter observations.
   - Balances class distribution of the target variable (`snow`) through undersampling.
   - Performs chi-square tests to assess feature independence with the target.

3. **`feature_engineering_discrete(df, n_bins=5)`**  
   - Discretizes continuous variables into `n_bins` using `KBinsDiscretizer`.
   - Ranks features by mutual information scores.

4. **`build_bayesian_network_discrete(X_train, y_train)`**  
   - Uses `HillClimbSearch` and `BicScore` to learn the Bayesian Network structure.
   - Visualizes the Bayesian Network graph.

5. **`fit_bayesian_network(model, data)`**  
   - Fits the Bayesian Network parameters using Maximum Likelihood Estimation (MLE).

6. **`predict_bayesian_network(model, X_test)`**  
   - Performs inference on test data using the Bayesian Network.
   - Computes probabilities and makes predictions.

7. **`evaluate_model(y_test, y_pred, y_pred_proba)`**  
   - Evaluates model performance using:
     - Accuracy, precision, recall.
     - Confusion matrix.

---
## Usage
1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/yourusername/bayesian-snow-prediction.git
   cd bayesian-snow-prediction
