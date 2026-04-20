#  Laptop Price Prediction using Machine Learning
This project predicts laptop prices based on their specifications using multiple Machine Learning models. It includes data analysis, preprocessing, model training, evaluation, and comparison across different algorithms.


##  Project Overview
The goal of this project is to predict the price of a laptop using its hardware specifications and brand.
The project includes:
- Data loading and exploration  
- Exploratory Data Analysis (EDA)  
- Data preprocessing and encoding  
- Training multiple ML models  
- Model evaluation and comparison  
- Feature importance analysis  
- Actual vs predicted visualization  


##  Dataset
File used: `laptop_dataset.csv`
### Features:
- Brand  
- RAM (GB)  
- CPU, GPU, Storage, and other specifications (encoded)  
### Target:
- Price (continuous numeric value)

## Exploratory Data Analysis (EDA)

###  Price Distribution
- Histogram with KDE  
- Helps understand overall price distribution  

###  RAM vs Price
- Scatter plot  
- Shows strong positive correlation between RAM and price  

###  Brand vs Price
- Bar plot (average price per brand)  
- Helps compare pricing trends across brands  

###  Correlation Heatmap
- Displays relationships between numerical features  
##  Data Preprocessing
### Steps performed:
- Separate features and target:
y = df['Price']
X = df.drop('Price', axis=1)


## Model Evaluation
Models were evaluated using the following metrics:
- R² Score → Measures how well the model fits the data  
- MAE (Mean Absolute Error) → Average prediction error  
- RMSE (Root Mean Squared Error) → Penalizes large errors  


##  Feature Importance
Random Forest provides feature importance scores to understand the impact of each feature on price prediction.

### Key Findings:
- RAM is one of the most important features  
- Brand significantly influences pricing  
- Hardware specifications collectively affect the price  



##  Actual vs Predicted Analysis

- Scatter plot comparison:
  - X-axis → Actual Price  
  - Y-axis → Predicted Price  
- A diagonal line represents perfect predictions  
- Points closer to the line indicate better model performance  

##  Prediction Pipeline
1. Input laptop features  
2. Data preprocessing (encoding)  
3. Model prediction  
4. Output → Predicted price  

##  Key Insights
- RAM strongly affects laptop price  
- Brand influences pricing trends  
- Random Forest outperforms other models  
- Feature encoding is crucial for performance  

##  Limitations
- No hyperparameter tuning  
- No cross-validation  
- Single train-test split  
- High dimensionality due to one-hot encoding  

##  Future Improvements
- Hyperparameter tuning using GridSearchCV  
- Cross-validation for better generalization  
- Try advanced models (XGBoost, LightGBM)  
- Feature selection / dimensionality reduction  
- Deployment using Streamlit or Flask  


##  Tech Stack
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- Jupyter Notebook  


##  Conclusion
This project demonstrates a complete machine learning pipeline for laptop price prediction using multiple regression models.
It includes:
- Data preprocessing and visualization  
- Training and comparing multiple models  
- Evaluating performance using regression metrics  
Among all models tested, **Random Forest achieved the best performance (R² = 0.96)**, making it the most effective model for this task.