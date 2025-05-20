# Media Company Viewership Analysis 

## Overview

This documentation explains the Python script developed for analyzing a digital media company's show that experienced a decline in viewership after an initially positive response. The analysis aims to identify key factors contributing to the viewership decline and provide actionable recommendations.

## Table of Contents

1. [Data Loading and Initial Exploration](#1-data-loading-and-initial-exploration)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Exploratory Data Analysis](#3-exploratory-data-analysis)
4. [Statistical Modeling](#4-statistical-modeling)
5. [Final Model Evaluation](#5-final-model-evaluation)
6. [Key Findings](#6-key-findings)
7. [Recommendations](#7-recommendations)
8. [Technical Implementation Details](#8-technical-implementation-details)

## 1. Data Loading and Initial Exploration

The script begins by importing the necessary Python libraries for data analysis and visualization:

- **pandas**: For data manipulation and analysis
- **numpy**: For numerical operations
- **plotly**: For interactive visualizations
- **statsmodels**: For statistical modeling
- **sklearn**: For machine learning implementations

The dataset is loaded from a CSV file named 'mediacompany.csv', and basic information about the dataset is displayed:
- Dataset dimensions (rows and columns)
- First few rows of the dataset
- Data types of each column
- Summary statistics
- Check for missing values

## 2. Data Preprocessing

Several data transformations are applied to prepare the data for analysis:

1. **Date Conversion**: The 'Date' column is converted to datetime format
2. **Days Since Launch**: A 'day' variable is created to represent days since the show started (February 28, 2017)
3. **Weekend Indicator**: A binary variable (1 for weekend, 0 for weekday) is created
4. **Weekday Variable**: A numeric variable (1-7) representing the day of the week is created
5. **Scaled Ad Impressions**: Ad impressions are converted to millions for better scaling in models
6. **Lagged Views**: A one-day lag variable for views is created to capture time dependency

## 3. Exploratory Data Analysis

The script generates 11 interactive visualizations to explore different aspects of the data:

### Visualization 1: Viewership Trend Over Time
- Line chart showing the declining trend of viewership over time
- Includes a trendline to highlight the overall decline pattern

### Visualization 2: Average Viewership by Day of Week
- Bar chart displaying average viewership for each day of the week
- Helps identify which days perform better/worse

### Visualization 3: Weekday vs Weekend Viewership
- Comparative bar chart showing difference between weekday and weekend viewership
- Quantifies the weekend effect on viewership

### Visualization 4: Correlation Heatmap
- Visual representation of correlations between all variables
- Helps identify potentially significant relationships

### Visualization 5: Key Relationships Analysis
- Four-panel visualization examining relationships between:
  - Visitors vs Show Views
  - Ad Impressions vs Show Views
  - Character A's presence impact on Views
  - Cricket Match impact on Views

### Visualization 6: Views and Ad Impressions Over Time
- Dual-axis plot comparing show views and ad impressions trends
- Helps visualize the relationship between these metrics over time

### Visualization 7: Actual vs Predicted Views
- Line chart comparing actual viewership with model predictions
- Demonstrates model fit and prediction accuracy

### Visualization 8: Residuals Plot
- Time series plot of model residuals
- Helps assess model validity and identify patterns in errors

### Visualization 9: Distribution of Residuals
- Histogram showing the distribution of model residuals
- Helps assess normality assumptions

### Visualization 10: Model Comparison
- Bar chart comparing performance metrics of top models
- Assists in selecting the best model

### Visualization 11: Feature Importance
- Horizontal bar chart showing the importance of each feature in the final model
- Color-coded to indicate positive/negative impacts

## 4. Statistical Modeling

The analysis tests various linear regression models with different combinations of features:

1. **Model 1**: Visitors & Weekday
2. **Model 2**: Visitors & Weekend
3. **Model 3**: Visitors, Weekend & Character_A
4. **Model 4**: Visitors, Character_A, Lag_Views & Weekend
5. **Model 5**: Weekend, Character_A & Platform Views
6. **Model 6**: Weekend, Character_A & Visitors
7. **Model 7**: Weekend, Character_A, Visitors & Ad Impressions
8. **Model 8**: Weekend, Character_A & Ad Impressions
9. **Model 9**: Weekend, Character_A, Ad Impressions (millions) & Cricket Match
10. **Model 10**: Weekend, Character_A & Ad Impressions (millions)

Each model is evaluated using:
- Adjusted R-squared
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)

## 5. Final Model Evaluation

After comparing all models, **Model 10** is selected as the best model based on its balanced performance:

- **Features**: Weekend, Character_A, and Ad Impressions (millions)
- **Interpretation**: These three factors have the most significant impact on viewership

The model evaluation includes:
- Comparison of actual vs. predicted values
- Analysis of residuals
- Feature importance assessment

## 6. Key Findings

The analysis reveals several important insights:

1. **Declining Trend**: Clear evidence of viewership decline over time
2. **Weekend Effect**: Significantly higher viewership on weekends compared to weekdays
3. **Character_A Impact**: Strong positive correlation between Character_A's presence and viewership
4. **Ad Impression Impact**: Significant negative correlation between ad impressions and viewership
5. **Cricket Match Effect**: Negative impact of cricket matches on viewership

## 7. Recommendations

Based on the analysis, the following recommendations are provided:

1. **Optimize Ad Strategy**: Reduce ad frequency or improve ad quality to minimize viewer fatigue
2. **Leverage Character_A**: Increase screen time or importance of Character_A, as this character drives viewership
3. **Weekend-Weekday Balance**: Consider special content or promotions for weekdays to balance viewership
4. **Avoid Scheduling Conflicts**: Avoid scheduling important episodes on days with cricket matches
5. **Further Investigation**: Continue monitoring the relationship between platform visitors and show viewership

## 8. Technical Implementation Details

### Libraries and Dependencies

```python
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date
```

### Custom Functions

#### Model Building Function
```python
def build_model(features, target, model_name="Model"):
    """Build and evaluate a linear regression model"""
    X = media[features].copy()
    y = media[target].copy()
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X_with_const).fit()
    
    # Model evaluation metrics
    predictions = model.predict(X_with_const)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    
    return model, predictions
```

### Visualization Export Format
All visualizations are exported as HTML files for interactive viewing:
- `visualization1_viewership_trend.html`
- `visualization2_weekday_viewership.html`
- `visualization3_weekend_vs_weekday.html`
- `visualization4_correlation_heatmap.html`
- `visualization5_key_relationships.html`
- `visualization6_views_ad_impressions.html`
- `visualization7_actual_vs_predicted.html`
- `visualization8_residuals.html`
- `visualization9_residual_histogram.html`
- `visualization10_model_comparison.html`
- `visualization11_feature_importance.html`

## Conclusion

The comprehensive analysis provides clear evidence that the declining viewership is primarily influenced by:

1. **Ad Impressions** (negative impact): Excessive advertisements appear to be driving viewers away
2. **Character_A's Presence** (positive impact): This character significantly drives viewership
3. **Day of Week** (timing impact): Weekend scheduling yields higher viewership

By addressing these key factors through targeted strategies, the media company can work to reverse the viewership decline trend and improve audience engagement.
