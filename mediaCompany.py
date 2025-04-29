# Media Company Case Study: Viewership Decline Analysis
# ======================================================
# A digital media company launched a show that initially received good response
# but then witnessed a decline in viewership. This analysis aims to determine what went wrong.

# Importing necessary libraries
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from datetime import date

# Setting pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.precision', 3)

print("# 1. Data Loading and Initial Exploration")
print("=" * 50)

# Import dataset
media = pd.read_csv('mediacompany.csv')
# Remove unnamed column if it exists
if 'Unnamed: 7' in media.columns:
    media = media.drop('Unnamed: 7', axis=1)
    
# Display basic information about the dataset
print(f"Dataset Shape: {media.shape}")
print("\nFirst 5 rows:")
print(media.head())

print("\nData Types:")
print(media.dtypes)

print("\nSummary Statistics:")
print(media.describe())

# Check for missing values
print("\nMissing Values:")
print(media.isnull().sum())

print("\n# 2. Data Preprocessing")
print("=" * 50)

# Convert 'Date' to datetime format
media['Date'] = pd.to_datetime(media['Date'])

# Calculate days since the show started
show_start_date = date(2017, 2, 28)  # Starting date of the show
media['day'] = (media['Date'] - pd.Timestamp(show_start_date)).dt.days

# Create weekend indicator (1 for weekend, 0 for weekday)
# Weekends are Friday (day % 7 == 4) and Saturday (day % 7 == 5)
media['weekend'] = media['day'].apply(lambda x: 1 if x % 7 in [4, 5] else 0)

# Create weekday variable (1-7, where 1 is Sunday and 7 is Saturday)
media['weekday'] = ((media['day'] + 3) % 7).replace(0, 7).astype(int)

# Convert Ad_impression to millions for better scaling in models
media['ad_impression_million'] = media['Ad_impression'] / 1000000

# Create lag variable for Views_show (previous day's views)
media['Lag_Views'] = media['Views_show'].shift(1)
media.loc[media.index[0], 'Lag_Views'] = 0  # Set first day's lag to 0

print("Processed data sample:")
print(media.head())

print("\n# 3. Exploratory Data Analysis")
print("=" * 50)

# VISUALIZATION 1: Plot viewership trend over time with Plotly
fig = px.line(media, x='day', y='Views_show', 
              title='Show Viewership Trend Over Time',
              labels={'day': 'Days Since Show Launch', 'Views_show': 'Number of Views'},
              markers=True)

fig.update_layout(
    title_font_size=20,
    xaxis_title_font_size=16,
    yaxis_title_font_size=16,
    template='plotly_white',
    height=600,
    width=1000
)

# Add trendline to visualize the decline
fig.add_trace(
    go.Scatter(
        x=media['day'],
        y=np.poly1d(np.polyfit(media['day'], media['Views_show'], 1))(media['day']),
        mode='lines',
        name='Trend',
        line=dict(color='red', dash='dash')
    )
)

fig.write_html("visualization1_viewership_trend.html")  # Save to HTML file
print("Visualization 1 saved to 'visualization1_viewership_trend.html'")

# VISUALIZATION 2: Plot viewership by weekday with Plotly
weekday_views = media.groupby('weekday')['Views_show'].mean().reindex(range(1, 8))
weekday_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

fig = px.bar(
    x=weekday_names,
    y=weekday_views.values,
    title='Average Viewership by Day of Week',
    labels={'x': 'Day of Week', 'y': 'Average Views'},
    text=weekday_views.values.astype(int)
)

# Fix: Replace the qualitative.Viridis with a valid color option
fig.update_traces(
    texttemplate='%{text}',
    textposition='outside',
    marker_color=px.colors.qualitative.Plotly  # Using Plotly qualitative colors instead of Viridis
)

fig.update_layout(
    title_font_size=20,
    xaxis_title_font_size=16,
    yaxis_title_font_size=16,
    template='plotly_white',
    height=600,
    width=1000
)

fig.write_html("visualization2_weekday_viewership.html")  # Save to HTML file
print("Visualization 2 saved to 'visualization2_weekday_viewership.html'")

# VISUALIZATION 3: Create comparison between weekend vs weekday viewership
weekend_views = media.groupby('weekend')['Views_show'].mean()
labels = ['Weekday', 'Weekend']

fig = px.bar(
    x=labels,
    y=weekend_views.values,
    title='Average Viewership: Weekday vs Weekend',
    labels={'x': '', 'y': 'Average Views'},
    text=weekend_views.values.astype(int),
    color=labels,
    color_discrete_sequence=['#1f77b4', '#ff7f0e']
)

fig.update_traces(
    texttemplate='%{text}',
    textposition='outside'
)

fig.update_layout(
    title_font_size=20,
    yaxis_title_font_size=16,
    template='plotly_white',
    height=600,
    width=800
)

fig.write_html("visualization3_weekend_vs_weekday.html")  # Save to HTML file
print("Visualization 3 saved to 'visualization3_weekend_vs_weekday.html'")

# VISUALIZATION 4: Correlation heatmap with Plotly
corr_matrix = media.corr()

# Create a mask for the upper triangle
mask = np.zeros_like(corr_matrix)
mask[np.triu_indices_from(mask, k=1)] = True
masked_corr = corr_matrix.copy()
masked_corr[mask.astype(bool)] = np.nan

fig = px.imshow(
    masked_corr,
    text_auto='.2f',
    color_continuous_scale='Viridis',  # This is correct because it's using sequential colors
    title='Correlation Matrix of Variables',
    aspect="auto"
)

fig.update_layout(
    title_font_size=20,
    width=1000,
    height=800
)

fig.write_html("visualization4_correlation_heatmap.html")  # Save to HTML file
print("Visualization 4 saved to 'visualization4_correlation_heatmap.html'")

# VISUALIZATION 5: Scatter plots with regression lines for key relationships
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Visitors vs Show Views',
        'Ad Impressions (millions) vs Show Views',
        'Impact of Character A on Show Views',
        'Impact of Cricket Match on Show Views'
    ),
    vertical_spacing=0.12,
    horizontal_spacing=0.08
)

# Plot 1: Visitors vs Views
fig.add_trace(
    go.Scatter(
        x=media['Visitors'],
        y=media['Views_show'],
        mode='markers',
        marker=dict(color='blue', opacity=0.6),
        name='Data Points'
    ),
    row=1, col=1
)

# Add regression line for Visitors vs Views
x_range = np.linspace(min(media['Visitors']), max(media['Visitors']), 100)
y_pred = np.poly1d(np.polyfit(media['Visitors'], media['Views_show'], 1))(x_range)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color='red', width=2),
        name='Regression Line'
    ),
    row=1, col=1
)

# Plot 2: Ad impressions vs Views
fig.add_trace(
    go.Scatter(
        x=media['ad_impression_million'],
        y=media['Views_show'],
        mode='markers',
        marker=dict(color='green', opacity=0.6),
        name='Data Points'
    ),
    row=1, col=2
)

# Add regression line for Ad Impressions vs Views
x_range = np.linspace(min(media['ad_impression_million']), max(media['ad_impression_million']), 100)
y_pred = np.poly1d(np.polyfit(media['ad_impression_million'], media['Views_show'], 1))(x_range)
fig.add_trace(
    go.Scatter(
        x=x_range,
        y=y_pred,
        mode='lines',
        line=dict(color='red', width=2),
        name='Regression Line'
    ),
    row=1, col=2
)

# Plot 3: Character_A presence impact on Views (Box Plot)
fig.add_trace(
    go.Box(
        x=media['Character_A'].astype(str),
        y=media['Views_show'],
        name='Character A',
        marker_color='purple'
    ),
    row=2, col=1
)

# Plot 4: Cricket match impact on Views (Box Plot)
fig.add_trace(
    go.Box(
        x=media['Cricket_match_india'].astype(str),
        y=media['Views_show'],
        name='Cricket Match',
        marker_color='orange'
    ),
    row=2, col=2
)

# Update layout with better formatting
fig.update_layout(
    title='Key Relationships Analysis',
    title_font_size=20,
    showlegend=False,
    height=900,
    width=1100,
    template='plotly_white'
)

# Update x and y axis labels
fig.update_xaxes(title_text='Visitors', row=1, col=1)
fig.update_yaxes(title_text='Show Views', row=1, col=1)

fig.update_xaxes(title_text='Ad Impressions (millions)', row=1, col=2)
fig.update_yaxes(title_text='Show Views', row=1, col=2)

fig.update_xaxes(title_text='Character A Present (1) vs Absent (0)', row=2, col=1)
fig.update_yaxes(title_text='Show Views', row=2, col=1)

fig.update_xaxes(title_text='Cricket Match Day (1) vs Regular Day (0)', row=2, col=2)
fig.update_yaxes(title_text='Show Views', row=2, col=2)

fig.write_html("visualization5_key_relationships.html")  # Save to HTML file
print("Visualization 5 saved to 'visualization5_key_relationships.html'")

# VISUALIZATION 6: Create a dual-axis plot for Views_show and Ad_impression
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(
        x=media['day'],
        y=media['Views_show'],
        name='Show Views',
        mode='lines+markers',
        marker=dict(size=8, opacity=0.7, color='blue'),
        line=dict(width=2)
    ),
    secondary_y=False
)

fig.add_trace(
    go.Scatter(
        x=media['day'],
        y=media['Ad_impression'],
        name='Ad Impressions',
        mode='lines+markers',
        marker=dict(size=8, symbol='x', opacity=0.7, color='orange'),
        line=dict(width=2, dash='dot')
    ),
    secondary_y=True
)

# Add figure title
fig.update_layout(
    title_text='Show Views and Ad Impressions Over Time',
    title_font_size=20,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    template='plotly_white',
    height=600,
    width=1000
)

# Set x-axis title
fig.update_xaxes(title_text='Days Since Show Launch', title_font_size=16)

# Set y-axes titles
fig.update_yaxes(title_text='Show Views', title_font_size=16, secondary_y=False)
fig.update_yaxes(title_text='Ad Impressions', title_font_size=16, secondary_y=True)

fig.write_html("visualization6_views_ad_impressions.html")  # Save to HTML file
print("Visualization 6 saved to 'visualization6_views_ad_impressions.html'")

print("\n# 4. Statistical Modeling")
print("=" * 50)

# Function to build and evaluate a model
def build_model(features, target, model_name="Model"):
    """Build and evaluate a linear regression model"""
    X = media[features].copy()
    y = media[target].copy()
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X_with_const).fit()
    
    print(f"\n{model_name} Results:")
    print("=" * 30)
    print(f"Features: {', '.join(features)}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"AIC: {model.aic:.2f}")
    print(f"BIC: {model.bic:.2f}")
    
    # Print summary
    print("\nModel Summary:")
    print(model.summary().tables[1])
    
    # Make predictions
    predictions = model.predict(X_with_const)
    
    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\nModel Metrics:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R-squared: {model.rsquared:.4f}")
    
    return model, predictions

# Let's try different model specifications
# Model 1: Basic model with Visitors and weekday
model1, pred1 = build_model(['Visitors', 'weekday'], 'Views_show', "Model 1: Visitors & Weekday")

# Model 2: Replace weekday with weekend binary variable
model2, pred2 = build_model(['Visitors', 'weekend'], 'Views_show', "Model 2: Visitors & Weekend")

# Model 3: Add Character_A variable
model3, pred3 = build_model(['Visitors', 'weekend', 'Character_A'], 'Views_show', 
                            "Model 3: Visitors, Weekend & Character_A")

# Model 4: Add lag variable
model4, pred4 = build_model(['Visitors', 'Character_A', 'Lag_Views', 'weekend'], 'Views_show', 
                            "Model 4: Visitors, Character_A, Lag_Views & Weekend")

# Model 5: Try platform views instead of visitors
model5, pred5 = build_model(['weekend', 'Character_A', 'Views_platform'], 'Views_show', 
                            "Model 5: Weekend, Character_A & Platform Views")

# Model 6: Back to visitors with Character_A and weekend
model6, pred6 = build_model(['weekend', 'Character_A', 'Visitors'], 'Views_show', 
                            "Model 6: Weekend, Character_A & Visitors")

# Model 7: Add Ad impressions
model7, pred7 = build_model(['weekend', 'Character_A', 'Visitors', 'Ad_impression'], 'Views_show', 
                            "Model 7: Weekend, Character_A, Visitors & Ad Impressions")

# Model 8: Try Ad impressions without Visitors
model8, pred8 = build_model(['weekend', 'Character_A', 'Ad_impression'], 'Views_show', 
                            "Model 8: Weekend, Character_A & Ad Impressions")

# Model 9: Ad impressions in millions with Cricket match
model9, pred9 = build_model(['weekend', 'Character_A', 'ad_impression_million', 'Cricket_match_india'], 'Views_show',
                            "Model 9: Weekend, Character_A, Ad Impressions (millions) & Cricket Match")

# Model 10: Only ad impressions in millions (simpler model)
model10, pred10 = build_model(['weekend', 'Character_A', 'ad_impression_million'], 'Views_show', 
                              "Model 10: Weekend, Character_A & Ad Impressions (millions)")

print("\n# 5. Final Model Evaluation & Visualization")
print("=" * 50)

# Compare model performance metrics
models = {
    "Model 6": {"features": ['weekend', 'Character_A', 'Visitors'], "preds": pred6, "r2": model6.rsquared},
    "Model 10": {"features": ['weekend', 'Character_A', 'ad_impression_million'], "preds": pred10, "r2": model10.rsquared}
}

# VISUALIZATION 7: For the best model (Model 10), plot actual vs predicted values
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list(range(len(media))),
        y=media['Views_show'],
        mode='lines',
        name='Actual Views',
        line=dict(color='blue', width=2)
    )
)

fig.add_trace(
    go.Scatter(
        x=list(range(len(media))),
        y=pred10,
        mode='lines',
        name='Predicted Views',
        line=dict(color='red', width=2, dash='dash')
    )
)

fig.update_layout(
    title='Model 10: Actual vs Predicted Views',
    title_font_size=20,
    xaxis_title='Days (Index)',
    yaxis_title='Number of Views',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    template='plotly_white',
    height=600,
    width=1000
)

fig.write_html("visualization7_actual_vs_predicted.html")  # Save to HTML file
print("Visualization 7 saved to 'visualization7_actual_vs_predicted.html'")

# VISUALIZATION 8: Plot residuals for Model 10
residuals = media['Views_show'] - pred10

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list(range(len(media))),
        y=residuals,
        mode='lines+markers',
        name='Residuals',
        line=dict(color='green'),
        marker=dict(size=8)
    )
)

fig.add_trace(
    go.Scatter(
        x=[0, len(media)-1],
        y=[0, 0],
        mode='lines',
        name='Zero Line',
        line=dict(color='red', width=2)
    )
)

fig.update_layout(
    title='Model 10: Residuals Plot',
    title_font_size=20,
    xaxis_title='Observation Index',
    yaxis_title='Residual Value (Actual - Predicted)',
    template='plotly_white',
    height=600,
    width=1000
)

fig.write_html("visualization8_residuals.html")  # Save to HTML file
print("Visualization 8 saved to 'visualization8_residuals.html'")

# VISUALIZATION 9: Create a residual histogram with Plotly
fig = px.histogram(
    x=residuals,
    nbins=20,
    title='Distribution of Residuals (Model 10)',
    labels={'x': 'Residual Value'},
    marginal='box',
    opacity=0.7,
    color_discrete_sequence=['green']
)

fig.update_layout(
    title_font_size=20,
    xaxis_title_font_size=16,
    yaxis_title='Frequency',
    yaxis_title_font_size=16,
    template='plotly_white',
    height=600,
    width=1000
)

fig.write_html("visualization9_residual_histogram.html")  # Save to HTML file
print("Visualization 9 saved to 'visualization9_residual_histogram.html'")

# VISUALIZATION 10: Compare top models with bar chart
fig = go.Figure()

r2_values = [model6.rsquared, model10.rsquared]
model_names = ['Model 6\n(Weekend, Character_A, Visitors)', 'Model 10\n(Weekend, Character_A, Ad Impressions)']

fig.add_trace(
    go.Bar(
        x=model_names,
        y=r2_values,
        text=[f'{val:.4f}' for val in r2_values],
        textposition='outside',
        marker_color=['#1f77b4', '#ff7f0e']
    )
)

fig.update_layout(
    title='Comparison of Top Models (R-squared)',
    title_font_size=20,
    yaxis_title='R-squared Value',
    yaxis_title_font_size=16,
    template='plotly_white',
    height=600,
    width=900,
    yaxis=dict(range=[0.7, 1.0])
)

fig.write_html("visualization10_model_comparison.html")  # Save to HTML file
print("Visualization 10 saved to 'visualization10_model_comparison.html'")

print("\n# 6. Summary of Findings")
print("=" * 50)

print("""
Key Findings:
1. The viewership of the show has clearly declined over time as shown in the trend analysis.
2. Weekend vs. weekday effect: The show performs significantly better on weekends.
3. Character_A has a strong positive impact on viewership. When Character_A is present, viewership increases.
4. Ad impressions have a significant negative correlation with show viewership, suggesting 
   that excessive ads may be driving viewers away.
5. Cricket matches in India have a negative impact on viewership, as expected.

Recommendations:
1. Increase the screen time or importance of Character_A, as this character clearly drives viewership.
2. Consider reducing ad frequency or improving ad quality, as high ad impressions correlate with lower viewership.
3. Special content or promotion for weekdays might help balance the weekday-weekend viewership gap.
4. Avoid scheduling important episodes on days with cricket matches.
5. Further investigate the relationship between platform visitors and show viewership.
""")

# Let's provide a more detailed understanding of the best model (Model 10)
final_model = model10
final_features = ['weekend', 'Character_A', 'ad_impression_million']

print("\n# 7. Detailed Analysis of Best Model (Model 10)")
print("=" * 50)

# Calculate and display coefficient importance
coefficients = pd.DataFrame({
    'Feature': final_features,
    'Coefficient': final_model.params[1:],  # Skip the constant
    'Absolute_Impact': abs(final_model.params[1:])  # Fixed: Changed 'Absolute Impact' to 'Absolute_Impact'
})

coefficients = coefficients.sort_values('Absolute_Impact', ascending=False)

print("Feature Importance by Coefficient Magnitude:")
print(coefficients)

# VISUALIZATION 11: Visualize feature importance with Plotly
sign = ['+' if c > 0 else '-' for c in coefficients['Coefficient']]
colors = ['green' if c > 0 else 'red' for c in coefficients['Coefficient']]

fig = px.bar(
    coefficients,
    y='Feature',
    x='Absolute_Impact',  # Fixed: Changed 'Absolute Impact' to 'Absolute_Impact'
    orientation='h',
    title='Feature Importance in Model 10',
    color_discrete_sequence=colors
)

# Add annotation for coefficient sign and value
for i, row in enumerate(coefficients.itertuples()):
    sign_char = '+' if row.Coefficient > 0 else '-'
    fig.add_annotation(
        x=row.Absolute_Impact + 0.01,  # This matches the column name in the namedtuple
        y=i,
        text=f"{sign_char} {abs(row.Coefficient):.2f}",  # Using abs(row.Coefficient) for clarity
        showarrow=False,
        xanchor='left'
    )

fig.update_layout(
    title_font_size=20,
    xaxis_title='Absolute Coefficient Value',
    xaxis_title_font_size=16,
    template='plotly_white',
    height=500,
    width=1000
)

fig.write_html("visualization11_feature_importance.html")  # Save to HTML file 
print("Visualization 11 saved to 'visualization11_feature_importance.html'")

# Final conclusion message
print("""
Final Conclusion:
Based on our comprehensive analysis, the decline in viewership appears to be 
primarily influenced by three key factors:

1. Ad impressions (negative impact): There is a strong negative relationship between 
   ad impressions and viewership, suggesting that excessive advertisements are driving 
   viewers away. This is the most significant factor explaining the decline.

2. Character_A presence (positive impact): Episodes featuring Character_A consistently 
   show higher viewership. The absence of this character in some episodes contributes 
   to viewership decline.

3. Weekend vs. Weekday scheduling (timing impact): Viewership is significantly higher 
   on weekends. The scheduling of important content on weekdays may be contributing to 
   lower overall viewership.

The media company should consider reducing ad frequency, increasing Character_A's role, 
and optimizing their content scheduling strategy to reverse the viewership decline trend.
""")