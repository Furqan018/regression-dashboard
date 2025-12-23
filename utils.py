import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.graph_objects as go

def calculate_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    metrics = {
        'R2': r2_score(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }
    return metrics

def create_residual_plot(y_true, y_pred, title="Residual Plot"):
    """Create residual plot"""
    residuals = y_true - y_pred
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        marker=dict(color='blue', size=8),
        name='Residuals'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(
        title=title,
        xaxis_title='Predicted Values',
        yaxis_title='Residuals',
        showlegend=True
    )
    return fig

def create_actual_vs_predicted_plot(y_true, y_pred, title="Actual vs Predicted"):
    """Create actual vs predicted plot"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode='markers',
        marker=dict(color='blue', size=8),
        name='Predictions'
    ))
    fig.add_trace(go.Scatter(
        x=[y_true.min(), y_true.max()], 
        y=[y_true.min(), y_true.max()],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Actual Values',
        yaxis_title='Predicted Values',
        showlegend=True
    )
    return fig