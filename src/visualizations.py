"""
Visualization utilities for EDA
"""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def plot_success_rate_by_category(df, category_col, title=None):
    """Plot success rate by category"""
    success_rate = df.groupby(category_col)['Mission_Success'].agg(['sum', 'count', 'mean'])
    success_rate.columns = ['Successes', 'Total', 'Success_Rate']
    success_rate = success_rate.sort_values('Success_Rate', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(success_rate))
    
    ax.bar(x, success_rate['Success_Rate'], color='skyblue', edgecolor='navy', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(success_rate.index, rotation=45, ha='right')
    ax.set_ylabel('Success Rate')
    ax.set_ylim(0, 1.1)
    ax.set_title(title or f'Success Rate by {category_col}')
    
    # Add count labels
    for i, (idx, row) in enumerate(success_rate.iterrows()):
        ax.text(i, row['Success_Rate'] + 0.02, 
                f"{row['Successes']}/{row['Total']}\n({row['Success_Rate']:.1%})",
                ha='center', va='bottom', fontsize=9)
    
    ax.axhline(y=df['Mission_Success'].mean(), color='red', linestyle='--', 
               label=f"Overall: {df['Mission_Success'].mean():.1%}")
    ax.legend()
    plt.tight_layout()
    return fig

def plot_payload_distribution(df):
    """Plot payload mass distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['Payload Mass (kg)'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Payload Mass (kg)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Payload Mass')
    axes[0].axvline(df['Payload Mass (kg)'].median(), color='red', linestyle='--', 
                    label=f"Median: {df['Payload Mass (kg)'].median():.0f} kg")
    axes[0].legend()
    
    # Box plot by success
    axes[1].boxplot([df[df['Mission_Success']==0]['Payload Mass (kg)'].dropna(),
                     df[df['Mission_Success']==1]['Payload Mass (kg)'].dropna()],
                    labels=['Failure', 'Success'])
    axes[1].set_ylabel('Payload Mass (kg)')
    axes[1].set_title('Payload Mass by Mission Outcome')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_temporal_trends(df):
    """Plot trends over time"""
    yearly_stats = df.groupby('Year').agg({
        'Mission_Success': ['sum', 'count', 'mean'],
        'Payload Mass (kg)': 'mean'
    }).reset_index()
    
    yearly_stats.columns = ['Year', 'Successes', 'Total', 'Success_Rate', 'Avg_Payload']
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Success rate over time
    axes[0].plot(yearly_stats['Year'], yearly_stats['Success_Rate'], 
                 marker='o', linewidth=2, markersize=8, color='green')
    axes[0].fill_between(yearly_stats['Year'], yearly_stats['Success_Rate'], 
                         alpha=0.3, color='green')
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Mission Success Rate Over Time')
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(alpha=0.3)
    
    # Add count labels
    for _, row in yearly_stats.iterrows():
        axes[0].text(row['Year'], row['Success_Rate'] + 0.05, 
                    f"{row['Successes']}/{row['Total']}", 
                    ha='center', fontsize=9)
    
    # Launch frequency
    axes[1].bar(yearly_stats['Year'], yearly_stats['Total'], 
                color='steelblue', edgecolor='navy', alpha=0.7)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Number of Launches')
    axes[1].set_title('Launch Frequency Over Time')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(df, numeric_cols):
    """Plot correlation heatmap"""
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    return fig

def plot_orbit_launch_site_heatmap(df):
    """Plot heatmap of launches by orbit and launch site"""
    pivot = pd.crosstab(df['Orbit_Simplified'], df['Launch_Site_Simplified'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
    ax.set_title('Number of Launches by Orbit Type and Launch Site')
    ax.set_xlabel('Launch Site')
    ax.set_ylabel('Orbit Type')
    plt.tight_layout()
    return fig

def plot_booster_reuse_analysis(df):
    """Analyze booster reuse patterns"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Success rate by reuse
    reuse_stats = df.groupby('Booster_Reused')['Mission_Success'].agg(['sum', 'count', 'mean'])
    
    axes[0].bar(['New Booster', 'Reused Booster'], reuse_stats['mean'], 
                color=['lightcoral', 'lightgreen'], edgecolor='black', alpha=0.7)
    axes[0].set_ylabel('Success Rate')
    axes[0].set_title('Success Rate: New vs Reused Boosters')
    axes[0].set_ylim(0, 1.1)
    
    for i, rate in enumerate(reuse_stats['mean']):
        axes[0].text(i, rate + 0.02, f"{rate:.1%}", ha='center', fontsize=12)
    
    # Reuse trend over time
    reuse_trend = df.groupby('Year')['Booster_Reused'].mean()
    axes[1].plot(reuse_trend.index, reuse_trend.values, marker='o', 
                 linewidth=2, markersize=8, color='purple')
    axes[1].fill_between(reuse_trend.index, reuse_trend.values, alpha=0.3, color='purple')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Proportion of Reused Boosters')
    axes[1].set_title('Booster Reuse Trend Over Time')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_interactive_timeline(df):
    """Create interactive timeline with Plotly"""
    fig = px.scatter(df, x='DateTime', y='Payload Mass (kg)',
                     color='Mission_Success',
                     size='Payload Mass (kg)',
                     hover_data=['Payload', 'Orbit', 'Launch Site', 'Booster Version'],
                     title='SpaceX Launches Timeline',
                     labels={'Mission_Success': 'Success'},
                     color_discrete_map={0: 'red', 1: 'green'})
    
    fig.update_layout(height=600, showlegend=True)
    return fig

def create_3d_scatter(df):
    """Create 3D scatter plot"""
    fig = px.scatter_3d(df, x='Year', y='Payload Mass (kg)', z='Orbit_Difficulty',
                        color='Mission_Success',
                        size='Payload Mass (kg)',
                        hover_data=['Payload', 'Booster_Type'],
                        title='3D Visualization: Year, Payload, Orbit Difficulty',
                        color_discrete_map={0: 'red', 1: 'green'},
                        labels={'Mission_Success': 'Success'})
    
    fig.update_layout(height=700)
    return fig

def plot_cumulative_success_rate(df):
    """Plot cumulative success rate over time"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df['DateTime'], df['Cumulative_Success_Rate'], 
            linewidth=2, color='blue', label='Cumulative Success Rate')
    ax.fill_between(df['DateTime'], df['Cumulative_Success_Rate'], 
                     alpha=0.3, color='blue')
    ax.axhline(y=df['Mission_Success'].mean(), color='red', linestyle='--', 
               label=f"Overall Average: {df['Mission_Success'].mean():.1%}")
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Success Rate')
    ax.set_title('SpaceX Mission Success Rate Evolution')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def create_summary_stats(df):
    """Create summary statistics"""
    stats = {
        'Total Launches': len(df),
        'Successful Launches': df['Mission_Success'].sum(),
        'Failed Launches': len(df) - df['Mission_Success'].sum(),
        'Overall Success Rate': f"{df['Mission_Success'].mean():.1%}",
        'Avg Payload Mass (kg)': f"{df['Payload Mass (kg)'].mean():.1f}",
        'Total Years Active': df['Year'].max() - df['Year'].min() + 1,
        'Launches with Reused Boosters': df['Booster_Reused'].sum(),
        'Unique Launch Sites': df['Launch Site'].nunique(),
        'Unique Orbit Types': df['Orbit'].nunique()
    }
    return pd.Series(stats)
