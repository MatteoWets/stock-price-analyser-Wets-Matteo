<<<<<<< HEAD
# This script creates the final visualization for the results
=======
# This script creates the final visualizations with all the results

>>>>>>> 8197f3c8b1bc4b6c148354d233c9296efa9890e9

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the test results
df = pd.read_csv('test_results.csv')

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Define colors for each model
colors = {
    'catboost': '#4878A8',
    'lightgbm': '#E07B39', 
    'ensemble': '#5A9F5F',
    'xgboost': '#C94D4D'
}

# Metrics to plot
metrics = ['accuracy', 'auc', 'f1']
titles = ['Accuracy by Target', 'Auc by Target', 'F1 by Target']

# Get unique targets and sort them
targets = sorted(df['target'].unique())

# Set width of bars
bar_width = 0.2
x = np.arange(len(targets))

# Plot each metric
for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes[idx]
    
    # Get unique models
    models = ['catboost', 'lightgbm', 'ensemble', 'xgboost']
    
    # Plot bars for each model
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        values = []
        
        for target in targets:
            target_data = model_data[model_data['target'] == target]
            if not target_data.empty:
                values.append(target_data[metric].values[0])
            else:
                values.append(0)
        
        ax.bar(x + i * bar_width, values, bar_width, 
               label=model, color=colors[model], alpha=0.9)
    
    # Customize the plot
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(targets, rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Adjust y-axis limits for better visibility of differences
    all_values = []
    for model in models:
        model_data = df[df['model'] == model]
        for target in targets:
            target_data = model_data[model_data['target'] == target]
            if not target_data.empty:
                all_values.append(target_data[metric].values[0])
    
    if all_values:
        min_val = min(all_values)
        max_val = max(all_values)
        range_val = max_val - min_val
        ax.set_ylim(min_val - range_val * 0.1, max_val + range_val * 0.1)
    
    # Add legend only to the first plot
    if idx == 0:
        ax.legend(title='Model', loc='upper left', framealpha=0.9)

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('test_results_visualization.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'test_results_visualization.png'")

# Show the plot
plt.show()