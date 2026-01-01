import pandas as pd

# Data from the table
data = {
    'Dataset': ['Air Pollution', 'Air Pollution', 'Air Pollution', 'Air Pollution',
                'Microsoft Stock', 'Microsoft Stock', 'Microsoft Stock', 'Microsoft Stock',
                'Wind Power Generation', 'Wind Power Generation', 'Wind Power Generation', 'Wind Power Generation',
                'HPC', 'HPC', 'HPC', 'HPC',
                'QPS', 'QPS', 'QPS', 'QPS',
                'Sales', 'Sales', 'Sales', 'Sales'],
    'Horizon': [96, 192, 336, 720] * 6,
    'DLinear': [0.122, 0.149, 0.175, 0.229, 2.468, 3.470, 7.631, 21.068, 
                0.702, 0.736, 0.754, 0.775, 0.608, 0.622, 0.627, 0.626,
                0.090, 0.230, 0.470, 0.759, 1.050, 1.052, 1.054, 1.063],
    'CATS': [0.122, 0.152, 0.177, 0.247, 0.797, 3.270, 11.031, 22.730,
             0.696, 0.736, 0.754, 0.777, 0.606, 0.631, 0.642, 0.643,
             0.041, 0.085, 0.184, 0.352, 1.041, 1.043, 1.045, 1.053],
    'PatchTST': [0.131, 0.163, 0.194, 0.267, 1.248, 3.088, 5.894, 18.673,
                 0.760, 0.810, 0.851, 0.878, 0.638, 0.665, 0.681, 0.679,
                 0.063, 0.207, 0.536, 1.116, 1.068, 1.068, 1.070, 1.080],
    'NHITS': [0.158, 0.214, 0.232, 0.281, 0.926, 1.246, 1.343, 1.410,
              0.831, 0.898, 0.917, 0.867, 0.658, 0.689, 0.695, 0.710,
              0.043, 0.084, 0.083, 0.088, 1.174, 1.178, 1.190, 1.205],
    'FEDformer': [0.151, 0.182, 0.211, 0.236, 2.361, 4.311, 8.188, 19.593,
                  0.762, 0.819, 0.865, 0.875, 0.633, 0.656, 0.664, 0.667,
                  0.086, 0.185, 0.263, 0.250, 1.050, 1.051, 1.052, 1.058],
    'Autoformer': [0.188, 0.194, 0.268, 0.281, 2.440, 4.677, 8.249, 20.534,
                   0.783, 0.848, 0.850, 0.867, 0.672, 0.702, 0.699, 0.687,
                   0.363, 0.308, 0.533, 0.655, 1.071, 1.065, 1.059, 1.066],
    'Naive': [0.448, 0.482, 0.516, 0.579, 1.218, 3.109, 6.606, 19.435,
              1.325, 1.425, 1.496, 1.528, 1.340, 1.363, 1.376, 1.359,
              0.153, 0.426, 0.826, 1.555, 2.279, 2.280, 2.282, 2.230]
}

df = pd.DataFrame(data)

# Calculate ranks for each dataset-horizon combination (lower MSE = better rank)
models = ['DLinear', 'CATS', 'PatchTST', 'NHITS', 'FEDformer', 'Autoformer', 'Naive']

# Add rank columns for each model
# For each row (dataset-horizon combo), rank all models
rank_cols = []
for idx, row in df.iterrows():
    model_values = [(model, row[model]) for model in models]
    model_values_sorted = sorted(model_values, key=lambda x: x[1])
    ranks = {}
    for rank, (model, value) in enumerate(model_values_sorted, 1):
        ranks[model] = rank
    rank_cols.append(ranks)

for model in models:
    df[f'{model}_rank'] = [row[model] for row in rank_cols]

# Calculate average rank for each horizon
avg_ranks = {}
for horizon in [96, 192, 336, 720]:
    horizon_data = df[df['Horizon'] == horizon]
    avg_ranks[horizon] = {}
    for model in models:
        avg_ranks[horizon][model] = horizon_data[f'{model}_rank'].mean()

# Create markdown table format
print("| Horizon | DLinear | CATS | PatchTST | NHITS | FEDformer | Autoformer | Naive |")
print("|---------|---------|------|----------|-------|-----------|------------|-------|")
for horizon in [96, 192, 336, 720]:
    row = f"| {horizon} "
    for model in models:
        row += f"| {avg_ranks[horizon][model]:.2f} "
    row += "|"
    print(row)

# Calculate overall average rank (across all horizons and datasets)
print("\n\nOverall Average Rank (across all datasets and horizons):")
for model in models:
    overall_avg = df[f'{model}_rank'].mean()
    print(f"{model}: {overall_avg:.2f}")
