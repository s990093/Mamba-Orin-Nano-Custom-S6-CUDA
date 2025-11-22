import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Prepare the data
data = {
    'Seq_Len': [128, 512, 1024, 2048, 4096],
    'Transformer_Time_ms': [3.32, 4.88, 9.77, 205.62, 189.41],
    'Mamba_Time_ms': [2.89, 3.35, 3.58, 5.08, 23.25]
}
df = pd.DataFrame(data)

# 2. Set up the figure
plt.figure(figsize=(10, 6))

# 3. Plot Transformer data (MPS)
plt.plot(
    df['Seq_Len'],
    df['Transformer_Time_ms'],
    label='Transformer (MPS)',
    marker='o',
    linestyle='-',
    color='tab:blue',
    linewidth=2
)

# 4. Plot Mamba data (Metal)
plt.plot(
    df['Seq_Len'],
    df['Mamba_Time_ms'],
    label='Mamba (Metal)',
    marker='s', # Use square marker
    linestyle='--', # Use dashed line
    color='tab:orange',
    linewidth=2
)

# 5. Set up axes and title
# Key setting: Use logarithmic scale for the X-axis (Sequence Length)
plt.xscale('log', base=2) 

# For Y-axis, we use a linear scale to clearly show the huge jump at 2048.
# To emphasize the relative difference (and suppress the extreme jump), 
# uncomment the line below to use a logarithmic Y-axis scale:
# plt.yscale('log') 

plt.title('Inference Time Comparison: Transformer vs Mamba', fontsize=16)
plt.xlabel('Sequence Length (Seq Len)', fontsize=14)
plt.ylabel('Time (ms)', fontsize=14)

# Ensure X-axis ticks only show the data points (128, 512, 1024, etc.)
plt.xticks(df['Seq_Len'], labels=[str(s) for s in df['Seq_Len']], fontsize=12)
plt.yticks(fontsize=12)

# Add grid lines for readability
plt.grid(True, which="both", ls="--", alpha=0.6)

# 6. Add the legend
plt.legend(fontsize=12)

# 7. Display the plot
plt.tight_layout()
plt.show()