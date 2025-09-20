import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Create charts directory if it doesn't exist
os.makedirs('charts', exist_ok=True)

# Load dataset
df = pd.read_csv('metrics.csv')

# Set matplotlib parameters for publication-ready figures
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
    'legend.frameon': False,
    'legend.fontsize': 9,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'axes.axisbelow': True
})

# Define professional color palette
colors = {
    'before': '#E74C3C',  # Professional red
    'after': '#2E86C1',  # Professional blue
    'improvement': '#28B463'  # Professional green
}


def create_grouped_bar_chart():
    """Create grouped bar chart comparing Before vs After"""

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(df['Metric']))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df['Before'], width,
                   label='Before', color=colors['before'], alpha=0.8)
    bars2 = ax.bar(x + width / 2, df['After'], width,
                   label='After', color=colors['after'], alpha=0.8)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Before vs After Performance Comparison', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Metric'], rotation=25, ha='right')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('charts/grouped_bar_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_improvement_chart():
    """Create improvement bar chart"""

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(df['Metric'], df['Improvement'],
                  color=colors['improvement'], alpha=0.8)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Improvement')
    ax.set_title('Performance Improvement by Metric', fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('charts/improvement_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_radar_chart():
    """Create radar chart for comprehensive metric comparison"""

    metrics = df['Metric'].tolist()
    before_values = df['Before'].tolist()
    after_values = df['After'].tolist()

    # Number of variables
    N = len(metrics)

    # Compute angles for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Close the plot data
    before_values += before_values[:1]
    after_values += after_values[:1]

    # Create figure with polar subplot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))

    # Plot data with clean styling
    ax.plot(angles, before_values, 'o-', linewidth=2,
            label='Before', color=colors['before'], markersize=5)
    ax.fill(angles, before_values, alpha=0.15, color=colors['before'])

    ax.plot(angles, after_values, 's-', linewidth=2,
            label='After', color=colors['after'], markersize=5)
    ax.fill(angles, after_values, alpha=0.15, color=colors['after'])

    # Customize the radar chart
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_title('Radar Chart: Before vs After Performance Profile',
                 fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax.grid(True, alpha=0.3)

    # Set radial limits for better visualization
    ax.set_ylim(0, max(max(df['Before']), max(df['After'])) * 1.1)

    plt.tight_layout()
    plt.savefig('charts/radar_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_line_chart():
    """Create line chart showing trends across metrics"""

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df['Metric'], df['Before'], marker='o', linewidth=2,
            label='Before', color=colors['before'], markersize=6)
    ax.plot(df['Metric'], df['After'], marker='s', linewidth=2,
            label='After', color=colors['after'], markersize=6)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Performance Trends: Before vs After', fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=25)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('charts/line_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def generate_summary_stats():
    """Generate summary statistics for the dataset"""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Number of metrics: {len(df)}")
    print(f"Average improvement: {df['Improvement'].mean():.4f}")
    print(f"Best performing metric: {df.loc[df['Improvement'].idxmax(), 'Metric']}")
    print(f"Largest improvement: {df['Improvement'].max():.4f}")
    print("\nMetrics data:")
    print(df.to_string(index=False, float_format='%.4f'))


# Main execution
if __name__ == "__main__":
    try:
        print("Loading metrics.csv...")
        print(f"Dataset loaded: {df.shape[0]} metrics, {df.shape[1]} columns")

        print("\nGenerating individual charts...")

        # Create all charts separately
        create_grouped_bar_chart()
        print("✓ Grouped bar chart saved: charts/grouped_bar_chart.png")

        create_improvement_chart()
        print("✓ Improvement chart saved: charts/improvement_chart.png")

        create_radar_chart()
        print("✓ Radar chart saved: charts/radar_chart.png")

        create_line_chart()
        print("✓ Line chart saved: charts/line_chart.png")

        # Display summary
        generate_summary_stats()

    except FileNotFoundError:
        print("Error: 'metrics.csv' file not found.")
        print("Please ensure the CSV file is in the current directory.")
    except Exception as e:
        print(f"Error: {e}")
