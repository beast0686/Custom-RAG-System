import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_and_prepare_data(csv_path):
    """Loads and prepares the comparison data for plotting with percentage calculations."""
    try:
        df = pd.read_csv(csv_path)
        if 'Before' in df.columns and 'After' in df.columns:
            # Calculate improvement and percentage improvement
            df['Improvement'] = df['After'] - df['Before']
            df['Percentage Improvement'] = np.where(df['Before'] != 0, (df['Improvement'] / df['Before']) * 100, 0)

            # Convert values to percentages for display
            df['Before_Pct'] = df['Before'] * 100
            df['After_Pct'] = df['After'] * 100
            df['Improvement_Pct'] = df['Improvement'] * 100

            # Add trend indicators
            df['Trend'] = df['Percentage Improvement'].apply(
                lambda x: 'Increase' if x > 0 else 'Decrease' if x < 0 else 'No Change')
            df['Trend_Symbol'] = df['Percentage Improvement'].apply(lambda x: '↑' if x > 0 else '↓' if x < 0 else '→')

        return df
    except FileNotFoundError:
        print(f"Error: '{csv_path}' file not found.")
        return None


def create_percentage_comparison_chart(df, colors):
    """Create grouped bar chart with percentage values and trend indicators."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(df['Metric']))
    width = 0.35

    bars1 = ax.bar(x - width / 2, df['Before_Pct'], width,
                   label='Before (%)', color=colors['before'], alpha=0.85,
                   edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width / 2, df['After_Pct'], width,
                   label='After (%)', color=colors['after'], alpha=0.85,
                   edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Metrics', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=10, fontweight='bold')
    ax.set_title('Before vs After Performance Comparison with Trend Indicators', fontweight='bold', fontsize=12, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Metric'], rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=10, loc='upper left', fancybox=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')

    # Add value labels on bars with trend indicators
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        # Before value
        height1 = bar1.get_height()
        ax.text(bar1.get_x() + bar1.get_width() / 2., height1 + 0.5,
                f'{height1:.1f}%', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='black')

        # After value with trend symbol
        height2 = bar2.get_height()
        trend_symbol = df.iloc[i]['Trend_Symbol']
        trend_color = colors['increase'] if df.iloc[i]['Percentage Improvement'] > 0 else colors['decrease']

        ax.text(bar2.get_x() + bar2.get_width() / 2., height2 + 0.5,
                f'{height2:.1f}%', ha='center', va='bottom', fontsize=8,
                fontweight='bold', color='black')

        # Add trend arrow above the after bar
        ax.text(bar2.get_x() + bar2.get_width() / 2., height2 + 2.5,
                trend_symbol, ha='center', va='bottom', fontsize=12,
                fontweight='bold', color=trend_color)

    plt.tight_layout()
    plt.savefig('charts/percentage_comparison_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_percentage_improvement_chart(df, colors):
    """Create improvement chart showing percentage change with color coding."""
    fig, ax = plt.subplots(figsize=(12, 7))

    df_sorted = df.sort_values('Percentage Improvement', ascending=False)

    # Color bars based on increase/decrease
    bar_colors = [colors['increase'] if val > 0 else colors['decrease'] if val < 0 else colors['neutral']
                  for val in df_sorted['Percentage Improvement']]

    bars = ax.bar(df_sorted['Metric'], df_sorted['Percentage Improvement'],
                  color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Metrics', fontsize=10, fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontsize=10, fontweight='bold')
    ax.set_title('Percentage Improvement by Metric (↑ Increase | ↓ Decrease)', fontweight='bold', fontsize=12, pad=20)
    ax.tick_params(axis='x', rotation=30, labelsize=9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Add value labels on bars with trend symbols
    for i, bar in enumerate(bars):
        height = bar.get_height()
        metric_data = df_sorted.iloc[i]
        trend_symbol = metric_data['Trend_Symbol']

        # Position text above or below bar depending on positive/negative
        y_pos = height + abs(height) * 0.02 if height >= 0 else height - abs(height) * 0.02
        va_pos = 'bottom' if height >= 0 else 'top'

        ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                f'{trend_symbol} {height:.2f}%', ha='center', va=va_pos, fontsize=8,
                fontweight='bold', color='black')

    # Add legend for trend indicators
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['increase'], label='Increase ↑'),
        Patch(facecolor=colors['decrease'], label='Decrease ↓'),
        Patch(facecolor=colors['neutral'], label='No Change →')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig('charts/percentage_improvement_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_percentage_radar_chart(df, colors):
    """Create radar chart with percentage values and trend indicators."""
    metrics = df['Metric'].tolist()
    before_values = df['Before_Pct'].tolist()
    after_values = df['After_Pct'].tolist()

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    before_values += before_values[:1]
    after_values += after_values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    ax.plot(angles, before_values, 'o-', linewidth=3,
            label='Before (%)', color=colors['before'], markersize=8,
            markerfacecolor=colors['before'], markeredgecolor='white', markeredgewidth=2)
    ax.fill(angles, before_values, alpha=0.25, color=colors['before'])

    ax.plot(angles, after_values, 's-', linewidth=3,
            label='After (%)', color=colors['after'], markersize=8,
            markerfacecolor=colors['after'], markeredgecolor='white', markeredgewidth=2)
    ax.fill(angles, after_values, alpha=0.25, color=colors['after'])

    # Add trend symbols for each metric
    for i, (angle, metric) in enumerate(zip(angles[:-1], metrics)):
        trend_symbol = df.iloc[i]['Trend_Symbol']
        trend_color = colors['increase'] if df.iloc[i]['Percentage Improvement'] > 0 else colors['decrease']
        max_val = max(before_values[i], after_values[i])

        ax.text(angle, max_val * 1.15, trend_symbol, ha='center', va='center',
                fontsize=14, fontweight='bold', color=trend_color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9, fontweight='bold')
    ax.set_title('Radar Chart: Performance Comparison with Trends (↑↓)', fontweight='bold', fontsize=12, pad=40)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10, fancybox=True, framealpha=0.9)
    ax.grid(True, alpha=0.4, linestyle='--', color='gray')

    max_val = max(max(df['Before_Pct']), max(df['After_Pct']))
    ax.set_ylim(0, max_val * 1.2)

    plt.tight_layout()
    plt.savefig('charts/percentage_radar_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_percentage_line_chart(df, colors):
    """Create line chart with percentage values and trend indicators."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = range(len(df['Metric']))

    ax.plot(x_pos, df['Before_Pct'], marker='o', linewidth=3,
            label='Before (%)', color=colors['before'], markersize=8,
            markerfacecolor=colors['before'], markeredgecolor='white', markeredgewidth=2)
    ax.plot(x_pos, df['After_Pct'], marker='s', linewidth=3,
            label='After (%)', color=colors['after'], markersize=8,
            markerfacecolor=colors['after'], markeredgecolor='white', markeredgewidth=2)

    # Add trend arrows between points
    for i in range(len(df)):
        before_val = df.iloc[i]['Before_Pct']
        after_val = df.iloc[i]['After_Pct']
        trend_symbol = df.iloc[i]['Trend_Symbol']
        trend_color = colors['increase'] if df.iloc[i]['Percentage Improvement'] > 0 else colors['decrease']

        mid_y = (before_val + after_val) / 2
        ax.annotate(trend_symbol, xy=(i, mid_y), fontsize=12, fontweight='bold',
                    color=trend_color, ha='center', va='center')

    ax.set_xlabel('Metrics', fontsize=10, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=10, fontweight='bold')
    ax.set_title('Performance Trends with Change Indicators (↑↓)', fontweight='bold', fontsize=12, pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df['Metric'], rotation=30, ha='right', fontsize=9)
    ax.legend(fontsize=10, loc='upper left', fancybox=True, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')

    plt.tight_layout()
    plt.savefig('charts/percentage_line_chart.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def generate_percentage_summary_stats(df):
    """Generate detailed summary statistics with trend analysis."""
    print("\n" + "=" * 70)
    print("PERCENTAGE-BASED DATASET SUMMARY WITH TREND ANALYSIS")
    print("=" * 70)
    print(f"Number of metrics: {len(df)}")
    print(f"Average percentage improvement: {df['Percentage Improvement'].mean():.2f}%")

    increases = df[df['Percentage Improvement'] > 0]
    decreases = df[df['Percentage Improvement'] < 0]
    no_change = df[df['Percentage Improvement'] == 0]

    print(f"\nTrend Analysis:")
    print(f"↑ Metrics with increase: {len(increases)} ({len(increases) / len(df) * 100:.1f}%)")
    print(f"↓ Metrics with decrease: {len(decreases)} ({len(decreases) / len(df) * 100:.1f}%)")
    print(f"→ Metrics with no change: {len(no_change)} ({len(no_change) / len(df) * 100:.1f}%)")

    if len(increases) > 0:
        best_metric = increases.loc[increases['Percentage Improvement'].idxmax()]
        print(f"\n↑ Best performing metric: {best_metric['Metric']}")
        print(f"   Largest improvement: +{best_metric['Percentage Improvement']:.2f}%")
        print(f"   Average improvement: +{increases['Percentage Improvement'].mean():.2f}%")

    if len(decreases) > 0:
        worst_metric = decreases.loc[decreases['Percentage Improvement'].idxmin()]
        print(f"\n↓ Worst performing metric: {worst_metric['Metric']}")
        print(f"   Largest decrease: {worst_metric['Percentage Improvement']:.2f}%")
        print(f"   Average decrease: {decreases['Percentage Improvement'].mean():.2f}%")

    print("\nDetailed metrics data with trends:")
    display_df = df[['Metric', 'Before_Pct', 'After_Pct', 'Percentage Improvement', 'Trend_Symbol']].copy()
    display_df.columns = ['Metric', 'Before (%)', 'After (%)', 'Change (%)', 'Trend']
    print(display_df.to_string(index=False, float_format='%.2f'))


# Main execution
if __name__ == "__main__":
    os.makedirs('charts', exist_ok=True)

    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'Times New Roman',
        'font.weight': 'normal',
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'grid.linestyle': '--',
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.framealpha': 0.9,
        'legend.facecolor': 'white',
        'legend.edgecolor': 'gray',
        'legend.fontsize': 10,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'axes.axisbelow': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

    # Enhanced color palette with trend indicators
    colors = {
        'before': '#E74C3C',  # Vibrant red
        'after': '#3498DB',  # Bright blue
        'improvement': '#2ECC71',  # Emerald green
        'increase': '#27AE60',  # Green for increases
        'decrease': '#E74C3C',  # Red for decreases
        'neutral': '#95A5A6',  # Gray for no change
        'improvement_gradient': [
            '#2ECC71', '#F39C12', '#9B59B6', '#E67E22',
            '#1ABC9C', '#34495E', '#E91E63', '#FF5722'
        ]
    }

    df = load_and_prepare_data('metrics.csv')

    if df is not None:
        print(f"Dataset loaded: {df.shape[0]} metrics, {df.shape[1]} columns")
        print("\nGenerating enhanced charts with trend indicators...")

        create_percentage_comparison_chart(df, colors)
        print("✓ Enhanced comparison chart with trends saved: charts/percentage_comparison_chart.png")

        create_percentage_improvement_chart(df, colors)
        print("✓ Color-coded improvement chart saved: charts/percentage_improvement_chart.png")

        create_percentage_radar_chart(df, colors)
        print("✓ Radar chart with trend indicators saved: charts/percentage_radar_chart.png")

        create_percentage_line_chart(df, colors)
        print("✓ Line chart with change indicators saved: charts/percentage_line_chart.png")

        generate_percentage_summary_stats(df)
