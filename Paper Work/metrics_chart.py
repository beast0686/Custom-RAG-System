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

            # Add trend indicators
            df['Trend'] = df['Percentage Improvement'].apply(
                lambda x: 'Increase' if x > 0 else 'Decrease' if x < 0 else 'No Change')
            df['Trend_Symbol'] = df['Percentage Improvement'].apply(lambda x: '↑' if x > 0 else '↓' if x < 0 else '→')

        return df
    except FileNotFoundError:
        print(f"Error: '{csv_path}' file not found.")
        return None


def create_percentage_improvement_comparison_chart(df, colors):
    """Create bar chart showing percentage improvement for each metric."""
    fig, ax = plt.subplots(figsize=(12, 7))

    # Sort by percentage improvement for better visualization
    df_sorted = df.sort_values('Percentage Improvement', ascending=True)

    # Color bars based on increase/decrease
    bar_colors = [colors['increase'] if val > 0 else colors['decrease'] if val < 0 else colors['neutral']
                  for val in df_sorted['Percentage Improvement']]

    bars = ax.barh(df_sorted['Metric'], df_sorted['Percentage Improvement'],
                   color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Percentage Improvement (%)', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('Metrics', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_title('Percentage Improvement by Metric (↑ Increase | ↓ Decrease)',
                 fontweight='bold', fontsize=14, fontfamily='Times New Roman', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--', color='gray')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    # Set tick label font
    ax.tick_params(axis='both', labelsize=12)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

    # Add value labels on bars with trend symbols
    for i, bar in enumerate(bars):
        width = bar.get_width()
        metric_data = df_sorted.iloc[i]
        trend_symbol = metric_data['Trend_Symbol']

        # Position text to the right or left of bar depending on positive/negative
        x_pos = width + abs(width) * 0.02 if width >= 0 else width - abs(width) * 0.02
        ha_pos = 'left' if width >= 0 else 'right'

        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{trend_symbol} {width:.2f}%', ha=ha_pos, va='center', fontsize=11,
                fontweight='bold', color='black', fontfamily='Times New Roman')

    plt.tight_layout()
    plt.savefig('charts/percentage_improvement_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_percentage_improvement_chart(df, colors):
    """Create vertical bar chart showing percentage improvement."""
    fig, ax = plt.subplots(figsize=(12, 7))

    df_sorted = df.sort_values('Percentage Improvement', ascending=False)

    # Color bars based on increase/decrease
    bar_colors = [colors['increase'] if val > 0 else colors['decrease'] if val < 0 else colors['neutral']
                  for val in df_sorted['Percentage Improvement']]

    bars = ax.bar(df_sorted['Metric'], df_sorted['Percentage Improvement'],
                  color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('Percentage Improvement (%)', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_title('Percentage Improvement by Metric (↑ Increase | ↓ Decrease)',
                 fontweight='bold', fontsize=14, fontfamily='Times New Roman', pad=20)
    ax.tick_params(axis='x', rotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # Set tick label font
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

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
                f'{trend_symbol} {height:.2f}%', ha='center', va=va_pos, fontsize=11,
                fontweight='bold', color='black', fontfamily='Times New Roman')

    # Add legend for trend indicators
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['increase'], label='Increase ↑'),
        Patch(facecolor=colors['decrease'], label='Decrease ↓'),
        Patch(facecolor=colors['neutral'], label='No Change →')
    ]
    legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')

    plt.tight_layout()
    plt.savefig('charts/percentage_improvement_vertical.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_percentage_improvement_radar_chart(df, colors):
    """Create radar chart showing percentage improvement values."""
    metrics = df['Metric'].tolist()
    improvement_values = df['Percentage Improvement'].tolist()

    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    improvement_values += improvement_values[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Color code the line based on improvements
    ax.plot(angles, improvement_values, 'o-', linewidth=3,
            label='Improvement (%)', color=colors['improvement'], markersize=8,
            markerfacecolor=colors['improvement'], markeredgecolor='white', markeredgewidth=2)
    ax.fill(angles, improvement_values, alpha=0.25, color=colors['improvement'])

    # Add trend symbols for each metric
    for i, (angle, metric) in enumerate(zip(angles[:-1], metrics)):
        trend_symbol = df.iloc[i]['Trend_Symbol']
        trend_color = colors['increase'] if df.iloc[i]['Percentage Improvement'] > 0 else colors['decrease']
        val = improvement_values[i]

        # Position symbols outside the data points
        radius = abs(val) * 1.2 if val != 0 else 5
        ax.text(angle, radius, trend_symbol, ha='center', va='center',
                fontsize=14, fontweight='bold', color=trend_color, fontfamily='Times New Roman')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_title('Radar Chart: Percentage Improvement with Trends (↑↓)',
                 fontweight='bold', fontsize=14, fontfamily='Times New Roman', pad=40)

    # Set radial tick labels font
    ax.tick_params(axis='y', labelsize=12)
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12,
                       fancybox=True, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')

    ax.grid(True, alpha=0.4, linestyle='--', color='gray')

    # Set appropriate scale for improvement values
    max_abs_val = max(abs(val) for val in improvement_values[:-1])
    ax.set_ylim(-max_abs_val * 1.3, max_abs_val * 1.3)

    plt.tight_layout()
    plt.savefig('charts/percentage_improvement_radar.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def create_percentage_improvement_line_chart(df, colors):
    """Create line chart showing percentage improvement trends."""
    fig, ax = plt.subplots(figsize=(12, 7))

    x_pos = range(len(df['Metric']))

    # Sort by metric name for consistent ordering
    df_sorted = df.sort_values('Metric')

    ax.plot(x_pos, df_sorted['Percentage Improvement'], marker='o', linewidth=3,
            label='Improvement (%)', color=colors['improvement'], markersize=10,
            markerfacecolor=colors['improvement'], markeredgecolor='white', markeredgewidth=2)

    # Color code points based on increase/decrease
    for i, (x, y) in enumerate(zip(x_pos, df_sorted['Percentage Improvement'])):
        color = colors['increase'] if y > 0 else colors['decrease'] if y < 0 else colors['neutral']
        trend_symbol = df_sorted.iloc[i]['Trend_Symbol']

        # Add colored markers
        ax.scatter(x, y, color=color, s=150, alpha=0.8, edgecolors='white', linewidth=2, zorder=5)

        # Add trend symbols above/below points
        y_offset = abs(y) * 0.1 + 2 if y >= 0 else -(abs(y) * 0.1 + 2)
        ax.text(x, y + y_offset, trend_symbol, ha='center', va='center',
                fontsize=12, fontweight='bold', color=color, fontfamily='Times New Roman')

    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_ylabel('Percentage Improvement (%)', fontsize=12, fontweight='bold', fontfamily='Times New Roman')
    ax.set_title('Percentage Improvement Trends with Change Indicators (↑↓)',
                 fontweight='bold', fontsize=14, fontfamily='Times New Roman', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(df_sorted['Metric'], rotation=45, ha='right', fontsize=12, fontfamily='Times New Roman')
    ax.tick_params(axis='y', labelsize=12)

    # Set tick label font
    for tick in ax.get_yticklabels():
        tick.set_fontfamily('Times New Roman')

    legend = ax.legend(fontsize=12, loc='upper left', fancybox=True, framealpha=0.9)
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')

    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig('charts/percentage_improvement_line.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def generate_percentage_improvement_summary(df):
    """Generate summary statistics focused on percentage improvements."""
    print("\n" + "=" * 70)
    print("PERCENTAGE IMPROVEMENT ANALYSIS")
    print("=" * 70)
    print(f"Number of metrics analyzed: {len(df)}")
    print(f"Average percentage improvement: {df['Percentage Improvement'].mean():.2f}%")
    print(
        f"Total improvement range: {df['Percentage Improvement'].min():.2f}% to {df['Percentage Improvement'].max():.2f}%")

    increases = df[df['Percentage Improvement'] > 0]
    decreases = df[df['Percentage Improvement'] < 0]
    no_change = df[df['Percentage Improvement'] == 0]

    print(f"\nImprovement Distribution:")
    print(f"↑ Metrics showing improvement: {len(increases)} ({len(increases) / len(df) * 100:.1f}%)")
    print(f"↓ Metrics showing decline: {len(decreases)} ({len(decreases) / len(df) * 100:.1f}%)")
    print(f"→ Metrics with no change: {len(no_change)} ({len(no_change) / len(df) * 100:.1f}%)")

    if len(increases) > 0:
        best_metric = increases.loc[increases['Percentage Improvement'].idxmax()]
        print(f"\n🏆 Best improvement: {best_metric['Metric']}")
        print(f"   Percentage increase: +{best_metric['Percentage Improvement']:.2f}%")
        print(f"   From {best_metric['Before']:.4f} to {best_metric['After']:.4f}")

    if len(decreases) > 0:
        worst_metric = decreases.loc[decreases['Percentage Improvement'].idxmin()]
        print(f"\n⚠️  Largest decline: {worst_metric['Metric']}")
        print(f"   Percentage decrease: {worst_metric['Percentage Improvement']:.2f}%")
        print(f"   From {worst_metric['Before']:.4f} to {worst_metric['After']:.4f}")

    print(f"\nRanked by Percentage Improvement:")
    display_df = df.sort_values('Percentage Improvement', ascending=False)[
        ['Metric', 'Before', 'After', 'Percentage Improvement', 'Trend_Symbol']].copy()
    display_df.columns = ['Metric', 'Before', 'After', 'Improvement (%)', 'Trend']
    print(display_df.to_string(index=False, float_format='%.4f'))


# Main execution
if __name__ == "__main__":
    os.makedirs('charts', exist_ok=True)

    # Professional font settings for research papers
    plt.rcParams.update({
        'font.size': 12,  # Base font size for body text
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
        'legend.fontsize': 12,  # Legend text size
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.labelsize': 12,  # X-axis tick label size
        'ytick.labelsize': 12,  # Y-axis tick label size
        'axes.labelsize': 12,  # Axis label size
        'axes.titlesize': 14,  # Title size for headings
        'axes.axisbelow': True,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white'
    })

    # Enhanced color palette for improvement visualization
    colors = {
        'before': '#E74C3C',
        'after': '#3498DB',
        'improvement': '#9B59B6',  # Purple for improvement line
        'increase': '#27AE60',  # Green for increases
        'decrease': '#E74C3C',  # Red for decreases
        'neutral': '#95A5A6',  # Gray for no change
    }

    df = load_and_prepare_data('metrics.csv')

    if df is not None:
        print(f"Dataset loaded: {df.shape[0]} metrics, {df.shape[1]} columns")
        print("\nGenerating percentage improvement charts...")

        create_percentage_improvement_comparison_chart(df, colors)
        print("✓ Horizontal improvement chart saved: charts/percentage_improvement_comparison.png")

        create_percentage_improvement_chart(df, colors)
        print("✓ Vertical improvement chart saved: charts/percentage_improvement_vertical.png")

        create_percentage_improvement_radar_chart(df, colors)
        print("✓ Improvement radar chart saved: charts/percentage_improvement_radar.png")

        create_percentage_improvement_line_chart(df, colors)
        print("✓ Improvement line chart saved: charts/percentage_improvement_line.png")

        generate_percentage_improvement_summary(df)
