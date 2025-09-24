import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_and_prepare_data(csv_path):
    """Loads and prepares the comparison data for plotting with percentage calculations."""
    try:
        df = pd.read_csv(csv_path)
        if 'Before' in df.columns and 'After' in df.columns:
            df['Improvement'] = df['After'] - df['Before']
            df['Percentage Improvement'] = np.where(df['Before'] != 0, (df['Improvement'] / df['Before']) * 100, 0)
            df['Trend'] = df['Percentage Improvement'].apply(
                lambda x: 'Increase' if x > 0 else 'Decrease' if x < 0 else 'No Change')
            df['Trend_Symbol'] = df['Percentage Improvement'].apply(lambda x: '↑' if x > 0 else '↓' if x < 0 else '→')
        return df
    except FileNotFoundError:
        print(f"Error: '{csv_path}' file not found.")
        return None


def create_percentage_improvement_comparison_chart(df, colors):
    """Create horizontal bar chart showing percentage improvement for each metric."""
    fig, ax = plt.subplots(figsize=(18 / 2.54, 11 / 2.54))
    fig.patch.set_alpha(0)

    df_sorted = df.sort_values('Percentage Improvement', ascending=True)
    bar_colors = [colors['increase'] if val > 0 else colors['decrease'] if val < 0 else colors['neutral']
                  for val in df_sorted['Percentage Improvement']]

    bars = ax.barh(df_sorted['Metric'], df_sorted['Percentage Improvement'],
                   color=bar_colors, alpha=0.85, edgecolor='white', linewidth=1.5)

    ax.set_xlabel('Percentage Improvement (%)')
    ax.set_ylabel('Metrics')
    ax.set_title('Percentage Improvement by Metric (↑ Increase | ↓ Decrease)', pad=20)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--', color='gray')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)

    for i, bar in enumerate(bars):
        width = bar.get_width()
        metric_data = df_sorted.iloc[i]
        trend_symbol = metric_data['Trend_Symbol']
        x_pos = width + 1 if width >= 0 else width - 1
        ha_pos = 'left' if width >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{trend_symbol} {width:.2f}%', ha=ha_pos, va='center', fontsize=12,
                fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig('charts/percentage_improvement_comparison.svg', format='svg', bbox_inches='tight',
                transparent=True)
    plt.close()


def create_percentage_improvement_lollipop_chart(df, colors):
    """Create a vertical lollipop chart showing percentage improvement."""
    fig, ax = plt.subplots(figsize=(18 / 2.54, 12 / 2.54))  # slightly taller for breathing space
    fig.patch.set_alpha(0)

    # Sort metrics by percentage improvement
    df_sorted = df.sort_values('Percentage Improvement', ascending=False)
    
    # Choose colors based on trend
    dot_colors = [colors['increase'] if val > 0 else colors['decrease'] if val < 0 else colors['neutral']
                  for val in df_sorted['Percentage Improvement']]

    # Lollipop stems
    ax.vlines(x=np.arange(len(df_sorted)), ymin=0, ymax=df_sorted['Percentage Improvement'], 
              color=dot_colors, alpha=0.7, linewidth=2)
              
    # Lollipop dots
    ax.scatter(np.arange(len(df_sorted)), df_sorted['Percentage Improvement'], 
               color=dot_colors, s=150, alpha=1, zorder=3,
               edgecolor='white', linewidth=1.5)

    # Axis labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentage Improvement (%)')
    ax.set_title('Percentage Improvement by Metric', pad=25)
    
    # Replace x-axis ticks with metric names
    ax.set_xticks(np.arange(len(df_sorted)))
    ax.set_xticklabels(df_sorted['Metric'], rotation=45, ha='right')
    
    ax.grid(True, alpha=0.3, axis='y', linestyle='--', color='gray')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2)

    # --- Add value labels above/below dots ---
    x_positions = np.arange(len(df_sorted))
    metric_to_x = dict(zip(df_sorted['Metric'], x_positions))

    for i, row in df_sorted.iterrows():
        height = row['Percentage Improvement']
        trend_symbol = row['Trend_Symbol']
        
        y_offset = 5
        y_pos = height + y_offset if height >= 0 else height - y_offset
        va_pos = 'bottom' if height >= 0 else 'top'

        x_val = metric_to_x[row['Metric']]
        ha_pos = 'center'

        # Special case: nudge "bleu" to the right so it doesn’t overlap the y-axis
        if row['Metric'].lower() == "bleu":
            x_val += 0.1
            ha_pos = 'left'

        ax.text(x_val, y_pos,
                f"{trend_symbol} {height:.2f}%", 
                ha=ha_pos, va=va_pos,
                fontweight='bold', color='black')

    # Legend in top-right corner
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['increase'], label='Increase ↑')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Add breathing space above the highest bar
    ymax = df_sorted['Percentage Improvement'].max()
    ax.set_ylim(0, ymax * 1.6)

    # Adjust layout to leave room for legend and title
    plt.subplots_adjust(top=0.85, right=0.95)

    # Save chart
    plt.savefig('charts/percentage_improvement_lollipop.svg', 
                format='svg', bbox_inches='tight', transparent=True)
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
    fig.patch.set_alpha(0)

    ax.plot(angles, improvement_values, 'o-', linewidth=3,
            label='Improvement (%)', color=colors['improvement'], markersize=8,
            markerfacecolor=colors['improvement'], markeredgecolor='white', markeredgewidth=2)
    ax.fill(angles, improvement_values, alpha=0.25, color=colors['improvement'])

    for i, (angle, metric) in enumerate(zip(angles[:-1], metrics)):
        trend_symbol = df.iloc[i]['Trend_Symbol']
        trend_color = colors['increase'] if df.iloc[i]['Percentage Improvement'] > 0 else colors['decrease']
        val = improvement_values[i]
        radius = abs(val) * 1.2 if val != 0 else 5
        ax.text(angle, radius, trend_symbol, ha='center', va='center',
                fontsize=14, fontweight='bold', color=trend_color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Radar Chart: Percentage Improvement with Trends (↑↓)', pad=40)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True, alpha=0.4, linestyle='--', color='gray')
    max_abs_val = max(abs(val) for val in improvement_values[:-1]) if improvement_values[:-1] else 1
    ax.set_ylim(-max_abs_val * 1.3, max_abs_val * 1.3)

    plt.tight_layout()
    plt.savefig('charts/percentage_improvement_radar.svg', format='svg', bbox_inches='tight',
                transparent=True)
    plt.close()


def create_percentage_improvement_line_chart(df, colors):
    """Create line chart showing percentage improvement trends."""
    fig, ax = plt.subplots(figsize=(18 / 2.54, 11 / 2.54))
    fig.patch.set_alpha(0)

    df_sorted = df.sort_values('Metric')
    
    ax.plot(df_sorted['Metric'], df_sorted['Percentage Improvement'], marker='o', linewidth=3,
            label='Improvement (%)', color=colors['improvement'], markersize=10,
            markerfacecolor=colors['improvement'], markeredgecolor='white', markeredgewidth=2)

    for i, row in df_sorted.iterrows():
        y = row['Percentage Improvement']
        color = colors['increase'] if y > 0 else colors['decrease'] if y < 0 else colors['neutral']
        trend_symbol = row['Trend_Symbol']
        ax.scatter(row['Metric'], y, color=color, s=150, alpha=0.8, edgecolors='white', linewidth=2, zorder=5)
        y_offset = abs(y) * 0.1 + 2 if y >= 0 else -(abs(y) * 0.1 + 2)
        ax.text(row['Metric'], y + y_offset, trend_symbol, ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Percentage Improvement (%)')
    ax.set_title('Percentage Improvement Trends with Change Indicators (↑↓)', pad=20)
    plt.xticks(rotation=45, ha='right')

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--', color='gray')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig('charts/percentage_improvement_line.svg', format='svg', bbox_inches='tight',
                transparent=True)
    plt.close()


def generate_percentage_improvement_summary(df):
    """Generate summary statistics focused on percentage improvements."""
    print("\n" + "=" * 70)
    print("PERCENTAGE IMPROVEMENT ANALYSIS")
    print("=" * 70)
    print(f"Number of metrics analyzed: {len(df)}")
    print(f"Average percentage improvement: {df['Percentage Improvement'].mean():.2f}%")
    print(f"Total improvement range: {df['Percentage Improvement'].min():.2f}% to {df['Percentage Improvement'].max():.2f}%")

    increases = df[df['Percentage Improvement'] > 0]
    decreases = df[df['Percentage Improvement'] < 0]
    no_change = df[df['Percentage Improvement'] == 0]

    print(f"\nImprovement Distribution:")
    print(f"↑ Metrics showing improvement: {len(increases)} ({len(increases) / len(df) * 100:.1f}%)")
    print(f"↓ Metrics showing decline: {len(decreases)} ({len(decreases) / len(df) * 100:.1f}%)")
    print(f"→ Metrics with no change: {len(no_change)} ({len(no_change) / len(df) * 100:.1f}%)")

    if not increases.empty:
        best_metric = increases.loc[increases['Percentage Improvement'].idxmax()]
        print(f"\n🏆 Best improvement: {best_metric['Metric']}")
        print(f"   Percentage increase: +{best_metric['Percentage Improvement']:.2f}%")
        print(f"   From {best_metric['Before']:.4f} to {best_metric['After']:.4f}")

    if not decreases.empty:
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

    plt.rcParams.update({
        'font.size': 16,
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
        'legend.fontsize': 14,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'axes.axisbelow': True,
        'figure.facecolor': 'none',
        'axes.facecolor': 'none'
    })

    colors = {
        'before': '#E74C3C',
        'after': '#3498DB',
        'improvement': '#9B59B6',
        'increase': '#27AE60',
        'decrease': '#E74C3C',
        'neutral': '#95A5A6',
    }

    df = load_and_prepare_data('metrics.csv')

    if df is not None:
        print(f"Dataset loaded: {df.shape[0]} metrics, {df.shape[1]} columns")
        print("\nGenerating percentage improvement charts (SVG format for high quality)...")

        create_percentage_improvement_comparison_chart(df, colors)
        print("✓ Horizontal improvement chart saved: charts/percentage_improvement_comparison.svg")

        create_percentage_improvement_lollipop_chart(df, colors)
        print("✓ Lollipop improvement chart saved: charts/percentage_improvement_lollipop.svg")

        create_percentage_improvement_radar_chart(df, colors)
        print("✓ Improvement radar chart saved: charts/percentage_improvement_radar.svg")

        create_percentage_improvement_line_chart(df, colors)
        print("✓ Improvement line chart saved: charts/percentage_improvement_line.svg")

        generate_percentage_improvement_summary(df)