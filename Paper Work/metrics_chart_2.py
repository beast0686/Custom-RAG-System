import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_prepare_data(csv_path):
    """Loads and prepares the comparison data for plotting."""
    try:
        df = pd.read_csv(csv_path)
        if 'Before' in df.columns and 'After' in df.columns:
            df['Improvement'] = df['After'] - df['Before']
            # Calculate Percentage Improvement, handling potential division by zero
            df['Percentage Improvement'] = np.where(df['Before'] != 0, (df['Improvement'] / df['Before']) * 100, 0)
        return df
    except FileNotFoundError:
        print(f"Error: '{csv_path}' file not found.")
        print("Please ensure the CSV file is in the current directory.")
        return None

def create_faceted_bar_chart(df, colors):
    """
    Create a single figure with two faceted (paneled) plots to visualize
    metrics with different scales.
    """
    high_value_metrics = ['factual_accuracy', 'completeness', 'coherence', 'helpfulness']
    low_value_metrics = ['bleu', 'rouge_l']
    df_high = df[df['Metric'].isin(high_value_metrics)].set_index('Metric')
    df_low = df[df['Metric'].isin(low_value_metrics)].set_index('Metric')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Before vs After Performance Comparison', fontweight='bold')
    width = 0.35

    x_high = np.arange(len(df_high))
    ax1.bar(x_high - width/2, df_high['Before'], width, label='Before', color=colors['before'], alpha=0.8)
    ax1.bar(x_high + width/2, df_high['After'], width, label='After', color=colors['after'], alpha=0.8)
    ax1.set_ylabel('Score (Scale 1-5)')
    ax1.set_title('Quality and Helpfulness Metrics', pad=10)
    ax1.set_xticks(x_high)
    ax1.set_xticklabels(df_high.index, rotation=15, ha='right')
    ax1.set_ylim(0, 5)

    x_low = np.arange(len(df_low))
    ax2.bar(x_low - width/2, df_low['Before'], width, label='Before', color=colors['before'], alpha=0.8)
    ax2.bar(x_low + width/2, df_low['After'], width, label='After', color=colors['after'], alpha=0.8)
    ax2.set_xlabel('Metric')
    ax2.set_ylabel('Score (Scale 0-1)')
    ax2.set_title('Text Similarity Metrics', pad=10)
    ax2.set_xticks(x_low)
    ax2.set_xticklabels(df_low.index)
    ax2.set_ylim(0, max(df_low['Before'].max(), df_low['After'].max()) * 1.2)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.95))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('charts2/faceted_bar_chart.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Faceted Before/After chart saved: charts/faceted_bar_chart.png")

def create_faceted_improvement_chart(df, colors):
    """Create a faceted bar chart for absolute improvements."""
    high_imp_metrics = ['factual_accuracy', 'completeness', 'coherence', 'helpfulness']
    low_imp_metrics = ['bleu', 'rouge_l']
    df_high = df[df['Metric'].isin(high_imp_metrics)].set_index('Metric')
    df_low = df[df['Metric'].isin(low_imp_metrics)].set_index('Metric')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=False, gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('Absolute Improvement by Metric (After - Before)', fontweight='bold')

    ax1.bar(df_high.index, df_high['Improvement'], color=colors['improvement'], alpha=0.8)
    ax1.set_ylabel('Absolute Improvement')
    ax1.tick_params(axis='x', rotation=15, labelsize=9)

    ax2.bar(df_low.index, df_low['Improvement'], color=colors['improvement'], alpha=0.8)
    ax2.set_ylabel('Absolute Improvement')
    ax2.tick_params(axis='x', rotation=0)

    for ax in [ax1, ax2]:
        for bar in ax.patches:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('charts2/faceted_improvement_chart.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Faceted improvement chart saved: charts/faceted_improvement_chart.png")

def create_percentage_improvement_chart(df, colors):
    """Create a bar chart showing percentage improvement."""
    fig, ax = plt.subplots(figsize=(8, 5))
    df_sorted = df.sort_values('Percentage Improvement', ascending=False)
    bars = ax.bar(df_sorted['Metric'], df_sorted['Percentage Improvement'], color=colors['improvement'], alpha=0.8)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('Percentage Improvement by Metric', fontweight='bold', pad=15)
    ax.tick_params(axis='x', rotation=25)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.2f}%', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig('charts2/percentage_improvement_chart.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print("✓ Percentage improvement chart saved: charts/percentage_improvement_chart.png")

def generate_summary_stats(df):
    """Generate summary statistics for the dataset"""
    print("\n" + "=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Number of metrics analyzed: {len(df)}")
    best_abs = df.loc[df['Improvement'].idxmax()]
    best_pct = df.loc[df['Percentage Improvement'].idxmax()]
    print(f"Largest absolute improvement in '{best_abs['Metric']}' ({best_abs['Improvement']:.4f})")
    print(f"Largest percentage improvement in '{best_pct['Metric']}' ({best_pct['Percentage Improvement']:.2f}%)")
    print("\nMetrics data:")
    print(df[['Metric', 'Before', 'After', 'Improvement', 'Percentage Improvement']].to_string(index=False, float_format='%.4f'))

# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs('charts', exist_ok=True)
    plt.rcParams.update({
        'font.size': 10, 'font.family': 'serif', 'axes.linewidth': 0.8,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.linewidth': 0.5, 'legend.frameon': False,
        'legend.fontsize': 9, 'xtick.major.size': 3, 'ytick.major.size': 3,
        'axes.axisbelow': True
    })
    colors = {'before': '#E74C3C', 'after': '#2E86C1', 'improvement': '#28B463'}
    df = load_and_prepare_data('metrics.csv')

    if df is not None:
        print(f"Dataset loaded: {df.shape[0]} metrics.")
        print("\nGenerating charts...")
        create_faceted_bar_chart(df, colors)
        create_faceted_improvement_chart(df, colors)
        create_percentage_improvement_chart(df, colors)
        generate_summary_stats(df)

