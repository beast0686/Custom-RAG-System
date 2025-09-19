import json
import csv

def compare_metrics(file1_path, file2_path, output_csv_path):
    """
    Compares metrics from two JSON files and exports the result to a CSV file.

    Args:
        file1_path (str): Path to the first JSON file.
        file2_path (str): Path to the second JSON file.
        output_csv_path (str): Path to the output CSV file.
    """
    try:
        with open(file1_path, 'r') as f1:
            data1 = json.load(f1)
        with open(file2_path, 'r') as f2:
            data2 = json.load(f2)
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure the JSON files are in the correct directory.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    header = ['Model', 'Metric', 'File1_Average', 'File2_Average', 'Count', 'Average_of_Both', 'Better']
    rows = []

    # Use a comprehensive set of models from both files
    all_models = set(data1.keys()) | set(data2.keys())

    for model in sorted(list(all_models)):
        if model in data1 and model in data2:
            all_metrics = set(data1[model].keys()) | set(data2[model].keys())
            for metric in sorted(list(all_metrics)):
                if metric in data1[model] and metric in data2[model]:
                    avg1 = data1[model][metric]['average']
                    avg2 = data2[model][metric]['average']
                    # Assuming count is the same, taking from file 1
                    count = data1[model][metric]['count']
                    avg_both = (avg1 + avg2) / 2
                    better = max(avg1, avg2)

                    rows.append([model, metric, avg1, avg2, count, avg_both, better])

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Successfully created comparison file: {output_csv_path}")
    except IOError as e:
        print(f"Error writing to CSV file: {e}")

# --- Main Execution ---
# The script will now read these files directly from the directory it is run in.
# Make sure 'final_metrics.json' and 'final_metrics_2.json' exist.
print("Starting metrics comparison...")
compare_metrics('final_metrics.json', 'final_metrics_2.json', 'comparison.csv')
print("Comparison finished.")

