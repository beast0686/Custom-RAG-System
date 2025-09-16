import json
from collections import defaultdict

def compute_average_metrics(input_filename="feedback_metrics.json", output_filename="final_metrics.json"):
    """
    Reads a JSON file with feedback metrics, calculates the average for each metric
    per model type, and writes the result to a new JSON file.

    Args:
        input_filename (str): The name of the input JSON file.
        output_filename (str): The name of the output JSON file.
    """
    try:
        with open(input_filename, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded {len(data)} records from '{input_filename}'.")
    except FileNotFoundError:
        print(f"Error: The file '{input_filename}' was not found.")
        print("Please make sure the JSON file is in the same directory as the script.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{input_filename}' is not a valid JSON file.")
        return

    # Initialize a structure to hold the sum and count for each metric
    # defaultdict simplifies adding new keys without checking if they exist
    aggregated_data = defaultdict(lambda: defaultdict(lambda: {'sum': 0, 'count': 0}))

    # Use a set to track session_ids for which calculated_metrics have been processed
    # This prevents triple-counting these metrics for each session
    processed_sessions_for_calc_metrics = set()

    # --- Data Aggregation ---
    for record in data:
        model_type = record.get("model_type")
        session_id = record.get("session_id")
        
        if not model_type:
            continue

        # 1. Process human ratings (specific to each record's model_type)
        human_ratings = record.get("human_ratings", {})
        for metric, value in human_ratings.items():
            aggregated_data[model_type][metric]['sum'] += value
            aggregated_data[model_type][metric]['count'] += 1

        # 2. Process calculated metrics (once per session)
        if session_id and session_id not in processed_sessions_for_calc_metrics:
            calculated_metrics = record.get("calculated_metrics", {})
            for m_type, metrics in calculated_metrics.items():
                for metric, value in metrics.items():
                    aggregated_data[m_type][metric]['sum'] += value
                    aggregated_data[m_type][metric]['count'] += 1
            processed_sessions_for_calc_metrics.add(session_id)
            
    # --- Averages Calculation ---
    final_metrics = defaultdict(dict)
    
    for model_type, metrics in aggregated_data.items():
        for metric_name, values in metrics.items():
            count = values['count']
            if count > 0:
                average = values['sum'] / count
                final_metrics[model_type][metric_name] = {
                    "average": average,
                    "count": count
                }

    # --- Writing Output File ---
    try:
        with open(output_filename, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        print(f"Successfully computed metrics and saved to '{output_filename}'.")
    except IOError as e:
        print(f"Error writing to file '{output_filename}': {e}")

if __name__ == "__main__":
    compute_average_metrics()