import os
import json

def collect_metrics(root_folder: str, output_file: str):
    aggregated_metrics = []
    
    for subdir in os.listdir(root_folder):
        model_dir = os.path.join(root_folder, subdir)
        metrics_path = os.path.join(model_dir, "metrics.json")
        
        if os.path.isdir(model_dir) and os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as fp:
                metrics = json.load(fp)
                aggregated_metrics.append({
                    "name": subdir,
                    "metrics": metrics
                })
    
    with open(output_file, "w", encoding="utf-8") as fp:
        json.dump(aggregated_metrics, fp, indent=2, ensure_ascii=False)
    print(f"Zbiorczy plik zapisany w {output_file}")

collect_metrics("tests", "aggregated_metrics.json")