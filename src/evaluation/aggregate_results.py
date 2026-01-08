import os
import json

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def run_aggregation():
    all_preds = []
    all_gts = []
    failures = []

    cfg = load_config(BaseConfig, "src/configs/base.json")
    results_path = cfg.paths.local_results
    eval_scenes = cfg.scenes.eval
    
    for scene_name in eval_scenes:
        # Construct the path for each specific eval scene
        scene_path = os.path.join(results_path, scene_name)
        pred_file = os.path.join(scene_path, "predictions.jsonl")
        
        if not os.path.exists(pred_file): continue

        with open(pred_file, "r") as f:
            for line in f:
                data = json.loads(line)
                pred, gt = data["prediction"], data["gt"]
                all_preds.append(pred)
                all_gts.append(gt)
                
                if pred != gt:
                    # Sort IDs to handle view_i/view_j swaps consistently
                    sorted_ids = sorted([data["inst_i"], data["inst_j"]])
                    
                    failures.append({
                        "scene": office_name,
                        "inst_i": sorted_ids[0],
                        "inst_j": sorted_ids[1],
                        "type": "False Positive" if pred == 1 else "False Negative"
                    })

    # 1. Metrics Calculation
    precision, recall, f1, _ = precision_recall_fscore_support(all_gts, all_preds, average='binary')
    accuracy = accuracy_score(all_gts, all_preds)

    print(f"\n{'='*20} GLOBAL STATS {'='*20}")
    print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

    # 2. Per-Scene Difficulty Analysis
    fail_df = pd.DataFrame(failures)
    if not fail_df.empty:
        # Group by scene AND IDs to find specific objects in specific rooms
        top_failures = fail_df.groupby(['scene', 'inst_i', 'inst_j', 'type']).size().reset_index(name='fail_count')
        top_failures = top_failures.sort_values(by='fail_count', ascending=False)

        top_n = 30
        print(f"\n{'='*20} TOP {top_n} SPECIFIC OBJECT FAILURES {'='*20}")
        # This shows exactly which object in which office is causing the most trouble
        print(top_failures.head(top_n).to_string(index=False))

        # 3. Scene Difficulty Ranking
        print(f"\n{'='*20} TOUGHEST SCENES (Total Errors) {'='*20}")
        scene_counts = fail_df['scene'].value_counts().head(5)
        print(scene_counts.to_string())

    else:
        print("No failures found.")

if __name__ == "__main__":
    run_aggregation()