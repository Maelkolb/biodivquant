import pandas as pd
import json
from collections import defaultdict

with open("Annotated_Tuples_GPT4_4_8.json", "r") as f:
    annotations = json.load(f)
print(f"Valid JSON entries loaded: {len(annotations)}")

tuples_df = pd.read_csv("output_dataset_8_4_final_2024-11-28.csv", delimiter="@")
print("CSV Columns:", tuples_df.columns)
tuples_df.columns = tuples_df.columns.str.strip()
tuples_df['custom_id'] = tuples_df.index + 1

text_counts = defaultdict(lambda: {'B': 0, 'W': 0, 'M': 0})
text_columns = ["Text 1", "Text 2", "Text 3", "Text 4"]

for annotation in annotations:
    try:
        custom_id = int(annotation["custom_id"])
    except ValueError:
        custom_id = int(annotation["custom_id"].split('-')[1])
    response_text = annotation.get("response", "")
    if not response_text:
        print(f"Warning: No response found for custom_id {custom_id}")
        continue
    try:
        response = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for custom_id {custom_id}: {e}\nRaw response: {response_text}")
        continue
    best_raw = response["Best"]
    worst_raw = response["Worst"]
    best_index = best_raw - 1 if isinstance(best_raw, int) else best_raw[0] - 1
    worst_index = worst_raw - 1 if isinstance(worst_raw, int) else worst_raw[0] - 1

    row = tuples_df.loc[tuples_df["custom_id"] == custom_id]
    if row.empty:
        print(f"No matching custom_id {custom_id} found in CSV.")
        continue
    row = row.iloc[0]
    for i, col in enumerate(text_columns):
        text = row[col]
        if i == best_index:
            text_counts[text]['B'] += 1
        elif i == worst_index:
            text_counts[text]['W'] += 1
        else:
            text_counts[text]['M'] += 1

scores = {}
for text, counts in text_counts.items():
    total = counts['B'] + counts['W'] + counts['M']
    if total:
        raw_score = (counts['B'] - counts['W']) / total
        adjusted_score = round((raw_score + 1) / 2, 2)
    else:
        adjusted_score = 0
    scores[text] = adjusted_score

unique_scores = set(scores.values())
print(f"Unique adjusted scores: {sorted(unique_scores)}")
print(f"Number of unique adjusted scores: {len(unique_scores)}")

scores_df = pd.DataFrame([{'Text': text, 'Score': score} for text, score in scores.items()])
output_path = "BW_Scores_8_4.csv"
scores_df.to_csv(output_path, index=False, sep="@")
print(f"Scores saved to {output_path}")
