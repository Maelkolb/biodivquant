# Prepare JSONL for ChatGPT-4-API
import pandas as pd
import json

# Load
file_path_tuples = "output_dataset_8_4_final_2024-11-28.csv"
tuples_df = pd.read_csv(file_path_tuples, delimiter='@')

jsonl_content = []

for idx, row in tuples_df.iterrows():
    # Collect texts from columns "Text 1" to "Text 4"
    texts = [
        row[f'Text {i+1}'] 
        for i in range(4) 
        if pd.notna(row[f'Text {i+1}'])
    ]
    
    # prompt
    prompt = (
        "Task: From the following German texts about animal occurrence, identify:\n"
        "- Best: The text conveying the highest quantity (e.g., presence, frequency, population size)\n"
        "- Worst: The text conveying the lowest quantity\n\n"
        + "\n".join(texts)
        + "\n\nJSON format for your answer:\n"
        "{\n  \"Best\": [Text Number],\n  \"Worst\": [Text Number]\n}\n"
    )
    
    jsonl_content.append({
        "custom_id": f"tuple-{idx+1}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert annotator specializing in Best-Worst Scaling of German texts based on quantity information about animal occurrences."
                },
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150
        }
    })

    # Save  JSONL file
jsonl_file_path = "Tuples_LLM_input.jsonl"
with open(jsonl_file_path, "w", encoding="utf-8") as file:
    for entry in jsonl_content:
        file.write(json.dumps(entry) + "\n")

print(f"JSONL file saved at: {jsonl_file_path}")

