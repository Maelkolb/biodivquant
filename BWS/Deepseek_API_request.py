import pandas as pd
import json

# Load
file_path_tuples = "output_dataset_8_4_final_2024-11-28.csv"
tuples_df = pd.read_csv(file_path_tuples, delimiter='@')


# Prepare JSONL
# for Deepseek R1
jsonl_content = []
for idx, row in tuples_df.iterrows():
    texts = [row[f'Text {i+1}'] for i in range(4) if pd.notna(row[f'Text {i+1}'])]
    prompt = (
        "Task: From the following German texts about animal occurrence, identify:\n "
        "- Best: The text conveying the highest quantity (e.g., presence, frequency, population size)\n"
        "- Worst: The text conveying the lowest quantity\n"
        "Do not explain your answer, just answer with the text numbers in the following json format."
    )
    for i, text in enumerate(texts):
        prompt += f"{i+1}. {text}\n"
    prompt += (
        "\n JSON format for your answer:\n"
        "{\n  \"Best\": [Text Number],\n  \"Worst\": [Text Number]\n}\n\n"
    )

    jsonl_content.append({
        "custom_id": f"tuple-{idx+1}",
        "method": "POST",
        "url": "https://api.fireworks.ai/inference/v1/chat/completions",  
        "body": {
            "model": "accounts/fireworks/models/deepseek-v3",  
            "messages": [
                {"role": "system", "content": "You are an expert annotator specializing in Best-Worst Scaling of German texts based on quantity information about animal occurrences."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 150,  
            "stream": False  
        }
    })
# Save  JSONL file
jsonl_file_path = "Deepseek_V3_fireworks_Input_1000_4_8_test.jsonl"
with open(jsonl_file_path, "w", encoding="utf-8") as file:
    for entry in jsonl_content:
        file.write(json.dumps(entry) + "\n")

print(f"JSONL file saved at: {jsonl_file_path}")

import requests
import json

# API Configuration
api_url = "https://api.fireworks.ai/inference/v1/chat/completions"
api_key = "Your_Key"

# Input File
jsonl_file_path = "Deepseek_V3_fireworks_Input_1000_4_8_test.jsonl"

# Output List
annotations = []

# Read Input File
with open(jsonl_file_path, "r", encoding="utf-8") as file:
    tuples = [json.loads(line) for line in file]

# Process Tuples
for entry in tuples:
    custom_id = entry["custom_id"]
    messages = entry["body"]["messages"]
    model = entry["body"]["model"]

    payload = {
        "model": model,
        "max_tokens": entry["body"].get("max_tokens", 50),
        "top_p": 1,
        "top_k": 40,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "temperature": 0.9,
        "messages": messages
    }

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Make API Call
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))

        if response.status_code == 200:
            content = response.json()["choices"][0]["message"]["content"].strip()
            annotations.append({"custom_id": custom_id, "response": content})
            print(f"Processed: {custom_id}")
        else:
            print(f"Error processing {custom_id}: {response.status_code} {response.reason}")
            annotations.append({"custom_id": custom_id, "error": response.text})

    except Exception as e:
        print(f"Exception processing {custom_id}: {str(e)}")
        annotations.append({"custom_id": custom_id, "error": str(e)})

# Save Output to JSON
output_file_path = "Annotated_Tuples_DeepSeek_4_8_Test.json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(annotations, output_file, ensure_ascii=False, indent=2)

print(f"Annotations for the first 10 tuples saved to: {output_file_path}")

