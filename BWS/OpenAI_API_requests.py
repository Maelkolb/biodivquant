from openai import OpenAI
import json

client = OpenAI(api_key="Your_Key")

with open("Tuples_LLM_input.jsonl", "r", encoding="utf-8") as file:
    entries = [json.loads(line) for line in file]

annotations = []

for entry in entries:
    custom_id = entry["custom_id"]
    prompt = entry["body"]["messages"][1]["content"]
    try:
        response = client.chat.completions.create(
            model=entry["body"]["model"],
            messages=[
                {"role": "system", "content": entry["body"]["messages"][0]["content"]},
                {"role": "user", "content": prompt}
            ],
            max_tokens=entry["body"].get("max_tokens", 50)
        )
        content = response.choices[0].message.content.strip()
        annotations.append({"custom_id": custom_id, "response": content})
        print(f"Processed: {custom_id}")
    except Exception as e:
        print(f"Error processing {custom_id}: {e}")
        annotations.append({"custom_id": custom_id, "error": str(e)})

with open("Annotated_Tuples_GPT4_4_8_new_prompt.json", "w", encoding="utf-8") as output_file:
    json.dump(annotations, output_file, ensure_ascii=False, indent=2)

print("Annotations saved to: Annotated_Tuples_GPT4_4_8_new_prompt.json")
