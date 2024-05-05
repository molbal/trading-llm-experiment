import argparse
import sqlite3
from tqdm import tqdm
import json
import ollama

def call_llm(row):
    global response
    response = ollama.chat(model=args.model_name, messages=[
        {
            'role': 'user',
            'content': f"""Analyse the following exchange data, predict the next minute's change and advise. Respond in a JSON object with 2 properties: predicted_open_change (8 decimals precision), and advice (BUY, NOOP or SELL) ### Input:{row[1]}""",
        },
    ])
    return response['message']['content']

parser = argparse.ArgumentParser(description='Create CSV database and run queries')
parser.add_argument('--database_path', required=False, default="training_data/kline_data.db", help='Path to database')
parser.add_argument('--model_name', required=False, default="atm", help='Name of Ollama model')

# Parse the arguments
args = parser.parse_args()

print(f"Database path: {args.database_path}")
print(f"Model name: {args.model_name}")

# Connect to the database
conn = sqlite3.connect(args.database_path)
cursor = conn.cursor()

print(f"Connected to database.")



# Define the SQL query to get the rows within the date range
query = f"""
    SELECT * FROM verification_data 
    WHERE result=''
"""

cursor.execute(query)

# Fetch all the rows
rows = cursor.fetchall()
print(f"Fetched {str(len(rows))} rows")





# Iterate through the rows
import re
import json

MAX_RETRIES = 5

for i, row in enumerate(tqdm(rows, desc="Running verification dataset", unit="row")):
    retries = 0
    while retries < MAX_RETRIES:
        raw = call_llm(row)
        match = re.search(r'{[^{}]+}', raw)  # find a JSON-like string
        if match:
            json_string = match.group(0)  # extract the matched JSON string
            try:
                prediction = json.loads(json_string)  # try to parse the JSON string
                cursor.execute("UPDATE verification_data SET result=? WHERE id=?",
                               (json_string, row[0]))
                conn.commit()
                break
            except json.JSONDecodeError:
                # if the extracted string is not a valid JSON, retry
                retries += 1
                continue
        else:
            # if no JSON-like string is found, retry
            retries += 1
            continue
    else:
        # if all retries fail, raise an exception
        raise ValueError(f"Failed to extract JSON string from LLM output after {MAX_RETRIES} retries")

# Close the database connection
conn.close()
