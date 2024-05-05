import argparse
import sqlite3
from tqdm import tqdm
import json
import random

def randomize_rejected_advice(advice):
    if advice == "BUY":
        options = ["NOOP", "SELL"]
        rejected_advice = random.choice(options)
    elif advice == "SELL":
        options = ["BUY", "NOOP"]
        rejected_advice = random.choice(options)
    else:
        options = ["SELL", "BUY"]
        rejected_advice = random.choice(options)
    return rejected_advice

parser = argparse.ArgumentParser(description='Create CSV database and run queries')
parser.add_argument('--database_path', required=False, default="training_data/kline_data.db", help='Path to database')
parser.add_argument('--from_epoch', required=False, default=1690848660000000,
                    help='From open_time for training data(Default: 2023-08-01 00:11:00)')
parser.add_argument('--to_epoch', required=False, default=1711929600000000,
                    help='To open_time for training data (Default: 2024-04-01 00:00:00)')

# Parse the arguments
args = parser.parse_args()

print(f"Database path: {args.database_path}")
print(f"From open_time: {args.from_epoch}")
print(f"To open_time: {args.to_epoch}")

# Connect to the database
conn = sqlite3.connect(args.database_path)
cursor = conn.cursor()

print(f"Connected to database.")


# Create a table to store the data
cursor.execute('''
CREATE TABLE IF NOT EXISTS training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context text,
    accepted text,
    rejected text
);
''')
conn.commit()
cursor.execute('DELETE FROM training_data;')
conn.commit()

# Define the SQL query to get the rows within the date range
query = f"""
    SELECT * FROM kline_data 
    WHERE open_time BETWEEN ? AND ?
    ORDER BY open_time ASC
"""

cursor.execute(query, (args.from_epoch, args.to_epoch))

# Fetch all the rows
rows = cursor.fetchall()
print(f"Fetched {str(len(rows))} rows")

# Iterate through the rows
for i, row in enumerate(tqdm(rows, desc="Processing rows", unit="rows")):
    # Get the current row's open time
    current_open_time = row[0]

    # Get the previous 10 minutes' data
    prev_10_min_query = """
        SELECT open, volume, count, high, low
        FROM kline_data 
        WHERE open_time <? 
        ORDER BY open_time DESC 
        LIMIT 10
    """
    cursor.execute(prev_10_min_query, (current_open_time,))
    prev_rows = cursor.fetchall()

    # Calculate the changes in high and low
    open_changes = []
    prev_volumes = []
    trade_counts = []

    compare = False
    for prev_row in prev_rows:
        if not compare:
            compare = prev_row
            continue

        open_changes.append('{:.8f}'.format((prev_row[0] - compare[0])))
        prev_volumes.append('{:.8f}'.format(prev_row[1]))
        trade_counts.append('{:.8f}'.format(prev_row[2]))
        compare = prev_row

    # Determine the advice (BUY, SELL, or NOOP)
    if row[2] > prev_row[3] * 1.002:
        advice = "BUY"
    elif row[3] < prev_row[4] * 0.998:
        advice = "SELL"
    else:
        advice = "NOOP"


    context = json.dumps({
        "open_changes": open_changes,
        "prev_volumes": prev_volumes,
        "prev_trade_counts": trade_counts,
    })

    accepted = json.dumps({
        'predicted_open_change': '{:.8f}'.format(row[1] - prev_row[0]),
        'advice': advice
    })

    rejected = json.dumps({
        'predicted_open_change': '{:.8f}'.format(compare[0] - prev_row[0]),
        'advice':  randomize_rejected_advice(advice)
    })

    cursor.execute("INSERT INTO training_data (context, accepted, rejected) VALUES(?,?,?)", (context, accepted, rejected))
    conn.commit()

# Close the database connection
conn.close()
