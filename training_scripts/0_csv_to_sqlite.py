import os
import csv
import sqlite3
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Convert CSV files to SQLite database')
parser.add_argument('--csv_files', required=False, type=str, default="training_data", help='CSV files to be converted')
parser.add_argument('--output', required=False, type=str, default="training_data/kline_data.db", help='Output Sqlite database')
parser.add_argument('--recreate', required=False, type=bool, default=False, help='Recreate the database')
# Parse the arguments
args = parser.parse_args()

print(f"Reading CSV files from {args.csv_files}")
print(f"Writing to {args.output}")
if args.recreate:
    print(f"⚠️ Recreating the database!")

    try:
        os.remove(args.output)
    except Exception as e:
        print(e)

# Connect to the SQLite database (it will create a new one if it doesn't exist)
conn = sqlite3.connect(args.output)

print("Opened database successfully")
c = conn.cursor()

# Create a table to store the data
c.execute('''
CREATE TABLE IF NOT EXISTS kline_data (
    open_time TIMESTAMP  PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    close_time INTEGER,
    quote_volume REAL,
    count INTEGER,
    taker_buy_volume REAL,
    taker_buy_quote_volume REAL,
    ignore INTEGER
);
''')
conn.commit()


# Function to insert data from a CSV file into the SQLite database
def insert_data_from_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            try:
                # Convert the open_time and close_time from string to integer
                open_time = int(float(row[0]) * 1000)
                close_time = int(float(row[6]) * 1000)
                row[0] = open_time
                row[6] = close_time
                # Insert the data into the database
                c.execute("INSERT INTO kline_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
            except ValueError:
                print(f"Skipping row due to error: {row}")


csv_files = [f for f in os.listdir(args.csv_files) if f.endswith('.csv')]
for file in tqdm(csv_files, desc="Inserting data"):
    insert_data_from_csv(os.path.join(args.csv_files, file))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("✔️ All data inserted into the SQLite database.")
