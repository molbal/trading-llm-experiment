import argparse
import sqlite3
import pandas as pd

parser = argparse.ArgumentParser(description='Export training data to Parquet file')
parser.add_argument('--database_path', required=False, default="training_data/kline_data.db", help='Path to database')
parser.add_argument('--output_path', required=False, default="training_data.parquet", help='Path to output Parquet file')

# Parse the arguments
args = parser.parse_args()

# Connect to the database
conn = sqlite3.connect(args.database_path)

# Read the training data from the database into a Pandas DataFrame
df = pd.read_sql_query("SELECT context, accepted, rejected FROM training_data", conn)

# Close the database connection
conn.close()

# Write the DataFrame to a Parquet file
df.to_parquet(args.output_path, engine='pyarrow', compression='snappy')
print(f"Training data exported to {args.output_path}")
