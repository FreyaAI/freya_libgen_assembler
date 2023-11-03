## Some Imports

import sqlite3
import argparse
from transformers import AutoModel
import pickle
from tqdm import tqdm
from numpy.linalg import norm
import sqlparse
import time


def parseArgs():
    parser = argparse.ArgumentParser(description='Embedder')
    parser.add_argument('dataset', type=str, default=False, help='Dataset name')
    parser.add_argument('--sql_file_path', type=str, required=True, help='Path to sql file')
    args = parser.parse_args()

    if not args.dataset:
        args.dataset = args.sql_file_path.split('/')[-1].split('.')[0]

    return args



def parse_sql_file(filename):
    buffer = []

    with open(filename, 'r', encoding="latin-1") as file:
        for line in file:
            # If we encounter a line with INSERT and the buffer is not empty,
            # finalize the previous command
    
            if "INSERT" in line and buffer:
                yield "".join(buffer).strip()
                buffer = []

            buffer.append(line)
    # If there's any remaining content in the buffer, yield it as a command
    if buffer:
        yield "".join(buffer).strip()

def openSQLLiteConnection(dataset_name):

    # Connect to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(f'{dataset_name}.db')
    cursor = conn.cursor()


    return conn, cursor

def closeSQLLiteConnection(_con):
    if _con:
        _con.close()



# Modify the executeSQLScript function:
def executeSQLCommand(_con, _cursor, command):

        try:
            print(command)
            _cursor.execute(command)
            _con.commit()
        except sqlite3.Error as e:
            print(f"An error occurred while executing command: {command[:50]}... : {e.args[0]}")
            raise e
            
            # Decide whether to break or continue based on the nature of the error

    

def main():
    args = parseArgs()

    DATASET_NAME = args.dataset
    SQL_FILE_PATH = args.sql_file_path

    # Open the connection to the database
    conn, cursor = openSQLLiteConnection(DATASET_NAME)

    for command in parse_sql_file(SQL_FILE_PATH):#tqdm(parse_sql_file(SQL_FILE_PATH)):

        subcommands = sqlparse.split(command)

        for subcommand in subcommands:
            print(subcommand)
            
            executeSQLCommand(conn, cursor, subcommand)
            print('-------------------')

    # Close the connection to the database
    closeSQLLiteConnection(conn)


    print('Database created successfully!')

    



if __name__ == "__main__":
    main()
