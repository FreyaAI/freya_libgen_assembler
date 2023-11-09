import sqlparse
import pymysql
import pathlib
from tqdm import tqdm
import time
import os

# Set this flag to True if you want to remove the SQL file after execution
SHOULD_REMOVE_AFTER_EXECUTION = False

# Function to connect to the MySQL database
def connect_to_db(host, user, password, db):
    try:
        connection = pymysql.connect(host=host,
                                     user=user,
                                     password=password,
                                     db=db,
                                     charset='utf8mb4',
                                     cursorclass=pymysql.cursors.DictCursor)
        return connection
    except Exception as e:
        print(f"Error connecting to MySQL database: {e}")
        return None

# Function to read SQL commands from a file and return a list of individual commands
def read_sql_file(file_path):
    with open(file_path, 'r', encoding="latin-1") as file:
        raw = file.read()
        print("Parsing SQL commands ... ")

        start_time = time.time()
        statements = sqlparse.split(raw)
        print(f"Parsing completed in {time.time() - start_time} seconds")

        return statements

# Function to execute SQL commands
def execute_commands(connection, commands, file_path):
    print("Executing commands ... ")

    with connection.cursor() as cursor:
        for command in tqdm(commands):
            # Skip certain commands
            if 'ALTER TABLE' in command and 'ENABLE KEYS'in command:
                continue

            try:
                cursor.execute(command)
                # Commit the transaction if necessary (for INSERT, UPDATE, DELETE)
                if 'select' not in command.lower():
                    connection.commit()
            except Exception as e:
                # Print the error and continue with the next statement
                print(f"Error executing command: {command}\nError: {e}")

    # Check the flag and remove the file if set to True
    if SHOULD_REMOVE_AFTER_EXECUTION:
        try:
            os.remove(file_path)
            print(f"Removed file {file_path} after execution.")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")

# Main function to run the script
def read_and_execute(sql_file_path, host, user, password, db):
    # Connect to the database
    connection = connect_to_db(host, user, password, db)
    if connection:
        try:
            # Read SQL file and get individual commands
            commands = read_sql_file(sql_file_path)
            # Execute the commands
            execute_commands(connection, commands, sql_file_path)
        finally:
            # Close the connection
            connection.close()


def read_filepaths_from_dir_with_extension(dir_path, extension):    
    return sorted(pathlib.Path(dir_path).glob(f'*.{extension}'), key=lambda p: p.name)


if __name__ == "__main__":
    # Replace with your actual database connection details and SQL file path
    HOST = 'localhost'
    USER = 'root'
    PASSWORD = 'Isa66076607.'
    DB = 'scimag'

    SQL_CHUNKS_FOLDER = 'C:/Users/isatu/Desktop/SQLDumpSplitter3_0.9.3/files/backup_libgen_scimag_split'

    for sql_file_path in read_filepaths_from_dir_with_extension(SQL_CHUNKS_FOLDER, 'sql'):
        print(f'Executing {sql_file_path}')

        read_and_execute(sql_file_path, HOST, USER, PASSWORD, DB)