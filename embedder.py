import sqlite3
from abc import ABC, abstractmethod
from typing import List, Tuple, Generator, Union, Any, Optional
import re
import inspect
import time
from transformers import AutoModel
from numpy.linalg import norm
import torch
import numpy as np
import json
from datetime import datetime
import os
import gc
import sys
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


###
####  SQLite3 and Database related classes
###

# Database Connection class
class DatabaseConnection:
    def __init__(self, db_path: str):
        print("[DatabaseConnection] Initializing...")
        self.db_path = db_path

    def __enter__(self) -> sqlite3.Cursor:
        print("[DatabaseConnection] Entering and connecting to the database...")
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        return self.cursor

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        print("[DatabaseConnection] Exiting and closing the connection...")
        self.conn.close()

# Book class to represent the 'info' table
class Book:
    def __init__(self, ID: int, MD5: str, Title: str, Author: str, Language: str, 
                 Type: str, Year: str, Pages: str, Extension: str, Filesize: int):
        self.ID = ID
        self.MD5 = MD5
        self.Title = Title
        self.Author = Author
        self.Language = Language
        self.Type = Type
        self.Year = Year
        self.Pages = Pages
        self.Extension = Extension
        self.Filesize = Filesize

    @classmethod
    def from_tuple(cls, data_tuple: Tuple) -> 'Book':
        return cls(*data_tuple)

    def toString(self):
        return self.Title

# Description class to represent the 'description' table
class Description:
    def __init__(self, MD5: str, Description: str):
        self.MD5 = MD5
        self.Description = Description

    @classmethod
    def from_tuple(cls, data_tuple: Tuple) -> 'Description':
        return cls(*data_tuple)
    
    def toString(self):
        return self.Description


class DataFetcherFactory(ABC):
    @abstractmethod
    def fetch_data(self, cursor: sqlite3.Cursor, param1: Any, param2: Any) -> Any:
        pass

# Fetcher for combined data from 'info' and 'description' tables
class CombinedDataFetcher(DataFetcherFactory):
    def fetch_data(self, cursor: sqlite3.Cursor, n: int, offset: int) -> List[Tuple[Any, ...]]:
        print(f"[CombinedDataFetcher] Fetching {n} rows with offset {offset}...")
        query = """
        SELECT i.*, d.*
        FROM info i
        JOIN description d ON i.MD5 = d.MD5
        LIMIT ? OFFSET ?
        """
        cursor.execute(query, (n, offset))
        return cursor.fetchall()


# Data Processor to handle fetching and processing data batches
class DataProcessor:
    def __init__(self, combined_fetcher: CombinedDataFetcher, database_connection: DatabaseConnection):
        print("[DataProcessor] Initializing with fetcher...")
        self.combined_fetcher = combined_fetcher
        self.database_connection = database_connection

    def get_data_batches(self, n: int) -> Generator[List[Tuple[Book, Description]], None, None]:
        offset = 0
        with self.database_connection as cursor:
            while True:
                combined_rows = self.combined_fetcher.fetch_data(cursor, n, offset)

                if not combined_rows:
                    print("[DataProcessor] No more combined rows to fetch. Breaking out...")
                    break

                batch_data = []
                for row in combined_rows:
                    
                    num_params_book = len(inspect.signature(Book.__init__).parameters) - 1 # -1 for self 
                    book_data = row[:num_params_book]
                    description_data = row[num_params_book:]

                    book = Book.from_tuple(book_data)
                    book_description = Description.from_tuple(description_data)

                    batch_data.append((book, book_description))

                print(f"[DataProcessor] Yielding batch of {len(batch_data)} data entries...")
                yield batch_data
                offset += n


##
###  Utility classes
##



from nltk.tokenize import sent_tokenize

class TextSplitter:
    def __init__(self, batch_sentence_count, max_sentence_character_count):
        self.batch_sentence_count = batch_sentence_count
        self.max_sentence_character_count = max_sentence_character_count

    def split_to_batches(self, text):
        tokenized_description = sent_tokenize(text)
        
        # Check for max_sentence_character_count and split if necessary
        processed_sentences = []
        for sentence in tokenized_description:
            while len(sentence) > self.max_sentence_character_count:
                # Split the sentence
                processed_sentences.append(sentence[:self.max_sentence_character_count])
                sentence = sentence[self.max_sentence_character_count:]
            processed_sentences.append(sentence)
        
        # Group sentences to batches
        batches = []
        batch = []
        for sentence in processed_sentences:
            if len(batch) < self.batch_sentence_count:
                batch.append(sentence)
            else:
                batches.append(' '.join(batch))
                batch = [sentence]
        if batch:  # Append the last batch if it's not empty
            batches.append(' '.join(batch))
        
        return batches


##
### Embedder related classes
##


class Embedder:
    def __init__(self, model_name: str, max_model_batch_size: int = 64):
        print("[Embedder] Initializing model...")

        self.max_model_batch_size = max_model_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval().to(self.device)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        print(f"[Embedder] Generating embeddings for {len(texts)} texts...")

        batch_size = min(self.max_model_batch_size, len(texts))

        print(f"[Embedder] Using Batch size: {batch_size}")

        with torch.no_grad():  # Deactivating gradient calculation for inference
            embeddings = self.model.encode(texts, batch_size= batch_size)
        return embeddings


    @staticmethod
    def cos_similarity(a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        a = np.array(a)
        b = np.array(b)
        return (a @ b.T) / (norm(a) * norm(b))

##
### Main
##

def get_size(obj, seen=None) -> int:
    """Recursively finds the size of objects in bytes."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def assamble_filtered_batch_element(title: str, description: Optional[str], desription_batch_size: int = 3) -> List[str]:
    if description is None or description.strip() == "":
        return [title]
    
    ## The 
    splitter = TextSplitter(batch_sentence_count=desription_batch_size, max_sentence_character_count= 300)

    try:
        description_batches = splitter.split_to_batches(description)
    except Exception as e:
        print(f"[Main] Error while splitting description: >{description}<")
        raise e


    return [title] + description_batches

def group_embeddings(embeddings: List[List[float]], batch_lengths: List[int]) -> List[List[List[float]]]:
    """Groups the embeddings into batches."""
    grouped_embeddings = []
    start_idx = 0
    for length in batch_lengths:
        grouped_embeddings.append(embeddings[start_idx: start_idx + length])
        start_idx += length
    return grouped_embeddings

def create_timestamped_directory(base_path: str) -> str:
    """Creates a directory with a timestamp as its name and returns its path."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir_path = os.path.join(base_path, timestamp)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def create_data_dictionary(batch: List[Tuple[Book, Description]], grouped_embeddings: List[List[List[float]]]) -> dict:
    """Constructs a dictionary for a given batch and its corresponding embeddings."""

    print("Title embedding size: ", len(grouped_embeddings[0][0]))
    # Description embeddings are a list of lists
    print("Description embedding sizes: ", [len(embedding) for embedding in grouped_embeddings[0][1:]])

    

    data_dict = {}
    for (book, _), embeddings in zip(batch, grouped_embeddings):
        md5 = book.MD5
        data_dict[md5] = {
            "title": embeddings[0],
            "description": embeddings[1:]
        }
    return data_dict

def save_accumulated_data(accumulated_data: dict, output_directory: str, batch_counter: int) -> None:
    """Saves the accumulated data to a single NumPy .npz file with MD5 keys."""

    start_time = time.time()

    # Prepare the dictionary for saving
    np_save_dict = {}
    for md5, data in accumulated_data.items():
        np_save_dict[f"{md5}_title"] = data["title"]
        for i, desc in enumerate(data["description"], start=1):
            np_save_dict[f"{md5}_description{i}"] = desc

    # Save the dictionary as a NumPy .npz file
    np.savez_compressed(os.path.join(output_directory, f"embeddings_batch_{batch_counter}.npz"), **np_save_dict)

    print(f"[Main] Saved NumPy batch {batch_counter} in {time.time() - start_time} seconds.")

def free_memory() -> None:
    """Frees up memory."""
    gc.collect()
    torch.cuda.empty_cache()



def process_batch(batch, pending_elements, accumulated_data, batch_counter, output_directory, embedder, DESCRIPTION_CHUNK_SIZE, MAX_MODEL_BATCH_SIZE, SAVE_EVERY_N_BATCHES, executor):

    free_memory()
    all_texts = []
    all_embeddings = []
    grouped_embeddings = []
    batch_data_dict = {}

    print(f"[Main] Processing batch {batch_counter}...")
    batch_size_in_bytes = get_size(batch)
    print(f"[DEBUG] Batch size in memory: {batch_size_in_bytes / (1024 * 1024):.2f} MB")

    filtered_batch = []
    filtered_batch_portion_lengths = []
    for book, description in batch:
        title = book.Title
        description = description.Description
        filtered_batch_element = assamble_filtered_batch_element(title, description, DESCRIPTION_CHUNK_SIZE)
        filtered_batch.append(filtered_batch_element)
        filtered_batch_portion_lengths.append(len(filtered_batch_element))
        del title, description, filtered_batch_element

    filtered_batch = pending_elements + filtered_batch
    filtered_batch_portion_lengths = [len(element) for element in pending_elements] + filtered_batch_portion_lengths

    pending_elements = []

    current_batch = []
    current_batch_portion_lengths = []

    cutoff_point  = max(math.floor(len([portion for sublist in filtered_batch for portion in sublist]) / MAX_MODEL_BATCH_SIZE), 1)  * MAX_MODEL_BATCH_SIZE

    counter = 0
    for filtered_batch_element, filtered_batch_element_portion_length in zip(filtered_batch, filtered_batch_portion_lengths):
        if counter + len(filtered_batch_element) <= cutoff_point:
            current_batch.append(filtered_batch_element)
            current_batch_portion_lengths.append(filtered_batch_element_portion_length)
            counter += len(filtered_batch_element)
            continue
        pending_elements.append(filtered_batch_element)
        counter += len(filtered_batch_element)
        
    del filtered_batch
    del filtered_batch_portion_lengths

    all_texts = [portion for sublist in current_batch for portion in sublist]

    max_length = max(len(text) for text in all_texts)
    print(f"[DEBUG] Max length of all_texts: {max_length}")

    all_embeddings = embedder.get_embeddings(all_texts)
    print(f"[DEBUG] Embeddings size in memory: {get_size(all_embeddings) / (1024 * 1024):.2f} MB")
    
    grouped_embeddings = group_embeddings(all_embeddings, current_batch_portion_lengths)

    batch_data_dict = create_data_dictionary(batch, grouped_embeddings)
    accumulated_data.update(batch_data_dict)

    batch_counter += 1

    total_allocated = torch.cuda.memory_allocated()  # in bytes
    print(f"Total allocated memory: {total_allocated / (1024**2):.2f} MB")
        
    if batch_counter % SAVE_EVERY_N_BATCHES == 0:
        # Run the save process in a separate thread
        if executor:
            executor.submit(save_accumulated_data, accumulated_data, output_directory, batch_counter)
            accumulated_data = {}

    return pending_elements, accumulated_data, batch_counter

def main() -> None:
    print("[Main] Starting...")

    ### Initialization ###
    DATABASE_PATH = "C:\Workzone\Datasets\LibgenDatabases\merged_fiction_nonfiction_sql3.db"
    BATCH_SIZE = 128
    DESCRIPTION_CHUNK_SIZE = 4
    OUTPUT_DIR = "./outputs/embedder"
    MAX_MODEL_BATCH_SIZE = 128  # Max batch size for the model
    SAVE_EVERY_N_BATCHES = 50  # Save results every N batches
    DEBUG_LIMIT = 30000  # Limit the number of elements processed

    output_directory = create_timestamped_directory(OUTPUT_DIR)
    print(f"[Main] Saving results in directory: {output_directory}")

    embedder = Embedder('jinaai/jina-embeddings-v2-base-en', max_model_batch_size=MAX_MODEL_BATCH_SIZE)
    combined_fetcher = CombinedDataFetcher()
    database_connection = DatabaseConnection(DATABASE_PATH)
    processor = DataProcessor(combined_fetcher, database_connection)

    ### Initialization Over ###

    batch_counter = 0
    total_elements_processed = 0  # Track total elements processed
    accumulated_data = {}  # Store accumulated results here
    pending_elements = []  # Store elements that couldn't be processed due to memory constraints

    with ThreadPoolExecutor(max_workers=2) as executor:
        for batch in processor.get_data_batches(BATCH_SIZE):
            pending_elements, accumulated_data, batch_counter = process_batch(
                batch, pending_elements, accumulated_data, batch_counter, 
                output_directory, embedder, DESCRIPTION_CHUNK_SIZE, 
                MAX_MODEL_BATCH_SIZE, SAVE_EVERY_N_BATCHES, executor
            )
    
            total_elements_processed += len(batch)
            if DEBUG_LIMIT and total_elements_processed >= DEBUG_LIMIT:
                print(f"[Main] Debug limit of {DEBUG_LIMIT} elements reached. Exiting early...")
                break

    if pending_elements:
        dummy_batch = []
        pending_elements, accumulated_data, batch_counter = process_batch(dummy_batch, pending_elements, accumulated_data, batch_counter, output_directory, embedder, DESCRIPTION_CHUNK_SIZE, MAX_MODEL_BATCH_SIZE, SAVE_EVERY_N_BATCHES, False)

    if accumulated_data:
        save_accumulated_data(accumulated_data, output_directory, batch_counter)
    
    print("[Main] Finished!")

if __name__ == '__main__':
    start_time = time.time()
    main()  
    print(f"Time taken: {time.time() - start_time} seconds.")