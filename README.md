
# Data Embedding Script

This script processes data from a SQLite database, creates embeddings for book titles and descriptions, and saves the output in an organized format. It uses a batch processing approach, which is efficient for handling large datasets.

## Requirements

- Python 3.6+
- SQLite3
- transformers (Hugging Face)
- NLTK
- NumPy
- Torch

## Installation

Before running the script, make sure you have the required dependencies installed. You can install them using the requirements file.

```bash
pip install -r requirements.txt
```
or manually

```bash
pip install sqlite3 transformers nltk numpy torch
```

## Usage

The script can be executed from the command line with several options to customize its behavior. Here are the options you can specify:

- `--database_path`: Path to the database file. REQUIRED
- `--batch_size`: Number of records or books to process in each batch. Default is `128`.
- `--description_chunk_size`: The number of sentences per chunk when splitting the description. Default is `4`.
- `--output_dir`: Directory where the processed outputs will be saved. Default is `"./outputs/embedder"`.
- `--max_model_batch_size`: Maximum batch size for the model to process at once. Default is `128`. This is the real number that specifies how many elements are going into the model at each time. This ideal should be same or slightly more than the "batch_size" argument. Adjust this for VRAM usage but using a big value while using relatively small batch_size won't make a difference so adjust both proportionally.
- `--save_every_n_batches`: Frequency of saving the processed batches to disk. Default is `50`.
- `--debug_limit`: Limits the number of elements to be processed, useful for debugging. Default is `None`, which means no limit.

To run the script with default options, use:

```bash
python embedder.py --database_path ".\merged_fiction_nonfiction_sql3.db"
```

To specify options, append them as arguments:

```bash
python embedder.py --database_path ".\merged_fiction_nonfiction_sql3.db" --batch_size 128 --description_chunk_size 4 --output_dir "./outputs/embedder" --max_model_batch_size 128 --save_every_n_batches 50
```

## Output

The script saves the embeddings in a compressed `.npz` format, with separate entries for title and description embeddings. The files are saved in a timestamped subdirectory under the specified `output_dir`.

## Contributing

Feel free to fork the repository and submit a pull request if you'd like to contribute to this project.

## License

This script is provided under the MIT License.

## Contact

For any questions or issues, please open an issue on the project's GitHub page.
