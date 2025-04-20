# src/finrag/data_loaders.py
import os
from typing import List, Dict, Any
import logging # Import standard logging
import json # Import json library

# Ensure logging is configured (by importing logging_conf)
import finrag.logging_conf 

# Import the specific chunking logic for turn data
from .chunk_utils import build_candidate_chunks

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def load_files_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """Loads and processes files from a directory.

    Handles .json files (expected to be lists of 'turn' dicts like dev_turn.json)
    by extracting structured chunks using build_candidate_chunks.
    Skips other file types for now.

    Args:
        directory_path: The path to the directory containing files.

    Returns:
        A list of dictionaries, each representing a text chunk with metadata.
        Example chunk dict: {'chunk_id': '...', 'text': '...', 'source_filename': '...'}
    """
    all_chunks = [] 
    logger.info(f"Scanning directory: {directory_path}")
    if not os.path.isdir(directory_path):
        logger.error(f"Directory not found: {directory_path}")
        return []

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath):
            # Handle JSON files (specifically expecting turn structure)
            if filename.lower().endswith('.json'):
                logger.info(f"Processing JSON file: {filepath}")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f) # Load JSON data

                    # Assuming data is a list of 'turn' dictionaries
                    if isinstance(data, list):
                        for turn in data:
                            if isinstance(turn, dict):
                                # Extract source filename from the turn itself
                                source_pdf_filename = turn.get("filename", "unknown_source")
                                # Use build_candidate_chunks to get structured chunks
                                turn_chunks = build_candidate_chunks(turn)
                                # Add source filename to each chunk
                                for chunk in turn_chunks:
                                    chunk['source_filename'] = source_pdf_filename
                                all_chunks.extend(turn_chunks)
                            else:
                                logger.warning(f"Skipping non-dict item in JSON list: {filepath}")
                    else:
                        logger.warning(f"JSON file does not contain a list as expected: {filepath}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON file {filepath}: {e}")
                except Exception as e:
                    logger.error(f"Failed to process JSON file {filepath}: {e}")
            else:
                logger.debug(f"Skipping non-JSON file: {filepath}")

    logger.info(f"Extracted {len(all_chunks)} chunks from JSON files in {directory_path}")
    return all_chunks
