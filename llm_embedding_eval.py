#! /usr/bin/env python

import click
import llm
import sys
import struct
import numpy as np
from functools import cmp_to_key
import torch
import torch.nn as nn
import sqlite3
import os
from transformers import AutoTokenizer, AutoModel

class EmbeddingModel:
    DEFAULT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = self.DEFAULT_MODEL
        self.model, self.tokenizer = self.load_model(model_path)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        
    def load_model(self, model_path):
        model = AutoModel.from_pretrained(model_path)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def get_embedding(self, text):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        embedding = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return embedding

DEFAULT_EVAL_PROMPT = """
Given the query:
{query}

Compare these two embedding results:

Embedding A metrics:
{metricsA}

Embedding B metrics:
{metricsB}

SemScore: {semscore:.4f}

Which embedding is more relevant to the query? Answer with "Embedding A" or "Embedding B".
""".strip()

def save_embedding(embedding, filename):
    """Save embedding as binary file."""
    # Convert to numpy if it's a torch tensor
    if torch.is_tensor(embedding):
        embedding = embedding.cpu().numpy()
    
    # Make sure it's the first embedding if there are multiple
    if len(embedding.shape) > 1:
        embedding = embedding[0]
    
    # Save as binary file
    embedding.astype(np.float32).tofile(filename)

def load_embedding(file_path_or_blob):
    """Load embedding from file path, file object, or binary blob."""
    if isinstance(file_path_or_blob, str) and file_path_or_blob.endswith('.db'):
        return load_embedding_from_db(file_path_or_blob)
    elif hasattr(file_path_or_blob, 'read'):
        return np.frombuffer(file_path_or_blob.read(), dtype=np.float32)
    else:
        return np.frombuffer(file_path_or_blob, dtype=np.float32)

def load_embedding_from_db(db_path, table_name="embeddings", column_name="embedding", row_id=1):
    """Load embedding from SQLite database file."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the specified table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            raise ValueError(f"Table '{table_name}' does not exist in the database")
        
        # Check if the column exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        if column_name not in columns:
            raise ValueError(f"Column '{column_name}' does not exist in table '{table_name}'")
        
        # Fetch the embedding
        cursor.execute(f"SELECT {column_name} FROM {table_name} WHERE rowid = ?", (row_id,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"No embedding found with rowid {row_id}")
        
        # Convert BLOB to numpy array
        embedding_blob = result[0]
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        
        return embedding
    
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

def get_db_schema(db_path, table_name="embeddings"):
    """Get the schema of a database table and suggest potential text columns."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the specified table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            raise ValueError(f"Table '{table_name}' does not exist in the database")
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = []
        text_column_candidates = []
        
        for col_info in cursor.fetchall():
            col_name = col_info[1]
            col_type = col_info[2].lower()
            columns.append((col_name, col_type))
            
            # Identify potential text columns based on type or name
            if ('text' in col_type or 
                'char' in col_type or 
                'string' in col_type or
                'text' in col_name.lower() or
                'content' in col_name.lower() or
                'string' in col_name.lower() or
                'query' in col_name.lower() or
                'document' in col_name.lower()):
                text_column_candidates.append(col_name)
        
        return columns, text_column_candidates
    
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

def get_text_from_db(db_path, table_name="embeddings", text_column="text", row_id=1):
    """Extract the original text from the database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the specified table exists
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
        if not cursor.fetchone():
            raise ValueError(f"Table '{table_name}' does not exist in the database")
        
        # Check if the text column exists
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [info[1] for info in cursor.fetchall()]
        if text_column not in columns:
            raise ValueError(f"Column '{text_column}' does not exist in table '{table_name}'")
        
        # Fetch the text
        cursor.execute(f"SELECT {text_column} FROM {table_name} WHERE rowid = ?", (row_id,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"No text found with rowid {row_id}")
        
        return result[0]
    
    except sqlite3.Error as e:
        raise RuntimeError(f"Database error: {str(e)}")
    finally:
        if conn:
            conn.close()

def calculate_metrics(embedding1, embedding2):
    """Calculate various similarity metrics between two embeddings."""
    # Convert to numpy if needed
    if torch.is_tensor(embedding1):
        embedding1 = embedding1.cpu().numpy()
    if torch.is_tensor(embedding2):
        embedding2 = embedding2.cpu().numpy()
        
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 > 0:
        embedding1 = embedding1 / norm1
    if norm2 > 0:
        embedding2 = embedding2 / norm2
        
    # Calculate metrics
    cosine_sim = np.dot(embedding1, embedding2)
    euclidean_dist = np.linalg.norm(embedding1 - embedding2)
    
    return {
        "cosine_similarity": float(cosine_sim),
        "euclidean_distance": float(euclidean_dist),
        "l1_norm": float(np.sum(np.abs(embedding1 - embedding2))),
        "magnitude_ratio": float(norm1 / norm2) if norm2 != 0 else float('inf')
    }

def calculate_semscore(text1, text2, model):
    """Calculate SemScore between two texts using embeddings."""
    emb1 = model.get_embedding(text1)
    emb2 = model.get_embedding(text2)
    
    # Calculate cosine similarity using the model's cosine similarity function
    similarity = model.cos(emb1, emb2).item()
    
    # SemScore is defined as a normalized similarity score
    semscore = (similarity + 1) / 2  # Normalize from [-1,1] to [0,1]
    return semscore

@llm.hookimpl
def register_commands(cli):
    @cli.command()
    @click.option(
        "--query",
        required=True,
        help="The query to use for embedding evaluation."
    )
    @click.option(
        "--metric",
        type=click.Choice(["cosine", "euclidean", "l1", "semscore", "llm"]),
        default="semscore",
        help="Metric to use for comparison"
    )
    @click.option(
        "--text1",
        help="First text for semantic comparison (optional for DB files)"
    )
    @click.option(
        "--text2",
        help="Second text for semantic comparison (optional for DB files)"
    )
    @click.option(
        "--model-path",
        help=f"Path to embedding model (default: {EmbeddingModel.DEFAULT_MODEL})"
    )
    @click.option("-m", "--llm-model", help="LLM model to use for evaluation")
    @click.option("--prompt", help="Custom evaluation prompt template")
    @click.option("--db-table", default="embeddings", help="Table name for DB files")
    @click.option("--db-column", default="embedding", help="Column name for embedding in DB")
    @click.option("--db-text-column", help="Column name for text in DB (if not provided, will try to auto-detect)")
    @click.option("--db-row1", default=1, type=int, help="Row ID for first embedding in DB")
    @click.option("--db-row2", default=1, type=int, help="Row ID for second embedding in DB")
    @click.argument("embedding1", type=click.Path(exists=True))
    @click.argument("embedding2", type=click.Path(exists=True))
    def eval(query, metric, text1, text2, model_path, llm_model, prompt, db_table, db_column, db_text_column, db_row1, db_row2, embedding1, embedding2):
        """
        Evaluate similarity between two embeddings

        This command compares two embeddings using various similarity metrics.
        For semantic scoring (semscore), original texts must be provided or
        available in the database for DB files.
        
        Supports both binary embedding files and SQLite DB files (.db extension).
        
        Example usage:
            # Compare embeddings using cosine similarity
            llm eval --query "How similar are these?" --metric cosine emb1.bin emb2.bin
            
            # Calculate semantic score between two texts
            llm eval --query "Compare semantics" --metric semscore --text1 "hello" --text2 "hi" emb1.bin emb2.bin
            
            # Compare embeddings from DB files
            llm eval --query "DB comparison" --metric cosine --db-table vectors --db-column vector_data --db-row1 3 --db-row2 5 emb1.db emb2.db
            
            # Calculate semantic score using text from DB with auto-detection
            llm eval --query "DB semantics" --metric semscore emb1.db emb2.db
        """
        # Load embeddings based on file type
        try:
            if embedding1.endswith('.db'):
                emb1_data = load_embedding_from_db(embedding1, db_table, db_column, db_row1)
            else:
                with open(embedding1, 'rb') as f:
                    emb1_data = load_embedding(f)
                
            if embedding2.endswith('.db'):
                emb2_data = load_embedding_from_db(embedding2, db_table, db_column, db_row2)
            else:
                with open(embedding2, 'rb') as f:
                    emb2_data = load_embedding(f)
        
        except Exception as e:
            click.echo(f"Error loading embeddings: {str(e)}", err=True)
            return
        
        # Calculate basic metrics
        metrics = calculate_metrics(emb1_data, emb2_data)
        
        # Initialize embedding model if needed
        if metric in ["semscore", "llm"]:
            # If texts not provided, try to get them from DB files
            if not (text1 and text2):
                if embedding1.endswith('.db') and embedding2.endswith('.db'):
                    # Try to auto-detect text column if not specified
                    if not db_text_column:
                        try:
                            # Analyze schema and suggest text columns
                            _, candidates1 = get_db_schema(embedding1, db_table)
                            _, candidates2 = get_db_schema(embedding2, db_table)
                            
                            # Find common candidates between both DBs
                            common_candidates = [col for col in candidates1 if col in candidates2]
                            
                            if common_candidates:
                                db_text_column = common_candidates[0]
                                click.echo(f"Auto-detected text column: '{db_text_column}'")
                            else:
                                # Use first candidate from either DB if no common ones
                                if candidates1:
                                    db_text_column = candidates1[0]
                                    click.echo(f"Using text column from first DB: '{db_text_column}'")
                                elif candidates2:
                                    db_text_column = candidates2[0]
                                    click.echo(f"Using text column from second DB: '{db_text_column}'")
                                else:
                                    # Schema analysis
                                    cols1, _ = get_db_schema(embedding1, db_table)
                                    cols2, _ = get_db_schema(embedding2, db_table)
                                    
                                    click.echo("Could not auto-detect text column. Available columns:")
                                    click.echo(f"DB1: {', '.join(name for name, _ in cols1)}")
                                    click.echo(f"DB2: {', '.join(name for name, _ in cols2)}")
                                    click.echo("Please specify a text column with --db-text-column", err=True)
                                    return
                        except Exception as e:
                            click.echo(f"Error analyzing database schema: {str(e)}", err=True)
                            click.echo("Please specify a text column with --db-text-column", err=True)
                            return
                    
                    try:
                        text1 = get_text_from_db(embedding1, db_table, db_text_column, db_row1)
                        text2 = get_text_from_db(embedding2, db_table, db_text_column, db_row2)
                        click.echo(f"Retrieved texts from column '{db_text_column}':")
                        click.echo(f"Text 1: '{text1[:50]}...'")
                        click.echo(f"Text 2: '{text2[:50]}...'")
                    except Exception as e:
                        click.echo(f"Error retrieving texts from database: {str(e)}", err=True)
                        
                        # Show available columns to help user
                        try:
                            cols1, _ = get_db_schema(embedding1, db_table)
                            cols2, _ = get_db_schema(embedding2, db_table)
                            
                            click.echo("Available columns:")
                            click.echo(f"DB1: {', '.join(name for name, _ in cols1)}")
                            click.echo(f"DB2: {', '.join(name for name, _ in cols2)}")
                        except:
                            pass
                            
                        click.echo("Please provide --text1 and --text2 arguments or use --db-text-column with a valid column name", err=True)
                        return
                else:
                    click.echo("Error: --text1 and --text2 are required for semantic scoring with binary files", err=True)
                    return
            
            embedding_model = EmbeddingModel(model_path)
            semscore = calculate_semscore(text1, text2, embedding_model)
            metrics["semscore"] = semscore
        
        if metric == "llm":
            # Initialize LLM model
            from llm.cli import get_default_model, get_key
            model_obj = llm.get_model(llm_model or get_default_model())
            if model_obj.needs_key:
                model_obj.key = get_key("", model_obj.needs_key, model_obj.key_env_var)
            
            # Format metrics for prompt
            metrics_str_a = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            metrics_str_b = "\n".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            
            # Use custom or default prompt
            prompt_template = prompt or DEFAULT_EVAL_PROMPT
            prompt_text = prompt_template.format(
                query=query,
                metricsA=metrics_str_a,
                metricsB=metrics_str_b,
                semscore=semscore
            )
            
            # Get LLM evaluation
            response = model_obj.prompt(prompt_text, system="You are a helpful assistant.").text().strip()
            click.echo(f"LLM Evaluation: {response}")
            
        elif metric == "semscore":
            click.echo(f"SemScore: {semscore:.4f}")
        else:
            # Return specific metric
            result = metrics[{
                "cosine": "cosine_similarity",
                "euclidean": "euclidean_distance",
                "l1": "l1_norm"
            }[metric]]
            click.echo(f"{metric}: {result:.4f}")