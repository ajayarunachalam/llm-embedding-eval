# llm-embedding-eval

[LLM](https://llm.datasette.io/) plugin that will compare two embeddings and determine their similarity/relevance based on various metrics. It also supports `SemScore` as proposed in a publication(https://arxiv.org/abs/2401.17072).

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-embedding-eval
```

## Usage

The plugin adds a new command, `llm eval`. 

Usage: llm eval [OPTIONS] EMBEDDING1 EMBEDDING2

Options:
  --query TEXT                    The query to use for embedding evaluation.
                                  [required]
  --metric [cosine|euclidean|l1|semscore|llm]
                                  Metric to use for comparison
  --text1 TEXT                    First text for semantic comparison (optional
                                  for DB files)
  --text2 TEXT                    Second text for semantic comparison
                                  (optional for DB files)
  --model-path TEXT               Path to embedding model (default: sentence-
                                  transformers/all-mpnet-base-v2)
  -m, --llm-model TEXT            LLM model to use for evaluation
  --prompt TEXT                   Custom evaluation prompt template
  --db-table TEXT                 Table name for DB files
  --db-column TEXT                Column name for embedding in DB
  --db-text-column TEXT           Column name for text in DB (if not provided,
                                  will try to auto-detect)
  --db-row1 INTEGER               Row ID for first embedding in DB
  --db-row2 INTEGER               Row ID for second embedding in DB
  --help                          Show this message and exit.

  Evaluate similarity between two embeddings

  This command compares two embeddings using various similarity metrics. For
  semantic scoring (semscore), original texts must be provided or available in
  the database for DB files.

  Supports both binary embedding files and SQLite DB files (.db extension).

  Example usage:         
  ```bash
  # Basic usage with auto-detection of text column with semscore
  llm eval --query "DB semantics" --metric semscore docs.db docs1.db

  # Basic usage with cosine similarity
  llm eval --query "How similar are these?" --metric cosine docs.db docs1.db

  # Basic usage custom prompt with metric llm
  llm eval --metric llm -m llama3.2 --query "Are the contents similar?" --prompt "Query: {query}\n\nMetrics for first text: {metricsA}\nMetrics for second text: {metricsB}\n\nBased on the semscore of {semscore}, are these texts similar? Give a detailed explanation." docs.db docs1.db
  ``` 

**Note**: The prompt template variables are `{query}`, `{metricsA}`, `{metricsB}` and `{semscore}`.

The default prompt used is:

> Given the query:
> {query}
> 
> Compare these two embedding results:
> 
> Embedding A metrics:
> {metricsA}
> 
> Embedding B metrics:
> {metricsB}
> 
> SemScore: {semscore:.4f}
> 
> Which embedding is more relevant to the query? Answer with "Embedding A" or "Embedding B".

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-embedding-eval
python3 -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
pip install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```

