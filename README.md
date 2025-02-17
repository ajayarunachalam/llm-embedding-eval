# llm-embedding-eval

[![PyPI](https://img.shields.io/pypi/v/llm-embedding-eval.svg)](https://pypi.org/project/llm-embedding-eval/)
[![Tests](https://github.com/ajayarunachalam/llm-embedding-eval/actions/workflows/test.yml/badge.svg)](https://github.com/ajayarunachalam/llm-embedding-eval/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/ajayarunachalam/llm-embedding-eval/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin that will compare two embeddings and determine their similarity/relevance based on various metrics. 
It also supports `SemScore` as proposed in a publication(https://arxiv.org/abs/2401.17072).

## Installation

Install this plugin in the same environment as LLM.
```bash
llm install llm-embedding-eval
```

## Usage

The plugin adds a new command, `llm eval`. 

Usage: llm eval [OPTIONS] EMBEDDING1 EMBEDDING2

![alt text](https://github.com/ajayarunachalam/llm-embedding-eval/blob/main/options.png)

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

