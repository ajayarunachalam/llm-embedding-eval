from click.testing import CliRunner
import numpy as np
import pytest
import llm
import torch
from llm.cli import cli

def create_test_embedding(values):
    """Create a test embedding file with given values."""
    return np.array(values, dtype=np.float32).tobytes()

class FakeResponse:
    def __init__(self, text):
        self._text = text
    
    def text(self):
        return self._text

class FakeModel:
    needs_key = False
    key = None
    
    def prompt(self, prompt, system):
        return FakeResponse("Embedding A")

def test_eval_semscore(monkeypatch, tmp_path):
    # Mock the embedding model
    class MockEmbeddingModel:
        def get_embedding(self, text):
            return torch.tensor([[1.0, 0.0, 0.0]])
        
        def cos(self, x, y):
            return torch.tensor([1.0])
    
    monkeypatch.setattr("llm_embedding_eval.EmbeddingModel", lambda model_path=None: MockEmbeddingModel())
    
    runner = CliRunner()
    
    # Create test embedding files
    emb1 = create_test_embedding([1.0, 0.0, 0.0])
    emb2 = create_test_embedding([0.0, 1.0, 0.0])
    
    file1 = tmp_path / "emb1.bin"
    file2 = tmp_path / "emb2.bin"
    
    file1.write_bytes(emb1)
    file2.write_bytes(emb2)
    
    result = runner.invoke(
        cli,
        ["eval", "--query", "Test", "--metric", "semscore", 
         "--text1", "hello", "--text2", "hi", 
         str(file1), str(file2)]
    )
    
    assert result.exit_code == 0
    assert "SemScore:" in result.output