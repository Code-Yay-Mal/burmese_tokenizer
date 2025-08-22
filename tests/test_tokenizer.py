"""tests for burmese tokenizer"""

import pytest
from pathlib import Path

from burmese_tokenizer import BurmeseTokenizer


class TestBurmeseTokenizer:
    """test cases for burmese tokenizer"""
    
    def test_tokenizer_initialization(self):
        """test tokenizer init with default model"""
        tokenizer = BurmeseTokenizer()
        assert tokenizer is not None
        assert hasattr(tokenizer, '_processor')
    
    def test_tokenizer_with_custom_model(self, tmp_path):
        """test tokenizer init with custom model path"""
        # test that it raises filenotfounderror
        with pytest.raises(FileNotFoundError):
            BurmeseTokenizer(str(tmp_path / "nonexistent.model"))
    
    def test_encode_burmese_text(self):
        """test encoding burmese text"""
        tokenizer = BurmeseTokenizer()
        text = "မင်္ဂလာပါ။"
        result = tokenizer.encode(text)
        
        assert isinstance(result, dict)
        assert "pieces" in result
        assert "ids" in result
        assert "text" in result
        assert result["text"] == text
        assert isinstance(result["pieces"], list)
        assert isinstance(result["ids"], list)
        assert len(result["pieces"]) == len(result["ids"])
    
    def test_decode_tokens(self):
        """test decoding tokens back to text"""
        tokenizer = BurmeseTokenizer()
        text = "မင်္ဂလာပါ။"
        encoded = tokenizer.encode(text)
        
        decoded = tokenizer.decode(encoded["pieces"])
        assert isinstance(decoded, str)
    
    def test_decode_ids(self):
        """test decoding token ids back to text"""
        tokenizer = BurmeseTokenizer()
        text = "မင်္ဂလာပါ။"
        encoded = tokenizer.encode(text)
        
        decoded = tokenizer.decode_ids(encoded["ids"])
        assert isinstance(decoded, str)
    
    def test_get_vocab_size(self):
        """test getting vocab size"""
        tokenizer = BurmeseTokenizer()
        vocab_size = tokenizer.get_vocab_size()
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
    
    def test_get_vocab(self):
        """test getting vocab mapping"""
        tokenizer = BurmeseTokenizer()
        vocab = tokenizer.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == tokenizer.get_vocab_size()


class TestIntegration:
    """integration tests"""
    
    def test_complete_workflow(self):
        """test complete encode-decode workflow"""
        tokenizer = BurmeseTokenizer()
        original_text = "မင်္ဂလာပါ။ နေကောင်းပါသလား။"
        
        # encode
        encoded = tokenizer.encode(original_text)
        
        # decode using pieces
        decoded_pieces = tokenizer.decode(encoded["pieces"])
        
        # decode using ids
        decoded_ids = tokenizer.decode_ids(encoded["ids"])
        
        # both should produce similar results
        assert isinstance(decoded_pieces, str)
        assert isinstance(decoded_ids, str)
        assert len(decoded_pieces) > 0
        assert len(decoded_ids) > 0
    
    def test_load_tokenizer_function(self):
        """test convenience load_tokenizer function"""
        from burmese_tokenizer import load_tokenizer
        
        tokenizer = load_tokenizer()
        assert isinstance(tokenizer, BurmeseTokenizer)
        
        # test it works
        result = tokenizer.encode("မင်္ဂလာပါ။")
        assert "pieces" in result
