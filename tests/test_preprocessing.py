import pytest
from src.preprocessing.text_processor import TextPreprocessor, FeatureExtractor, preprocess_for_model


class TestTextPreprocessor:
    def setup_method(self):
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text_removes_urls(self):
        text = "Check this out https://example.com amazing!"
        result = self.preprocessor.clean_text(text)
        assert "https" not in result
        assert "example.com" not in result
    
    def test_clean_text_removes_mentions(self):
        text = "@user This is a test post"
        result = self.preprocessor.clean_text(text)
        assert "@user" not in result
    
    def test_clean_text_removes_hashtag_symbol(self):
        text = "This is #awesome"
        result = self.preprocessor.clean_text(text)
        assert "#" not in result
        assert "awesome" in result
    
    def test_clean_text_lowercase(self):
        text = "THIS IS UPPERCASE"
        result = self.preprocessor.clean_text(text)
        assert result == result.lower()
    
    def test_preprocess_full_pipeline(self):
        text = "RT @user: Check out this link https://example.com #AI #MachineLearning"
        result = self.preprocessor.preprocess(text)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "@user" not in result
        assert "https" not in result


class TestFeatureExtractor:
    def setup_method(self):
        self.extractor = FeatureExtractor()
    
    def test_extract_features_basic(self):
        text = "This is a test post!"
        features = self.extractor.extract_features(text)
        
        assert 'original_length' in features
        assert 'num_words' in features
        assert 'preprocessed_text' in features
        assert features['original_length'] == len(text)
    
    def test_extract_features_hashtags(self):
        text = "This is #awesome and #great"
        features = self.extractor.extract_features(text)
        assert features['num_hashtags'] == 2
    
    def test_extract_features_mentions(self):
        text = "@user1 and @user2 are mentioned"
        features = self.extractor.extract_features(text)
        assert features['num_mentions'] == 2
    
    def test_extract_features_urls(self):
        text = "Check https://example.com and http://test.com"
        features = self.extractor.extract_features(text)
        assert features['num_urls'] == 2
    
    def test_extract_features_exclamation(self):
        text = "This is amazing!!! Really!!!"
        features = self.extractor.extract_features(text)
        assert features['num_exclamation'] == 6


def test_preprocess_for_model():
    text = "This is a test post with @mentions and #hashtags https://url.com"
    result = preprocess_for_model(text)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "@mentions" not in result
    assert "https" not in result
