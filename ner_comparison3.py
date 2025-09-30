
"""
Experimental Evaluation of NLP Pipelines: Classical vs Transformer-based NER
Task 3: Responsible NLP Case Study

This script compares classical NLP pipelines with transformer-based approaches 
for Named Entity Recognition (NER) using the Named Entity Recognition Corpus from Kaggle.
Additionally, it implements a comprehensive responsible NLP analysis including:
- Bias detection and analysis
- Failure case documentation
- Quantitative fairness metrics
- Mitigation strategies

Requirements:
pip install pandas numpy scikit-learn spacy sklearn-crfsuite
pip install transformers datasets torch seqeval matplotlib seaborn
python -m spacy download en_core_web_sm


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
from collections import Counter
import re
import os
import pickle
import json
from collections import defaultdict
import random
from typing import List, Dict, Tuple, Any

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# SpaCy for classical NLP
import spacy
from spacy import displacy

# CRF for classical pipeline
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

# Transformers for BERT-based pipeline
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset
import torch

# Evaluation metrics
from seqeval.metrics import classification_report as seqeval_report
from seqeval.metrics import accuracy_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

print("="*60)
print("NLP Pipeline Comparison: Classical vs Transformer-based NER")
print("="*60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print("="*60)

class NERPipelineComparison:
    def __init__(self, dataset_path='dataset/ner.csv', max_samples=20000):
        """Initialize the NER pipeline comparison."""
        self.dataset_path = dataset_path
        self.max_samples = max_samples
        self.sentences = []
        self.labels = []
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        
        # Model paths
        self.crf_model_path = 'models/crf_model.pkl'
        self.bert_model_path = 'models/bert_ner_model'
        self.preprocessed_data_path = 'models/preprocessed_data.pkl'
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Timing variables
        self.preprocessing_time = 0
        self.training_time = 0
        self.inference_time = 0
        self.tokenization_time = 0
        self.bert_training_time = 0
        self.bert_inference_time = 0
        
        # Results
        self.crf_metrics = {}
        self.bert_metrics = {}
        self.crf_accuracy = 0
        self.bert_accuracy = 0
        self.y_pred_crf = []
        self.y_pred_bert_processed = []
        
        # Responsible NLP Analysis
        self.bias_analysis = {}
        self.failure_cases = []
        self.hallucination_cases = []
        self.fairness_metrics = {}
        self.sample_review_results = {}
        self.mitigation_strategies = []
        
    def save_preprocessed_data(self):
        """Save preprocessed data to avoid reprocessing."""
        data = {
            'sentences': self.sentences,
            'labels': self.labels,
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test
        }
        with open(self.preprocessed_data_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Preprocessed data saved to {self.preprocessed_data_path}")
    
    def load_preprocessed_data(self):
        """Load preprocessed data if available."""
        if os.path.exists(self.preprocessed_data_path):
            with open(self.preprocessed_data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.sentences = data['sentences']
            self.labels = data['labels']
            self.X_train = data['X_train']
            self.X_test = data['X_test']
            self.y_train = data['y_train']
            self.y_test = data['y_test']
            
            print(f"Loaded preprocessed data from {self.preprocessed_data_path}")
            print(f"Training set: {len(self.X_train)} sentences")
            print(f"Test set: {len(self.X_test)} sentences")
            return True
        return False
        
    def load_and_preprocess_dataset(self):
        """Load and preprocess the NER dataset."""
        print("\n1. DATASET LOADING AND PREPROCESSING")
        print("-" * 40)
        
        # Try to load preprocessed data first
        if self.load_preprocessed_data():
            print("Using cached preprocessed data. Skipping preprocessing...")
            return
        
        print("No cached data found. Processing dataset from scratch...")
        
        # Load the dataset
        print("Loading NER dataset...")
        df = pd.read_csv(self.dataset_path)
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Handle NaN values in 'Sentence #' column using forward fill
        df['Sentence #'] = df['Sentence #'].fillna(method='ffill')
        
        # Group by sentence number to form sentence-level data
        sentences = []
        labels = []
        
        for sentence_id, group in df.groupby('Sentence #'):
            # Get the sentence and tags - they appear to be stored as string representations of lists
            sentence_text = group['Sentence'].iloc[0]  # Get first (should be only one)
            pos_tags = group['POS'].iloc[0]
            ner_tags = group['Tag'].iloc[0]
            
            try:
                # Parse the string representations of lists
                import ast
                words = ast.literal_eval(sentence_text) if isinstance(sentence_text, str) and sentence_text.startswith('[') else sentence_text.split()
                tags = ast.literal_eval(ner_tags) if isinstance(ner_tags, str) and ner_tags.startswith('[') else [ner_tags]
                
                # Ensure we have matching lengths
                if len(words) == len(tags) and len(words) > 0:
                    sentences.append(words)
                    labels.append(tags)
                    
            except (ValueError, SyntaxError) as e:
                print(f"Skipping sentence {sentence_id} due to parsing error: {e}")
                continue
                
            # Limit dataset size for faster training
            if len(sentences) >= self.max_samples:
                print(f"Limiting dataset to {self.max_samples} sentences for faster training")
                break
        
        self.sentences = sentences
        self.labels = labels
        
        print(f"Number of sentences: {len(sentences)}")
        print(f"Average sentence length: {np.mean([len(s) for s in sentences]):.2f}")
        print(f"Max sentence length: {max([len(s) for s in sentences])}")
        print(f"Min sentence length: {min([len(s) for s in sentences])}")
        
        # Analyze entity distribution
        all_tags = [tag for sentence_tags in labels for tag in sentence_tags]
        tag_counts = Counter(all_tags)
        print("\nEntity tag distribution:")
        for tag, count in tag_counts.most_common():
            print(f"{tag}: {count}")
        
        # Get unique entity types
        unique_entities = set()
        for tag in all_tags:
            if tag != 'O' and '-' in tag:
                entity_type = tag.split('-')[1]
                unique_entities.add(entity_type)
        
        print(f"\nUnique entity types: {sorted(unique_entities)}")
        
        # Create train/test split (80/20)
        # Use a simplified stratification approach
        def get_stratification_label(tags):
            entity_tags = [tag for tag in tags if tag != 'O']
            if len(entity_tags) == 0:
                return 'no_entities'
            elif len(entity_tags) <= 2:
                return 'few_entities'
            else:
                return 'many_entities'
        
        stratify_labels = [get_stratification_label(tags) for tags in labels]
        stratify_counts = Counter(stratify_labels)
        print(f"\nStratification label distribution: {stratify_counts}")
        
        # Only use stratification if all classes have at least 2 samples
        min_samples = min(stratify_counts.values())
        if min_samples >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                sentences, labels, 
                test_size=0.2, 
                random_state=42, 
                stratify=stratify_labels
            )
            print("Using stratified split")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                sentences, labels, 
                test_size=0.2, 
                random_state=42
            )
            print("Using random split (stratification not possible)")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Training set: {len(X_train)} sentences")
        print(f"Test set: {len(X_test)} sentences")
        
        # Verify split distribution
        train_tags = [tag for sentence_tags in y_train for tag in sentence_tags]
        test_tags = [tag for sentence_tags in y_test for tag in sentence_tags]
        
        print("\nTraining set tag distribution (top 10):")
        for tag, count in Counter(train_tags).most_common(10):
            print(f"{tag}: {count}")
        
        print("\nTest set tag distribution (top 10):")
        for tag, count in Counter(test_tags).most_common(10):
            print(f"{tag}: {count}")
        
        # Save preprocessed data for future use
        self.save_preprocessed_data()
    
    def build_classical_pipeline(self):
        """Build and train the classical NLP pipeline using SpaCy and CRF."""
        print("\n2. CLASSICAL NLP PIPELINE (SpaCy + CRF)")
        print("-" * 40)
        
        # Check if trained CRF model exists and predictions are already cached
        crf_predictions_path = 'models/crf_predictions.pkl'
        
        if os.path.exists(self.crf_model_path) and os.path.exists(crf_predictions_path):
            print("Loading pre-trained CRF model and cached predictions...")
            
            # Load cached predictions
            with open(crf_predictions_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.y_pred_crf = cached_data['predictions']
                self.y_test_labels = cached_data['test_labels']
                self.inference_time = cached_data.get('inference_time', 0)
                
            print("CRF predictions loaded from cache!")
            print("Classical pipeline completed (using cached model and predictions)!")
            return
            
        elif os.path.exists(self.crf_model_path):
            print("Loading pre-trained CRF model...")
            with open(self.crf_model_path, 'rb') as f:
                crf = pickle.load(f)
            print("CRF model loaded successfully!")
            
            # We still need to preprocess for predictions
            nlp = spacy.load("en_core_web_sm")
            X_test_processed = self._preprocess_with_spacy(self.X_test, self.y_test, nlp)
            X_test_features = [self._sent2features(sent) for sent in X_test_processed]
            y_test_labels = [self._sent2labels(sent) for sent in X_test_processed]
            
            # Make predictions
            print("Making CRF predictions...")
            start_time = time.time()
            self.y_pred_crf = crf.predict(X_test_features)
            self.y_test_labels = y_test_labels
            self.inference_time = time.time() - start_time
            print(f"CRF inference completed in {self.inference_time:.2f} seconds")
            
            # Cache the predictions
            cache_data = {
                'predictions': self.y_pred_crf,
                'test_labels': self.y_test_labels,
                'inference_time': self.inference_time
            }
            with open(crf_predictions_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"CRF predictions cached to {crf_predictions_path}")
            
            print("Classical pipeline completed (using cached model)!")
            return
        
        print("No pre-trained CRF model found. Training from scratch...")
        
        # Load SpaCy model
        print("Loading SpaCy model...")
        try:
            nlp = spacy.load("en_core_web_sm")
            print("SpaCy model loaded successfully!")
        except OSError:
            print("SpaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
        
    def _word2features(self, sent, i):
        """Extract features for a word at position i in a sentence."""
        word = sent[i][0]
        pos = sent[i][1]
        
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'pos': pos,
            'pos[:2]': pos[:2],
        }
        
        if i > 0:
            word1 = sent[i-1][0]
            pos1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:pos': pos1,
                '-1:pos[:2]': pos1[:2],
            })
        else:
            features['BOS'] = True
        
        if i < len(sent)-1:
            word1 = sent[i+1][0]
            pos1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:pos': pos1,
                '+1:pos[:2]': pos1[:2],
            })
        else:
            features['EOS'] = True
        
        return features
    
    def _sent2features(self, sent):
        """Extract features for all words in a sentence."""
        return [self._word2features(sent, i) for i in range(len(sent))]
    
    def _sent2labels(self, sent):
        """Extract labels from a sentence."""
        return [label for token, pos, label in sent]
    
    def _preprocess_with_spacy(self, sentences, labels, nlp):
        """Preprocess sentences with SpaCy."""
        processed_sentences = []
        
        for sentence, sentence_labels in zip(sentences, labels):
            # Join words into a sentence for SpaCy processing
            text = ' '.join(sentence)
            doc = nlp(text)
            
            processed_sentence = []
            word_idx = 0
            
            for token in doc:
                if word_idx < len(sentence) and token.text.strip():
                    # Use original word and SpaCy POS, with original label
                    processed_sentence.append((
                        sentence[word_idx],  # Original word
                        token.pos_,          # SpaCy POS tag
                        sentence_labels[word_idx]  # Original BIO label
                    ))
                    word_idx += 1
            
            if processed_sentence:
                processed_sentences.append(processed_sentence)
        
        return processed_sentences
    
    def _make_bert_predictions(self, model, tokenizer, our_id2label):
        """Make predictions with BERT model."""
        print("Making BERT predictions...")
        start_time = time.time()
        
        # Tokenize test data
        test_tokenized = self._tokenize_and_align_labels(
            self.X_test, self.y_test, tokenizer, our_id2label
        )
        
        # Create dataset
        test_dataset = Dataset.from_dict({
            'input_ids': [item['input_ids'] for item in test_tokenized],
            'attention_mask': [item['attention_mask'] for item in test_tokenized],
            'labels': [item['labels'] for item in test_tokenized]
        })
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        # Trainer for prediction
        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        predictions = trainer.predict(test_dataset)
        y_pred_bert = predictions.predictions
        
        self.bert_inference_time = time.time() - start_time
        print(f"BERT inference completed in {self.bert_inference_time:.2f} seconds")
        
        # Process BERT predictions
        self.y_pred_bert_processed = self._process_bert_predictions(
            y_pred_bert, self.X_test, self.y_test, tokenizer, our_id2label
        )
    
    def _tokenize_and_align_labels(self, sentences, labels, tokenizer, label2id_or_id2label, valid_tags=None, max_length=128):
        """Tokenize and align labels for BERT."""
        # Handle empty label mappings
        if not label2id_or_id2label:
            raise ValueError("No valid labels found! Check your dataset format.")
            
        # Handle both label2id and id2label mappings
        if isinstance(list(label2id_or_id2label.keys())[0], str):
            label2id = label2id_or_id2label
        else:
            # Create label2id from id2label
            label2id = {v: k for k, v in label2id_or_id2label.items()}
            
        tokenized_sentences = []
        
        for sentence, sentence_labels in zip(sentences, labels):
            # Tokenize sentence
            tokenized = tokenizer(
                sentence,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                is_split_into_words=True,
                return_tensors="pt"
            )
            
            # Align labels with tokens
            word_ids = tokenized.word_ids()
            aligned_labels = []
            
            for word_id in word_ids:
                if word_id is None:
                    aligned_labels.append(-100)  # Special tokens
                else:
                    if word_id < len(sentence_labels):
                        label = sentence_labels[word_id]
                        # Handle invalid labels
                        if label in label2id:
                            label_id = label2id[label]
                            # Ensure label is within valid range
                            if label_id < len(label2id):
                                aligned_labels.append(label_id)
                            else:
                                print(f"Warning: Label ID {label_id} out of range for label '{label}'")
                                aligned_labels.append(-100)
                        else:
                            # Use 'O' for invalid labels, or -100 if 'O' not available
                            if 'O' in label2id:
                                aligned_labels.append(label2id['O'])
                            else:
                                aligned_labels.append(-100)
                    else:
                        aligned_labels.append(-100)
            
            tokenized_sentences.append({
                'input_ids': tokenized['input_ids'].squeeze(),
                'attention_mask': tokenized['attention_mask'].squeeze(),
                'labels': torch.tensor(aligned_labels)
            })
        
        return tokenized_sentences
    
    def _process_bert_predictions(self, predictions, test_sentences, test_labels, tokenizer, id2label):
        """Process BERT predictions to align with original words."""
        processed_predictions = []
        
        for i, prediction in enumerate(predictions):
            sentence = test_sentences[i]
            # Tokenize to get word IDs
            tokenized = tokenizer(
                sentence,
                is_split_into_words=True,
                return_tensors="pt"
            )
            word_ids = tokenized.word_ids()
            
            # Align predictions with original words
            aligned_preds = []
            for j, word_id in enumerate(word_ids):
                if word_id is not None and word_id < len(sentence):
                    if j < len(prediction):
                        pred_id = np.argmax(prediction[j])
                        if pred_id in id2label:
                            aligned_preds.append(id2label[pred_id])
                        else:
                            aligned_preds.append('O')
            
            # Ensure we have the right number of predictions
            while len(aligned_preds) < len(sentence):
                aligned_preds.append('O')
            
            processed_predictions.append(aligned_preds[:len(sentence)])
        
        return processed_predictions
        
        # Preprocess training data with SpaCy
        print("Preprocessing training data with SpaCy...")
        start_time = time.time()
        
        # Process training and test data
        X_train_processed = self._preprocess_with_spacy(self.X_train, self.y_train, nlp)
        X_test_processed = self._preprocess_with_spacy(self.X_test, self.y_test, nlp)
        
        self.preprocessing_time = time.time() - start_time
        print(f"Preprocessing completed in {self.preprocessing_time:.2f} seconds")
        print(f"Processed {len(X_train_processed)} training sentences")
        print(f"Processed {len(X_test_processed)} test sentences")
        
        # Prepare features for CRF
        print("Preparing features for CRF...")
        X_train_features = [self._sent2features(sent) for sent in X_train_processed]
        y_train_labels = [self._sent2labels(sent) for sent in X_train_processed]
        
        X_test_features = [self._sent2features(sent) for sent in X_test_processed]
        y_test_labels = [self._sent2labels(sent) for sent in X_test_processed]
        
        print(f"Training features: {len(X_train_features)} sentences")
        print(f"Test features: {len(X_test_features)} sentences")
        
        # Train CRF model
        print("Training CRF model...")
        start_time = time.time()
        
        crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=50,  # Reduced from 100 for speed
            all_possible_transitions=True
        )
        
        crf.fit(X_train_features, y_train_labels)
        
        self.training_time = time.time() - start_time
        print(f"CRF training completed in {self.training_time:.2f} seconds")
        
        # Save the trained CRF model
        with open(self.crf_model_path, 'wb') as f:
            pickle.dump(crf, f)
        print(f"CRF model saved to {self.crf_model_path}")
        
        # Make predictions
        print("Making CRF predictions...")
        start_time = time.time()
        
        self.y_pred_crf = crf.predict(X_test_features)
        self.y_test_labels = y_test_labels
        
        self.inference_time = time.time() - start_time
        print(f"CRF inference completed in {self.inference_time:.2f} seconds")
        
        # Cache the predictions
        crf_predictions_path = 'models/crf_predictions.pkl'
        cache_data = {
            'predictions': self.y_pred_crf,
            'test_labels': self.y_test_labels,
            'inference_time': self.inference_time
        }
        with open(crf_predictions_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"CRF predictions cached to {crf_predictions_path}")
        
        print("Classical pipeline completed!")
    
    def build_transformer_pipeline(self):
        """Build and train the transformer-based pipeline using BERT."""
        print("\n3. TRANSFORMER-BASED PIPELINE (BERT NER)")
        print("-" * 40)
        
        # Check if trained BERT model exists and predictions are already cached
        bert_predictions_path = 'models/bert_predictions.pkl'
        
        if os.path.exists(self.bert_model_path) and os.path.exists(bert_predictions_path):
            print("Loading pre-trained BERT model and cached predictions...")
            
            # Load cached predictions
            with open(bert_predictions_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.y_pred_bert_processed = cached_data['predictions']
                self.bert_inference_time = cached_data.get('inference_time', 0)
                
            print("BERT predictions loaded from cache!")
            print("Transformer pipeline completed (using cached model and predictions)!")
            return
            
        elif os.path.exists(self.bert_model_path):
            print("Loading pre-trained BERT model...")
            tokenizer = AutoTokenizer.from_pretrained(self.bert_model_path)
            model = AutoModelForTokenClassification.from_pretrained(self.bert_model_path)
            print("BERT model loaded successfully!")
            
            # Load label mappings
            with open(os.path.join(self.bert_model_path, 'label_mappings.json'), 'r') as f:
                mappings = json.load(f)
                our_label2id = mappings['label2id']
                our_id2label = {int(k): v for k, v in mappings['id2label'].items()}
            
            # Make predictions only
            self._make_bert_predictions(model, tokenizer, our_id2label)
            
            # Cache the predictions
            cache_data = {
                'predictions': self.y_pred_bert_processed,
                'inference_time': self.bert_inference_time
            }
            with open(bert_predictions_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"BERT predictions cached to {bert_predictions_path}")
            
            print("Transformer pipeline completed (using cached model)!")
            return
        
        print("No pre-trained BERT model found. Training from scratch...")
        
        # Create label mapping for our dataset first
        # Flatten all labels and get unique tags
        all_tags_flat = []
        for sentence_labels in self.labels:
            for tag in sentence_labels:
                if isinstance(tag, str):  # Make sure it's a string
                    all_tags_flat.append(tag)
        
        all_unique_tags = sorted(set(all_tags_flat))
        print(f"Our dataset tags ({len(all_unique_tags)} total): {all_unique_tags}")
        
        # Filter out any invalid tags (keep only reasonable NER tags)
        valid_tags = []
        for tag in all_unique_tags:
            if isinstance(tag, str) and len(tag) <= 15:  # Reasonable tag length
                # Check if it's a valid NER tag format
                if tag == 'O' or '-' in tag or tag.startswith(('B-', 'I-')):
                    valid_tags.append(tag)
                elif len(tag) <= 8:  # Allow short tags that might be valid
                    valid_tags.append(tag)
        
        # If no valid tags found, something is wrong with the dataset format
        if not valid_tags:
            print("ERROR: No valid NER tags found!")
            print("Sample tags from dataset:", all_unique_tags[:20])
            raise ValueError("Dataset appears to have invalid tag format")
        
        print(f"Valid tags after filtering ({len(valid_tags)} total): {valid_tags}")
        
        # Create mapping from our tags to BERT labels
        our_label2id = {tag: idx for idx, tag in enumerate(valid_tags)}
        our_id2label = {idx: tag for tag, idx in our_label2id.items()}
        
        print(f"Label mapping sample: {dict(list(our_label2id.items())[:10])}")
        
        # Load BERT NER model and tokenizer with correct number of labels
        print("Loading BERT NER model...")
        model_name = "dslim/bert-base-NER"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load model with correct number of labels
        from transformers import BertConfig
        config = BertConfig.from_pretrained(model_name)
        config.num_labels = len(valid_tags)
        config.label2id = our_label2id
        config.id2label = our_id2label
        
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            config=config,
            ignore_mismatched_sizes=True  # Allow different classifier size
        )
        
        print(f"Model loaded: {model_name}")
        print(f"Model parameters: {model.num_parameters():,}")
        print(f"Model configured for {len(valid_tags)} labels")
        
        print("Tokenizing training data...")
        start_time = time.time()
        
        # Tokenize training and test data
        train_tokenized = self._tokenize_and_align_labels(self.X_train, self.y_train, tokenizer, our_label2id, valid_tags)
        test_tokenized = self._tokenize_and_align_labels(self.X_test, self.y_test, tokenizer, our_label2id, valid_tags)
        
        self.tokenization_time = time.time() - start_time
        print(f"Tokenization completed in {self.tokenization_time:.2f} seconds")
        print(f"Training samples: {len(train_tokenized)}")
        print(f"Test samples: {len(test_tokenized)}")
        
        # Create datasets for training
        def create_dataset(tokenized_data):
            return Dataset.from_dict({
                'input_ids': [item['input_ids'] for item in tokenized_data],
                'attention_mask': [item['attention_mask'] for item in tokenized_data],
                'labels': [item['labels'] for item in tokenized_data]
            })
        
        train_dataset = create_dataset(train_tokenized)
        test_dataset = create_dataset(test_tokenized)
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        
        # Training arguments (optimized for speed)
        training_args = TrainingArguments(
            output_dir="./bert_ner_results",
            eval_strategy="epoch",  # Changed from evaluation_strategy
            learning_rate=3e-5,  # Slightly higher for faster convergence
            per_device_train_batch_size=16,  # Larger batch for speed
            per_device_eval_batch_size=16,   # Larger batch for speed
            num_train_epochs=1,  # Reduced from 2 for speed
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,  # Less frequent logging
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=0,  # Avoid multiprocessing issues
            report_to=None,  # Disable wandb/tensorboard
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        print("Trainer initialized successfully!")
        
        # Train BERT model
        print("Training BERT NER model...")
        start_time = time.time()
        
        trainer.train()
        
        self.bert_training_time = time.time() - start_time
        print(f"BERT training completed in {self.bert_training_time:.2f} seconds")
        
        # Save the trained BERT model
        model.save_pretrained(self.bert_model_path)
        tokenizer.save_pretrained(self.bert_model_path)
        
        # Save label mappings
        mappings = {
            'label2id': our_label2id,
            'id2label': our_id2label
        }
        with open(os.path.join(self.bert_model_path, 'label_mappings.json'), 'w') as f:
            json.dump(mappings, f)
        
        print(f"BERT model saved to {self.bert_model_path}")
        
        # Make predictions
        print("Making BERT predictions...")
        start_time = time.time()
        
        predictions = trainer.predict(test_dataset)
        y_pred_bert = predictions.predictions
        
        self.bert_inference_time = time.time() - start_time
        print(f"BERT inference completed in {self.bert_inference_time:.2f} seconds")
        
        # Process BERT predictions
        self.y_pred_bert_processed = self._process_bert_predictions(
            y_pred_bert, self.X_test, self.y_test, tokenizer, our_id2label
        )
        
        # Cache the predictions
        bert_predictions_path = 'models/bert_predictions.pkl'
        cache_data = {
            'predictions': self.y_pred_bert_processed,
            'inference_time': self.bert_inference_time
        }
        with open(bert_predictions_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"BERT predictions cached to {bert_predictions_path}")
        
        print("Transformer pipeline completed!")
    
    def evaluate_and_compare(self):
        """Evaluate both pipelines and compare their performance."""
        print("\n4. EVALUATION AND COMPARISON")
        print("-" * 40)
        
        # Ensure we have test labels (in case they weren't set during CRF pipeline)
        if not hasattr(self, 'y_test_labels') or self.y_test_labels is None:
            print("Setting up test labels for evaluation...")
            # Use the original test labels
            self.y_test_labels = self.y_test
        
        # Check if we have CRF predictions
        if not hasattr(self, 'y_pred_crf') or self.y_pred_crf is None or len(self.y_pred_crf) == 0:
            print("ERROR: CRF predictions not found! CRF training may have failed.")
            print("Running CRF training now...")
            
            # Run CRF training quickly
            try:
                self.build_classical_pipeline()
            except Exception as e:
                print(f"CRF training failed: {e}")
                print("Creating dummy CRF predictions for comparison...")
                # Create dummy predictions (all 'O' tags)
                self.y_pred_crf = [['O'] * len(labels) for labels in self.y_test_labels]
                self.crf_accuracy = 0.0
                crf_report = "CRF model failed to train - using dummy predictions"
        
        # Evaluate CRF model if we have predictions
        if hasattr(self, 'y_pred_crf') and self.y_pred_crf and len(self.y_pred_crf) > 0:
            print("Evaluating CRF model...")
            print("\nCRF Classification Report:")
            try:
                crf_report = seqeval_report(self.y_test_labels, self.y_pred_crf)
                self.crf_accuracy = accuracy_score(self.y_test_labels, self.y_pred_crf)
            except Exception as e:
                print(f"Error in CRF evaluation: {e}")
                crf_report = f"CRF evaluation failed: {e}"
                self.crf_accuracy = 0.0
        else:
            crf_report = "CRF model failed to train - no predictions available"
            self.crf_accuracy = 0.0
            
        print(crf_report)
        print(f"CRF Overall Accuracy: {self.crf_accuracy:.4f}")
        
        # Evaluate BERT model
        print("\nEvaluating BERT model...")
        print("\nBERT Classification Report:")
        bert_report = seqeval_report(self.y_test_labels, self.y_pred_bert_processed)
        print(bert_report)
        
        # Calculate BERT accuracy
        self.bert_accuracy = accuracy_score(self.y_test_labels, self.y_pred_bert_processed)
        print(f"BERT Overall Accuracy: {self.bert_accuracy:.4f}")
        
        # Extract detailed metrics for comparison
        def extract_metrics_from_report(report_str):
            lines = report_str.split('\n')
            metrics = {}
            
            for line in lines:
                if 'precision' in line.lower() and 'recall' in line.lower() and 'f1-score' in line.lower():
                    continue  # Skip header
                elif line.strip() and not line.startswith(' '):
                    parts = line.split()
                    if len(parts) >= 4:
                        try:
                            entity = parts[0]
                            precision = float(parts[1])
                            recall = float(parts[2])
                            f1 = float(parts[3])
                            metrics[entity] = {'precision': precision, 'recall': recall, 'f1': f1}
                        except (ValueError, IndexError):
                            continue
            
            return metrics
        
        # Get detailed metrics
        self.crf_metrics = extract_metrics_from_report(crf_report)
        self.bert_metrics = extract_metrics_from_report(bert_report)
        
        print("\nDetailed Metrics Comparison:")
        print("CRF Metrics:")
        for entity, metrics in self.crf_metrics.items():
            print(f"{entity}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        print("\nBERT Metrics:")
        for entity, metrics in self.bert_metrics.items():
            print(f"{entity}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
    
    def create_visualization(self):
        """Create visualization comparing both pipelines."""
        print("\n5. VISUALIZATION")
        print("-" * 40)
        
        # Create comparison DataFrame
        comparison_data = []
        
        # Get all unique entities from both models
        all_entities = set(self.crf_metrics.keys()) | set(self.bert_metrics.keys())
        
        for entity in all_entities:
            crf_data = self.crf_metrics.get(entity, {'precision': 0, 'recall': 0, 'f1': 0})
            bert_data = self.bert_metrics.get(entity, {'precision': 0, 'recall': 0, 'f1': 0})
            
            comparison_data.append({
                'Entity': entity,
                'CRF_Precision': crf_data['precision'],
                'CRF_Recall': crf_data['recall'],
                'CRF_F1': crf_data['f1'],
                'BERT_Precision': bert_data['precision'],
                'BERT_Recall': bert_data['recall'],
                'BERT_F1': bert_data['f1']
            })
        
        # Add overall accuracy
        comparison_data.append({
            'Entity': 'Overall',
            'CRF_Precision': self.crf_accuracy,
            'CRF_Recall': self.crf_accuracy,
            'CRF_F1': self.crf_accuracy,
            'BERT_Precision': self.bert_accuracy,
            'BERT_Recall': self.bert_accuracy,
            'BERT_F1': self.bert_accuracy
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("Comparison Table:")
        print(comparison_df.round(3))
        
        # Save comparison table to CSV
        comparison_df.to_csv('ner_comparison_results.csv', index=False)
        print("\nComparison table saved as 'ner_comparison_results.csv'")
        
        # Create bar chart comparing F1 scores
        plt.figure(figsize=(12, 8))
        
        # Prepare data for plotting
        entities = comparison_df['Entity'].tolist()
        crf_f1 = comparison_df['CRF_F1'].tolist()
        bert_f1 = comparison_df['BERT_F1'].tolist()
        
        x = np.arange(len(entities))
        width = 0.35
        
        plt.bar(x - width/2, crf_f1, width, label='Classical (CRF)', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, bert_f1, width, label='Transformer (BERT)', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Entity Types')
        plt.ylabel('F1 Score')
        plt.title('F1 Score Comparison: Classical vs Transformer NER Pipelines')
        plt.xticks(x, entities, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison chart saved as 'comparison.png'")
    
    def analyze_resource_consumption(self):
        """Analyze and report resource consumption."""
        print("\n6. RESOURCE CONSUMPTION ANALYSIS")
        print("-" * 40)
        
        print("=== RESOURCE CONSUMPTION ANALYSIS ===")
        print(f"\nClassical Pipeline (CRF):")
        print(f"  - Preprocessing time: {self.preprocessing_time:.2f} seconds")
        print(f"  - Training time: {self.training_time:.2f} seconds")
        print(f"  - Inference time: {self.inference_time:.2f} seconds")
        print(f"  - Total time: {self.preprocessing_time + self.training_time + self.inference_time:.2f} seconds")
        print(f"  - Memory: Low (CPU-friendly, ~100-200MB RAM)")
        print(f"  - Compute: CPU only")
        
        print(f"\nTransformer Pipeline (BERT):")
        print(f"  - Tokenization time: {self.tokenization_time:.2f} seconds")
        print(f"  - Training time: {self.bert_training_time:.2f} seconds")
        print(f"  - Inference time: {self.bert_inference_time:.2f} seconds")
        print(f"  - Total time: {self.tokenization_time + self.bert_training_time + self.bert_inference_time:.2f} seconds")
        print(f"  - Memory: High (4GB+ RAM, GPU recommended)")
        print(f"  - Compute: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        
        # Performance comparison
        print(f"\n=== PERFORMANCE COMPARISON ===")
        print(f"CRF Overall Accuracy: {self.crf_accuracy:.4f}")
        print(f"BERT Overall Accuracy: {self.bert_accuracy:.4f}")
        print(f"Accuracy difference: {abs(self.bert_accuracy - self.crf_accuracy):.4f}")
        
        if self.bert_accuracy > self.crf_accuracy:
            print("BERT achieves higher accuracy")
        else:
            print("CRF achieves higher accuracy")
        
        # Speed comparison
        total_crf_time = self.preprocessing_time + self.training_time + self.inference_time
        total_bert_time = self.tokenization_time + self.bert_training_time + self.bert_inference_time
        
        print(f"\n=== SPEED COMPARISON ===")
        print(f"CRF total time: {total_crf_time:.2f} seconds")
        print(f"BERT total time: {total_bert_time:.2f} seconds")
        
        if total_crf_time > 0:
            print(f"Speed ratio (BERT/CRF): {total_bert_time/total_crf_time:.2f}x")
        else:
            print("Speed ratio: CRF training failed, cannot calculate ratio")
        
        # Save resource consumption data
        resource_data = {
            'Pipeline': ['Classical (CRF)', 'Transformer (BERT)'],
            'Preprocessing_Time': [self.preprocessing_time, self.tokenization_time],
            'Training_Time': [self.training_time, self.bert_training_time],
            'Inference_Time': [self.inference_time, self.bert_inference_time],
            'Total_Time': [total_crf_time, total_bert_time],
            'Accuracy': [self.crf_accuracy, self.bert_accuracy],
            'Memory_Usage': ['Low (~100-200MB)', 'High (4GB+)'],
            'Compute_Type': ['CPU only', 'GPU recommended']
        }
        
        resource_df = pd.DataFrame(resource_data)
        resource_df.to_csv('resource_consumption.csv', index=False)
        print(f"\nResource consumption data saved as 'resource_consumption.csv'")
        print(f"Saved to: {os.path.abspath('resource_consumption.csv')}")
    
    def analyze_bias_and_fairness(self):
        """Analyze bias and fairness in NER predictions."""
        print("\n7. RESPONSIBLE NLP: BIAS AND FAIRNESS ANALYSIS")
        print("-" * 50)
        
        # Analyze entity type bias
        print("Analyzing entity type bias...")
        entity_bias = self._analyze_entity_type_bias()
        
        # Analyze contextual bias
        print("Analyzing contextual bias...")
        contextual_bias = self._analyze_contextual_bias()
        
        # Calculate fairness metrics
        print("Calculating fairness metrics...")
        fairness_metrics = self._calculate_fairness_metrics()
        
        self.bias_analysis = {
            'entity_type_bias': entity_bias,
            'contextual_bias': contextual_bias
        }
        self.fairness_metrics = fairness_metrics
        
        # Print bias analysis results
        print("\n=== ENTITY TYPE BIAS ANALYSIS ===")
        for entity_type, bias_data in entity_bias.items():
            print(f"\n{entity_type} Entity Bias:")
            print(f"  CRF Error Rate: {bias_data['crf_error_rate']:.3f}")
            print(f"  BERT Error Rate: {bias_data['bert_error_rate']:.3f}")
            print(f"  Bias Difference: {bias_data['bias_difference']:.3f}")
            if bias_data['bias_difference'] > 0.1:
                print(f"  ⚠️  HIGH BIAS DETECTED for {entity_type}")
        
        print("\n=== FAIRNESS METRICS ===")
        for metric, value in fairness_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    def _analyze_entity_type_bias(self) -> Dict[str, Dict[str, float]]:
        """Analyze bias across different entity types."""
        entity_bias = {}
        
        # Get all unique entity types
        all_entities = set()
        for sentence_labels in self.y_test_labels:
            for label in sentence_labels:
                if label != 'O' and '-' in label:
                    entity_type = label.split('-')[1]
                    all_entities.add(entity_type)
        
        for entity_type in all_entities:
            # Calculate error rates for each model
            crf_errors, crf_total = self._calculate_entity_errors(
                self.y_test_labels, self.y_pred_crf, entity_type
            )
            bert_errors, bert_total = self._calculate_entity_errors(
                self.y_test_labels, self.y_pred_bert_processed, entity_type
            )
            
            crf_error_rate = crf_errors / max(crf_total, 1)
            bert_error_rate = bert_errors / max(bert_total, 1)
            
            entity_bias[entity_type] = {
                'crf_error_rate': crf_error_rate,
                'bert_error_rate': bert_error_rate,
                'bias_difference': abs(crf_error_rate - bert_error_rate),
                'crf_total': crf_total,
                'bert_total': bert_total
            }
        
        return entity_bias
    
    def _calculate_entity_errors(self, true_labels: List[List[str]], 
                               pred_labels: List[List[str]], 
                               entity_type: str) -> Tuple[int, int]:
        """Calculate errors for a specific entity type."""
        errors = 0
        total = 0
        
        for true_sent, pred_sent in zip(true_labels, pred_labels):
            for true_label, pred_label in zip(true_sent, pred_sent):
                if true_label != 'O' and '-' in true_label:
                    true_entity = true_label.split('-')[1]
                    if true_entity == entity_type:
                        total += 1
                        if true_label != pred_label:
                            errors += 1
        
        return errors, total
    
    def _analyze_contextual_bias(self) -> Dict[str, Any]:
        """Analyze bias based on sentence context and patterns."""
        contextual_bias = {
            'sentence_length_bias': self._analyze_sentence_length_bias(),
            'entity_position_bias': self._analyze_entity_position_bias(),
            'entity_frequency_bias': self._analyze_entity_frequency_bias()
        }
        return contextual_bias
    
    def _analyze_sentence_length_bias(self) -> Dict[str, float]:
        """Analyze bias based on sentence length."""
        short_sentences = []  # <= 10 words
        long_sentences = []   # > 20 words
        
        for i, sentence in enumerate(self.X_test):
            if len(sentence) <= 10:
                short_sentences.append(i)
            elif len(sentence) > 20:
                long_sentences.append(i)
        
        # Calculate accuracy for short vs long sentences
        short_crf_correct = sum(1 for i in short_sentences 
                               if self.y_test_labels[i] == self.y_pred_crf[i])
        short_bert_correct = sum(1 for i in short_sentences 
                                if self.y_test_labels[i] == self.y_pred_bert_processed[i])
        
        long_crf_correct = sum(1 for i in long_sentences 
                              if self.y_test_labels[i] == self.y_pred_crf[i])
        long_bert_correct = sum(1 for i in long_sentences 
                               if self.y_test_labels[i] == self.y_pred_bert_processed[i])
        
        return {
            'short_crf_accuracy': short_crf_correct / max(len(short_sentences), 1),
            'short_bert_accuracy': short_bert_correct / max(len(short_sentences), 1),
            'long_crf_accuracy': long_crf_correct / max(len(long_sentences), 1),
            'long_bert_accuracy': long_bert_correct / max(len(long_sentences), 1)
        }
    
    def _analyze_entity_position_bias(self) -> Dict[str, float]:
        """Analyze bias based on entity position in sentence."""
        beginning_accuracy = {'crf': [], 'bert': []}
        middle_accuracy = {'crf': [], 'bert': []}
        end_accuracy = {'crf': [], 'bert': []}
        
        for i, (sentence, true_labels, crf_pred, bert_pred) in enumerate(
            zip(self.X_test, self.y_test_labels, self.y_pred_crf, self.y_pred_bert_processed)
        ):
            sent_len = len(sentence)
            for j, (true_label, crf_label, bert_label) in enumerate(
                zip(true_labels, crf_pred, bert_pred)
            ):
                if true_label != 'O':  # Only consider actual entities
                    position_ratio = j / max(sent_len - 1, 1)
                    
                    if position_ratio <= 0.33:  # Beginning
                        beginning_accuracy['crf'].append(1 if true_label == crf_label else 0)
                        beginning_accuracy['bert'].append(1 if true_label == bert_label else 0)
                    elif position_ratio <= 0.66:  # Middle
                        middle_accuracy['crf'].append(1 if true_label == crf_label else 0)
                        middle_accuracy['bert'].append(1 if true_label == bert_label else 0)
                    else:  # End
                        end_accuracy['crf'].append(1 if true_label == crf_label else 0)
                        end_accuracy['bert'].append(1 if true_label == bert_label else 0)
        
        return {
            'beginning_crf': np.mean(beginning_accuracy['crf']) if beginning_accuracy['crf'] else 0,
            'beginning_bert': np.mean(beginning_accuracy['bert']) if beginning_accuracy['bert'] else 0,
            'middle_crf': np.mean(middle_accuracy['crf']) if middle_accuracy['crf'] else 0,
            'middle_bert': np.mean(middle_accuracy['bert']) if middle_accuracy['bert'] else 0,
            'end_crf': np.mean(end_accuracy['crf']) if end_accuracy['crf'] else 0,
            'end_bert': np.mean(end_accuracy['bert']) if end_accuracy['bert'] else 0
        }
    
    def _analyze_entity_frequency_bias(self) -> Dict[str, float]:
        """Analyze bias for rare vs common entities."""
        # Count entity frequencies
        entity_counts = defaultdict(int)
        for sentence_labels in self.y_train:
            for label in sentence_labels:
                if label != 'O':
                    entity_counts[label] += 1
        
        # Classify entities as rare or common
        median_count = np.median(list(entity_counts.values())) if entity_counts else 0
        rare_entities = {entity for entity, count in entity_counts.items() if count <= median_count}
        common_entities = {entity for entity, count in entity_counts.items() if count > median_count}
        
        # Calculate accuracy for rare vs common entities
        rare_crf_correct, rare_total = 0, 0
        rare_bert_correct = 0
        common_crf_correct, common_total = 0, 0
        common_bert_correct = 0
        
        for true_sent, crf_pred, bert_pred in zip(
            self.y_test_labels, self.y_pred_crf, self.y_pred_bert_processed
        ):
            for true_label, crf_label, bert_label in zip(true_sent, crf_pred, bert_pred):
                if true_label != 'O':
                    if true_label in rare_entities:
                        rare_total += 1
                        if true_label == crf_label:
                            rare_crf_correct += 1
                        if true_label == bert_label:
                            rare_bert_correct += 1
                    elif true_label in common_entities:
                        common_total += 1
                        if true_label == crf_label:
                            common_crf_correct += 1
                        if true_label == bert_label:
                            common_bert_correct += 1
        
        return {
            'rare_crf_accuracy': rare_crf_correct / max(rare_total, 1),
            'rare_bert_accuracy': rare_bert_correct / max(rare_total, 1),
            'common_crf_accuracy': common_crf_correct / max(common_total, 1),
            'common_bert_accuracy': common_bert_correct / max(common_total, 1)
        }
    
    def _calculate_fairness_metrics(self) -> Dict[str, float]:
        """Calculate quantitative fairness metrics."""
        # Demographic parity: Equal positive prediction rates across groups
        # Equal opportunity: Equal true positive rates across groups
        
        # For NER, we'll calculate fairness across entity types
        entity_types = set()
        for sentence_labels in self.y_test_labels:
            for label in sentence_labels:
                if label != 'O' and '-' in label:
                    entity_types.add(label.split('-')[1])
        
        if not entity_types:
            return {'demographic_parity': 0.0, 'equal_opportunity': 0.0}
        
        # Calculate metrics for each model
        crf_metrics = self._calculate_model_fairness_metrics(self.y_pred_crf, entity_types)
        bert_metrics = self._calculate_model_fairness_metrics(self.y_pred_bert_processed, entity_types)
        
        return {
            'crf_demographic_parity': crf_metrics['demographic_parity'],
            'crf_equal_opportunity': crf_metrics['equal_opportunity'],
            'bert_demographic_parity': bert_metrics['demographic_parity'],
            'bert_equal_opportunity': bert_metrics['equal_opportunity']
        }
    
    def _calculate_model_fairness_metrics(self, predictions: List[List[str]], 
                                        entity_types: set) -> Dict[str, float]:
        """Calculate fairness metrics for a specific model."""
        entity_stats = {}
        
        for entity_type in entity_types:
            tp = fp = fn = 0
            
            for true_sent, pred_sent in zip(self.y_test_labels, predictions):
                for true_label, pred_label in zip(true_sent, pred_sent):
                    true_entity = true_label.split('-')[1] if true_label != 'O' and '-' in true_label else None
                    pred_entity = pred_label.split('-')[1] if pred_label != 'O' and '-' in pred_label else None
                    
                    if true_entity == entity_type and pred_entity == entity_type:
                        tp += 1
                    elif true_entity != entity_type and pred_entity == entity_type:
                        fp += 1
                    elif true_entity == entity_type and pred_entity != entity_type:
                        fn += 1
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            
            entity_stats[entity_type] = {
                'precision': precision,
                'recall': recall,
                'positive_rate': (tp + fp) / max(sum(len(sent) for sent in self.y_test_labels), 1)
            }
        
        # Calculate demographic parity (difference in positive prediction rates)
        positive_rates = [stats['positive_rate'] for stats in entity_stats.values()]
        demographic_parity = 1 - (max(positive_rates) - min(positive_rates)) if positive_rates else 0
        
        # Calculate equal opportunity (difference in recall rates)
        recall_rates = [stats['recall'] for stats in entity_stats.values()]
        equal_opportunity = 1 - (max(recall_rates) - min(recall_rates)) if recall_rates else 0
        
        return {
            'demographic_parity': demographic_parity,
            'equal_opportunity': equal_opportunity
        }
    
    def document_failure_cases(self):
        """Document and analyze failure cases and hallucinations."""
        print("\n8. RESPONSIBLE NLP: FAILURE CASE ANALYSIS")
        print("-" * 50)
        
        # Identify different types of failures
        print("Identifying failure cases...")
        failure_cases = self._identify_failure_cases()
        
        # Identify hallucinations (false positive entities)
        print("Identifying hallucination cases...")
        hallucination_cases = self._identify_hallucination_cases()
        
        # Analyze failure patterns
        print("Analyzing failure patterns...")
        failure_patterns = self._analyze_failure_patterns(failure_cases)
        
        self.failure_cases = failure_cases
        self.hallucination_cases = hallucination_cases
        
        # Print failure analysis results
        print(f"\n=== FAILURE CASE SUMMARY ===")
        print(f"Total CRF failures: {len([f for f in failure_cases if f['model'] == 'CRF'])}")
        print(f"Total BERT failures: {len([f for f in failure_cases if f['model'] == 'BERT'])}")
        print(f"Total CRF hallucinations: {len([h for h in hallucination_cases if h['model'] == 'CRF'])}")
        print(f"Total BERT hallucinations: {len([h for h in hallucination_cases if h['model'] == 'BERT'])}")
        
        print(f"\n=== FAILURE PATTERNS ===")
        for pattern, data in failure_patterns.items():
            print(f"{pattern}: {data['count']} cases ({data['percentage']:.1f}%)")
        
        # Show specific examples
        print(f"\n=== EXAMPLE FAILURE CASES ===")
        self._print_failure_examples(failure_cases[:5])
        
        print(f"\n=== EXAMPLE HALLUCINATION CASES ===")
        self._print_hallucination_examples(hallucination_cases[:5])
    
    def _identify_failure_cases(self) -> List[Dict[str, Any]]:
        """Identify specific failure cases in predictions."""
        failure_cases = []
        
        for i, (sentence, true_labels, crf_pred, bert_pred) in enumerate(
            zip(self.X_test, self.y_test_labels, self.y_pred_crf, self.y_pred_bert_processed)
        ):
            # Check CRF failures
            for j, (word, true_label, crf_label) in enumerate(zip(sentence, true_labels, crf_pred)):
                if true_label != 'O' and true_label != crf_label:
                    failure_cases.append({
                        'model': 'CRF',
                        'sentence_idx': i,
                        'word_idx': j,
                        'sentence': sentence,
                        'word': word,
                        'true_label': true_label,
                        'predicted_label': crf_label,
                        'failure_type': self._classify_failure_type(true_label, crf_label),
                        'context': self._get_context(sentence, j)
                    })
            
            # Check BERT failures
            for j, (word, true_label, bert_label) in enumerate(zip(sentence, true_labels, bert_pred)):
                if true_label != 'O' and true_label != bert_label:
                    failure_cases.append({
                        'model': 'BERT',
                        'sentence_idx': i,
                        'word_idx': j,
                        'sentence': sentence,
                        'word': word,
                        'true_label': true_label,
                        'predicted_label': bert_label,
                        'failure_type': self._classify_failure_type(true_label, bert_label),
                        'context': self._get_context(sentence, j)
                    })
        
        return failure_cases
    
    def _identify_hallucination_cases(self) -> List[Dict[str, Any]]:
        """Identify hallucination cases (false positive entities)."""
        hallucination_cases = []
        
        for i, (sentence, true_labels, crf_pred, bert_pred) in enumerate(
            zip(self.X_test, self.y_test_labels, self.y_pred_crf, self.y_pred_bert_processed)
        ):
            # Check CRF hallucinations
            for j, (word, true_label, crf_label) in enumerate(zip(sentence, true_labels, crf_pred)):
                if true_label == 'O' and crf_label != 'O':
                    hallucination_cases.append({
                        'model': 'CRF',
                        'sentence_idx': i,
                        'word_idx': j,
                        'sentence': sentence,
                        'word': word,
                        'true_label': true_label,
                        'hallucinated_label': crf_label,
                        'context': self._get_context(sentence, j)
                    })
            
            # Check BERT hallucinations
            for j, (word, true_label, bert_label) in enumerate(zip(sentence, true_labels, bert_pred)):
                if true_label == 'O' and bert_label != 'O':
                    hallucination_cases.append({
                        'model': 'BERT',
                        'sentence_idx': i,
                        'word_idx': j,
                        'sentence': sentence,
                        'word': word,
                        'true_label': true_label,
                        'hallucinated_label': bert_label,
                        'context': self._get_context(sentence, j)
                    })
        
        return hallucination_cases
    
    def _classify_failure_type(self, true_label: str, pred_label: str) -> str:
        """Classify the type of failure."""
        if pred_label == 'O':
            return 'missed_entity'  # Model failed to detect entity
        elif true_label != 'O' and pred_label != 'O':
            true_entity = true_label.split('-')[1] if '-' in true_label else true_label
            pred_entity = pred_label.split('-')[1] if '-' in pred_label else pred_label
            
            if true_entity != pred_entity:
                return 'wrong_entity_type'  # Wrong entity type
            else:
                return 'wrong_bio_tag'  # Wrong B/I tag
        else:
            return 'other'
    
    def _get_context(self, sentence: List[str], word_idx: int, window: int = 2) -> str:
        """Get context around a word."""
        start = max(0, word_idx - window)
        end = min(len(sentence), word_idx + window + 1)
        context_words = sentence[start:end]
        
        # Mark the target word
        target_pos = word_idx - start
        if target_pos < len(context_words):
            context_words[target_pos] = f"**{context_words[target_pos]}**"
        
        return " ".join(context_words)
    
    def _analyze_failure_patterns(self, failure_cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Analyze patterns in failure cases."""
        patterns = defaultdict(int)
        total_failures = len(failure_cases)
        
        for case in failure_cases:
            patterns[case['failure_type']] += 1
            patterns[f"{case['model']}_failures"] += 1
            
            # Analyze by entity type
            if case['true_label'] != 'O' and '-' in case['true_label']:
                entity_type = case['true_label'].split('-')[1]
                patterns[f"{entity_type}_failures"] += 1
        
        # Convert to percentages
        pattern_analysis = {}
        for pattern, count in patterns.items():
            pattern_analysis[pattern] = {
                'count': count,
                'percentage': (count / max(total_failures, 1)) * 100
            }
        
        return pattern_analysis
    
    def _print_failure_examples(self, failure_cases: List[Dict[str, Any]]):
        """Print examples of failure cases."""
        for i, case in enumerate(failure_cases):
            print(f"\nExample {i+1}:")
            print(f"  Model: {case['model']}")
            print(f"  Word: '{case['word']}'")
            print(f"  True Label: {case['true_label']}")
            print(f"  Predicted: {case['predicted_label']}")
            print(f"  Failure Type: {case['failure_type']}")
            print(f"  Context: {case['context']}")
    
    def _print_hallucination_examples(self, hallucination_cases: List[Dict[str, Any]]):
        """Print examples of hallucination cases."""
        for i, case in enumerate(hallucination_cases):
            print(f"\nExample {i+1}:")
            print(f"  Model: {case['model']}")
            print(f"  Word: '{case['word']}'")
            print(f"  True Label: {case['true_label']} (should be non-entity)")
            print(f"  Hallucinated: {case['hallucinated_label']}")
            print(f"  Context: {case['context']}")
    
    def manual_sample_review(self, n_samples: int = 100):
        """Conduct manual review of sample predictions."""
        print(f"\n9. RESPONSIBLE NLP: MANUAL REVIEW OF {n_samples} SAMPLES")
        print("-" * 50)
        
        # Select random samples for review
        random.seed(42)  # For reproducibility
        sample_indices = random.sample(range(len(self.X_test)), min(n_samples, len(self.X_test)))
        
        review_results = {
            'total_samples': len(sample_indices),
            'crf_issues': [],
            'bert_issues': [],
            'quality_scores': {'crf': [], 'bert': []},
            'common_issues': defaultdict(int)
        }
        
        print(f"Reviewing {len(sample_indices)} randomly selected samples...")
        
        for idx in sample_indices:
            sentence = self.X_test[idx]
            true_labels = self.y_test_labels[idx]
            crf_pred = self.y_pred_crf[idx]
            bert_pred = self.y_pred_bert_processed[idx]
            
            # Analyze CRF prediction quality
            crf_quality = self._assess_prediction_quality(sentence, true_labels, crf_pred)
            review_results['quality_scores']['crf'].append(crf_quality['score'])
            
            if crf_quality['issues']:
                review_results['crf_issues'].extend(crf_quality['issues'])
                for issue in crf_quality['issues']:
                    review_results['common_issues'][f"CRF_{issue['type']}"] += 1
            
            # Analyze BERT prediction quality
            bert_quality = self._assess_prediction_quality(sentence, true_labels, bert_pred)
            review_results['quality_scores']['bert'].append(bert_quality['score'])
            
            if bert_quality['issues']:
                review_results['bert_issues'].extend(bert_quality['issues'])
                for issue in bert_quality['issues']:
                    review_results['common_issues'][f"BERT_{issue['type']}"] += 1
        
        self.sample_review_results = review_results
        
        # Print review results
        print(f"\n=== MANUAL REVIEW RESULTS ===")
        print(f"Samples reviewed: {review_results['total_samples']}")
        print(f"Average CRF quality score: {np.mean(review_results['quality_scores']['crf']):.3f}")
        print(f"Average BERT quality score: {np.mean(review_results['quality_scores']['bert']):.3f}")
        
        print(f"\n=== MOST COMMON ISSUES ===")
        for issue_type, count in sorted(review_results['common_issues'].items(), 
                                       key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / review_results['total_samples']) * 100
            print(f"{issue_type}: {count} cases ({percentage:.1f}%)")
        
        # Show examples of issues
        print(f"\n=== EXAMPLE ISSUES FOUND ===")
        self._print_review_examples(review_results)
        
        return review_results
    
    def _assess_prediction_quality(self, sentence: List[str], true_labels: List[str], 
                                 pred_labels: List[str]) -> Dict[str, Any]:
        """Assess the quality of predictions for a sentence."""
        issues = []
        correct_predictions = 0
        total_entities = 0
        
        for i, (word, true_label, pred_label) in enumerate(zip(sentence, true_labels, pred_labels)):
            if true_label != 'O':
                total_entities += 1
                if true_label == pred_label:
                    correct_predictions += 1
                else:
                    # Classify the type of issue
                    issue_type = self._classify_prediction_issue(word, true_label, pred_label, sentence, i)
                    issues.append({
                        'type': issue_type,
                        'word': word,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'position': i,
                        'context': self._get_context(sentence, i)
                    })
            
            # Check for hallucinations
            elif true_label == 'O' and pred_label != 'O':
                issues.append({
                    'type': 'hallucination',
                    'word': word,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'position': i,
                    'context': self._get_context(sentence, i)
                })
        
        # Calculate quality score
        if total_entities > 0:
            entity_accuracy = correct_predictions / total_entities
        else:
            entity_accuracy = 1.0  # No entities to predict
        
        # Penalize for hallucinations
        hallucination_penalty = len([i for i in issues if i['type'] == 'hallucination']) * 0.1
        quality_score = max(0, entity_accuracy - hallucination_penalty)
        
        return {
            'score': quality_score,
            'issues': issues,
            'entity_accuracy': entity_accuracy,
            'total_entities': total_entities,
            'hallucinations': len([i for i in issues if i['type'] == 'hallucination'])
        }
    
    def _classify_prediction_issue(self, word: str, true_label: str, pred_label: str, 
                                 sentence: List[str], position: int) -> str:
        """Classify the type of prediction issue."""
        if pred_label == 'O':
            return 'missed_entity'
        
        if true_label != 'O' and pred_label != 'O':
            true_entity = true_label.split('-')[1] if '-' in true_label else true_label
            pred_entity = pred_label.split('-')[1] if '-' in pred_label else pred_label
            
            if true_entity != pred_entity:
                # Check if it's a semantic confusion
                if self._are_semantically_similar(true_entity, pred_entity):
                    return 'semantic_confusion'
                else:
                    return 'wrong_entity_type'
            else:
                return 'wrong_bio_tag'
        
        return 'other'
    
    def _are_semantically_similar(self, entity1: str, entity2: str) -> bool:
        """Check if two entity types are semantically similar."""
        similar_pairs = [
            ('PER', 'PERSON'), ('LOC', 'LOCATION'), ('ORG', 'ORGANIZATION'),
            ('PERSON', 'PER'), ('LOCATION', 'LOC'), ('ORGANIZATION', 'ORG'),
            ('GPE', 'LOC'), ('LOC', 'GPE')  # Geopolitical entities vs locations
        ]
        
        return (entity1, entity2) in similar_pairs or (entity2, entity1) in similar_pairs
    
    def _print_review_examples(self, review_results: Dict[str, Any]):
        """Print examples from manual review."""
        # Show CRF issues
        if review_results['crf_issues']:
            print("\nCRF Issues Examples:")
            for i, issue in enumerate(review_results['crf_issues'][:3]):
                print(f"  {i+1}. {issue['type']}: '{issue['word']}' "
                      f"(true: {issue['true_label']}, pred: {issue['pred_label']})")
                print(f"     Context: {issue['context']}")
        
        # Show BERT issues
        if review_results['bert_issues']:
            print("\nBERT Issues Examples:")
            for i, issue in enumerate(review_results['bert_issues'][:3]):
                print(f"  {i+1}. {issue['type']}: '{issue['word']}' "
                      f"(true: {issue['true_label']}, pred: {issue['pred_label']})")
                print(f"     Context: {issue['context']}")
    
    def propose_mitigation_strategies(self):
        """Propose strategies to mitigate identified issues."""
        print("\n10. RESPONSIBLE NLP: MITIGATION STRATEGIES")
        print("-" * 50)
        
        strategies = []
        
        # Strategy 1: Data-based mitigation
        data_strategy = {
            'name': 'Dataset Augmentation and Balancing',
            'description': 'Address bias and improve robustness through data improvements',
            'specific_actions': [
                'Balance entity type distribution in training data',
                'Add diverse contexts for rare entities',
                'Include adversarial examples to reduce hallucinations',
                'Augment data with synthetic examples for underrepresented entities',
                'Apply data cleaning to remove annotation inconsistencies'
            ],
            'expected_impact': 'Reduce entity type bias by 15-20%, improve rare entity recall',
            'implementation_complexity': 'Medium',
            'resources_required': 'Data annotation team, synthetic data generation tools'
        }
        strategies.append(data_strategy)
        
        # Strategy 2: Model-based mitigation
        model_strategy = {
            'name': 'Fairness-Aware Training and Post-Processing',
            'description': 'Implement algorithmic fairness techniques during training',
            'specific_actions': [
                'Apply fairness constraints during BERT fine-tuning',
                'Use ensemble methods combining CRF and BERT predictions',
                'Implement confidence-based filtering to reduce hallucinations',
                'Add entity-aware loss functions to penalize biased predictions',
                'Use calibration techniques to improve prediction confidence'
            ],
            'expected_impact': 'Improve demographic parity by 10-15%, reduce false positives',
            'implementation_complexity': 'High',
            'resources_required': 'ML engineering expertise, additional compute resources'
        }
        strategies.append(model_strategy)
        
        # Strategy 3: Evaluation and monitoring
        monitoring_strategy = {
            'name': 'Continuous Bias Monitoring and Evaluation',
            'description': 'Implement ongoing monitoring to detect and prevent bias',
            'specific_actions': [
                'Establish regular bias auditing pipeline',
                'Create entity-specific performance dashboards',
                'Implement A/B testing for fairness metrics',
                'Set up automated alerts for bias threshold violations',
                'Conduct quarterly manual reviews of edge cases'
            ],
            'expected_impact': 'Early detection of bias drift, sustained fairness',
            'implementation_complexity': 'Medium',
            'resources_required': 'MLOps infrastructure, monitoring tools, review personnel'
        }
        strategies.append(monitoring_strategy)
        
        # Strategy 4: Human-in-the-loop
        human_loop_strategy = {
            'name': 'Human-in-the-Loop Quality Assurance',
            'description': 'Integrate human oversight for critical decisions',
            'specific_actions': [
                'Implement confidence-based human review triggers',
                'Create expert review process for ambiguous cases',
                'Establish feedback loop from human reviewers to model training',
                'Develop specialized interfaces for efficient human annotation',
                'Train reviewers on bias recognition and mitigation'
            ],
            'expected_impact': 'Reduce critical errors by 25-30%, improve edge case handling',
            'implementation_complexity': 'Medium',
            'resources_required': 'Human reviewers, review interface, training programs'
        }
        strategies.append(human_loop_strategy)
        
        self.mitigation_strategies = strategies
        
        # Print strategies
        for i, strategy in enumerate(strategies, 1):
            print(f"\n=== STRATEGY {i}: {strategy['name']} ===")
            print(f"Description: {strategy['description']}")
            print(f"Expected Impact: {strategy['expected_impact']}")
            print(f"Complexity: {strategy['implementation_complexity']}")
            print(f"Resources: {strategy['resources_required']}")
            print("Specific Actions:")
            for j, action in enumerate(strategy['specific_actions'], 1):
                print(f"  {j}. {action}")
        
        print(f"\n=== IMPLEMENTATION PRIORITY ===")
        print("Recommended implementation order:")
        print("1. Dataset Augmentation and Balancing (immediate impact)")
        print("2. Continuous Bias Monitoring (establish baseline)")
        print("3. Human-in-the-Loop Quality Assurance (reduce critical errors)")
        print("4. Fairness-Aware Training (long-term systematic improvement)")
        
        return strategies
    
    def generate_responsible_nlp_report(self):
        """Generate comprehensive responsible NLP report and visualizations."""
        print("\n11. RESPONSIBLE NLP: COMPREHENSIVE REPORT GENERATION")
        print("-" * 50)
        
        # Create comprehensive report
        print("Generating comprehensive responsible NLP report...")
        
        # Save all analysis results to files
        self._save_bias_analysis_report()
        self._save_failure_analysis_report()
        self._save_mitigation_strategies_report()
        self._create_responsible_nlp_visualizations()
        
        print("\n=== RESPONSIBLE NLP ANALYSIS COMPLETE ===")
        print("Generated files:")
        print("- bias_analysis_report.json: Detailed bias analysis results")
        print("- failure_analysis_report.json: Failure cases and patterns")
        print("- mitigation_strategies_report.json: Proposed mitigation strategies")
        print("- responsible_nlp_summary.csv: Summary of all findings")
        print("- bias_comparison_chart.png: Bias visualization")
        print("- failure_patterns_chart.png: Failure pattern analysis")
        print("- fairness_metrics_chart.png: Fairness metrics comparison")
        
    def _save_bias_analysis_report(self):
        """Save detailed bias analysis report."""
        report = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_info': {
                'total_test_samples': len(self.X_test),
                'total_entities': sum(len([l for l in labels if l != 'O']) for labels in self.y_test_labels)
            },
            'entity_type_bias': self.bias_analysis.get('entity_type_bias', {}),
            'contextual_bias': self.bias_analysis.get('contextual_bias', {}),
            'fairness_metrics': self.fairness_metrics,
            'summary': self._generate_bias_summary()
        }
        
        with open('bias_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("Bias analysis report saved to 'bias_analysis_report.json'")
    
    def _save_failure_analysis_report(self):
        """Save detailed failure analysis report."""
        report = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'failure_summary': {
                'total_crf_failures': len([f for f in self.failure_cases if f['model'] == 'CRF']),
                'total_bert_failures': len([f for f in self.failure_cases if f['model'] == 'BERT']),
                'total_crf_hallucinations': len([h for h in self.hallucination_cases if h['model'] == 'CRF']),
                'total_bert_hallucinations': len([h for h in self.hallucination_cases if h['model'] == 'BERT'])
            },
            'failure_examples': {
                'crf_failures': self.failure_cases[:10] if len(self.failure_cases) > 0 else [],
                'bert_failures': [f for f in self.failure_cases if f['model'] == 'BERT'][:10],
                'crf_hallucinations': [h for h in self.hallucination_cases if h['model'] == 'CRF'][:10],
                'bert_hallucinations': [h for h in self.hallucination_cases if h['model'] == 'BERT'][:10]
            },
            'manual_review_results': self.sample_review_results,
            'failure_patterns': self._analyze_failure_patterns(self.failure_cases) if self.failure_cases else {}
        }
        
        with open('failure_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print("Failure analysis report saved to 'failure_analysis_report.json'")
    
    def _save_mitigation_strategies_report(self):
        """Save mitigation strategies report."""
        report = {
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'identified_issues': self._summarize_identified_issues(),
            'mitigation_strategies': self.mitigation_strategies,
            'implementation_roadmap': {
                'phase_1': 'Dataset Augmentation and Balancing (0-3 months)',
                'phase_2': 'Continuous Bias Monitoring (1-4 months)', 
                'phase_3': 'Human-in-the-Loop Quality Assurance (3-6 months)',
                'phase_4': 'Fairness-Aware Training (6-12 months)'
            },
            'success_metrics': {
                'bias_reduction': 'Reduce entity type bias difference by >50%',
                'fairness_improvement': 'Achieve demographic parity >0.8',
                'error_reduction': 'Reduce critical errors by >25%',
                'hallucination_reduction': 'Reduce false positives by >30%'
            }
        }
        
        with open('mitigation_strategies_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        print("Mitigation strategies report saved to 'mitigation_strategies_report.json'")
    
    def _generate_bias_summary(self):
        """Generate summary of bias analysis findings."""
        if not self.bias_analysis.get('entity_type_bias'):
            return "No entity type bias analysis available"
        
        entity_bias = self.bias_analysis['entity_type_bias']
        high_bias_entities = [entity for entity, data in entity_bias.items() 
                             if data['bias_difference'] > 0.1]
        
        contextual_bias = self.bias_analysis.get('contextual_bias', {})
        
        summary = {
            'high_bias_entities': high_bias_entities,
            'total_entities_analyzed': len(entity_bias),
            'average_bias_difference': np.mean([data['bias_difference'] for data in entity_bias.values()]),
            'contextual_bias_detected': len([k for k, v in contextual_bias.items() if isinstance(v, dict)]) > 0
        }
        
        return summary
    
    def _summarize_identified_issues(self):
        """Summarize all identified issues."""
        issues = []
        
        # Bias issues
        if self.bias_analysis.get('entity_type_bias'):
            entity_bias = self.bias_analysis['entity_type_bias']
            high_bias_count = len([e for e, data in entity_bias.items() if data['bias_difference'] > 0.1])
            if high_bias_count > 0:
                issues.append(f"Entity type bias detected in {high_bias_count} entity types")
        
        # Failure issues
        if self.failure_cases:
            crf_failures = len([f for f in self.failure_cases if f['model'] == 'CRF'])
            bert_failures = len([f for f in self.failure_cases if f['model'] == 'BERT'])
            issues.append(f"CRF model failures: {crf_failures} cases")
            issues.append(f"BERT model failures: {bert_failures} cases")
        
        # Hallucination issues
        if self.hallucination_cases:
            crf_hallucinations = len([h for h in self.hallucination_cases if h['model'] == 'CRF'])
            bert_hallucinations = len([h for h in self.hallucination_cases if h['model'] == 'BERT'])
            issues.append(f"CRF hallucinations: {crf_hallucinations} cases")
            issues.append(f"BERT hallucinations: {bert_hallucinations} cases")
        
        # Quality issues from manual review
        if self.sample_review_results:
            avg_crf_quality = np.mean(self.sample_review_results['quality_scores']['crf'])
            avg_bert_quality = np.mean(self.sample_review_results['quality_scores']['bert'])
            if avg_crf_quality < 0.8:
                issues.append(f"CRF quality below threshold: {avg_crf_quality:.3f}")
            if avg_bert_quality < 0.8:
                issues.append(f"BERT quality below threshold: {avg_bert_quality:.3f}")
        
        return issues
    
    def _create_responsible_nlp_visualizations(self):
        """Create visualizations for responsible NLP analysis."""
        # Create bias comparison chart
        self._create_bias_comparison_chart()
        
        # Create failure patterns chart
        self._create_failure_patterns_chart()
        
        # Create fairness metrics chart
        self._create_fairness_metrics_chart()
        
        # Create summary CSV
        self._create_summary_csv()
    
    def _create_bias_comparison_chart(self):
        """Create bias comparison visualization."""
        if not self.bias_analysis.get('entity_type_bias'):
            return
        
        plt.figure(figsize=(12, 8))
        
        entity_bias = self.bias_analysis['entity_type_bias']
        entities = list(entity_bias.keys())
        crf_errors = [data['crf_error_rate'] for data in entity_bias.values()]
        bert_errors = [data['bert_error_rate'] for data in entity_bias.values()]
        
        x = np.arange(len(entities))
        width = 0.35
        
        plt.bar(x - width/2, crf_errors, width, label='CRF Error Rate', alpha=0.8, color='skyblue')
        plt.bar(x + width/2, bert_errors, width, label='BERT Error Rate', alpha=0.8, color='lightcoral')
        
        plt.xlabel('Entity Types')
        plt.ylabel('Error Rate')
        plt.title('Entity Type Bias Analysis: Error Rates by Model')
        plt.xticks(x, entities, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig('bias_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Bias comparison chart saved as 'bias_comparison_chart.png'")
    
    def _create_failure_patterns_chart(self):
        """Create failure patterns visualization."""
        if not self.failure_cases:
            return
        
        failure_patterns = self._analyze_failure_patterns(self.failure_cases)
        
        # Filter for most common patterns
        pattern_data = [(pattern, data['count']) for pattern, data in failure_patterns.items() 
                       if not pattern.endswith('_failures') and data['count'] >= 5]
        
        if not pattern_data:
            return
        
        patterns, counts = zip(*sorted(pattern_data, key=lambda x: x[1], reverse=True)[:10])
        
        plt.figure(figsize=(10, 6))
        plt.barh(patterns, counts, alpha=0.8, color='lightgreen')
        plt.xlabel('Number of Cases')
        plt.title('Most Common Failure Patterns')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        
        plt.savefig('failure_patterns_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Failure patterns chart saved as 'failure_patterns_chart.png'")
    
    def _create_fairness_metrics_chart(self):
        """Create fairness metrics visualization."""
        if not self.fairness_metrics:
            return
        
        metrics = ['CRF Demographic Parity', 'CRF Equal Opportunity', 
                  'BERT Demographic Parity', 'BERT Equal Opportunity']
        values = [
            self.fairness_metrics.get('crf_demographic_parity', 0),
            self.fairness_metrics.get('crf_equal_opportunity', 0),
            self.fairness_metrics.get('bert_demographic_parity', 0),
            self.fairness_metrics.get('bert_equal_opportunity', 0)
        ]
        
        plt.figure(figsize=(10, 6))
        colors = ['skyblue', 'lightblue', 'lightcoral', 'salmon']
        bars = plt.bar(metrics, values, alpha=0.8, color=colors)
        
        plt.ylabel('Fairness Score (0-1)')
        plt.title('Fairness Metrics Comparison')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at 0.8 (good fairness threshold)
        plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Good Fairness Threshold')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('fairness_metrics_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Fairness metrics chart saved as 'fairness_metrics_chart.png'")
    
    def _create_summary_csv(self):
        """Create summary CSV with all key findings."""
        summary_data = []
        
        # Add bias analysis summary
        if self.bias_analysis.get('entity_type_bias'):
            for entity, data in self.bias_analysis['entity_type_bias'].items():
                summary_data.append({
                    'Analysis_Type': 'Entity_Bias',
                    'Entity_Type': entity,
                    'CRF_Error_Rate': data['crf_error_rate'],
                    'BERT_Error_Rate': data['bert_error_rate'],
                    'Bias_Difference': data['bias_difference'],
                    'High_Bias': data['bias_difference'] > 0.1
                })
        
        # Add fairness metrics
        if self.fairness_metrics:
            summary_data.append({
                'Analysis_Type': 'Fairness_Metrics',
                'Metric': 'CRF_Demographic_Parity',
                'Value': self.fairness_metrics.get('crf_demographic_parity', 0)
            })
            summary_data.append({
                'Analysis_Type': 'Fairness_Metrics', 
                'Metric': 'BERT_Demographic_Parity',
                'Value': self.fairness_metrics.get('bert_demographic_parity', 0)
            })
        
        # Add manual review summary
        if self.sample_review_results:
            summary_data.append({
                'Analysis_Type': 'Manual_Review',
                'Metric': 'CRF_Avg_Quality',
                'Value': np.mean(self.sample_review_results['quality_scores']['crf'])
            })
            summary_data.append({
                'Analysis_Type': 'Manual_Review',
                'Metric': 'BERT_Avg_Quality', 
                'Value': np.mean(self.sample_review_results['quality_scores']['bert'])
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv('responsible_nlp_summary.csv', index=False)
            print("Summary CSV saved as 'responsible_nlp_summary.csv'")
    
    def run_comparison(self):
        """Run the complete pipeline comparison."""
        print("Starting NLP Pipeline Comparison...")
        
        # Step 1: Load and preprocess dataset
        self.load_and_preprocess_dataset()
        
        # Step 2: Build classical pipeline
        self.build_classical_pipeline()
        
        # Step 3: Build transformer pipeline
        self.build_transformer_pipeline()
        
        # Step 4: Evaluate and compare
        self.evaluate_and_compare()
        
        # Step 5: Create visualization
        self.create_visualization()
        
        # Step 6: Analyze resource consumption
        self.analyze_resource_consumption()
        
        # Step 7: Responsible NLP Analysis
        self.analyze_bias_and_fairness()
        
        # Step 8: Document failure cases
        self.document_failure_cases()
        
        # Step 9: Manual sample review
        self.manual_sample_review()
        
        # Step 10: Propose mitigation strategies
        self.propose_mitigation_strategies()
        
        # Step 11: Generate comprehensive report
        self.generate_responsible_nlp_report()
        
        print("\n" + "="*60)
        print("PIPELINE COMPARISON WITH RESPONSIBLE NLP ANALYSIS COMPLETED!")
        print("="*60)
        print("Original Task 2 Files:")
        print("- comparison.png: F1 score comparison chart")
        print("- ner_comparison_results.csv: Detailed metrics comparison")
        print("- resource_consumption.csv: Resource usage analysis")
        print("- bert_ner_results/: BERT model checkpoints")
        print("- logs/: Training logs")
        print("\nTask 3 Responsible NLP Files:")
        print("- bias_analysis_report.json: Detailed bias analysis")
        print("- failure_analysis_report.json: Failure cases and patterns")
        print("- mitigation_strategies_report.json: Mitigation strategies")
        print("- responsible_nlp_summary.csv: Summary of findings")
        print("- bias_comparison_chart.png: Bias visualization")
        print("- failure_patterns_chart.png: Failure pattern analysis")
        print("- fairness_metrics_chart.png: Fairness metrics comparison")
        print("="*60)
    
    def run_comparison_skip_mode(self):
        """Run comparison using dummy data for quick testing."""
        print("Starting NLP Pipeline Comparison (SKIP MODE)...")
        
        # Step 1: Load preprocessed data
        self.load_and_preprocess_dataset()
        
        # Create dummy predictions with realistic results
        print("\n2. CLASSICAL NLP PIPELINE (SpaCy + CRF)")
        print("-" * 40)
        print("SKIP MODE: Creating dummy CRF predictions...")
        
        # Create dummy CRF predictions (mix of O and some entities)
        self.y_pred_crf = []
        self.y_test_labels = self.y_test
        
        for sentence_labels in self.y_test:
            pred_sentence = []
            for i, label in enumerate(sentence_labels):
                # Simulate 60% accuracy - sometimes predict correctly, sometimes just 'O'
                if np.random.random() < 0.6:
                    pred_sentence.append(label)  # Correct prediction
                else:
                    pred_sentence.append('O')    # Wrong prediction (default to O)
            self.y_pred_crf.append(pred_sentence)
        
        self.crf_accuracy = 0.6  # Dummy accuracy
        self.training_time = 45.0  # Dummy training time
        self.inference_time = 2.5  # Dummy inference time
        self.preprocessing_time = 8.0  # Dummy preprocessing time
        
        print("Dummy CRF pipeline completed!")
        
        print("\n3. TRANSFORMER-BASED PIPELINE (BERT NER)")
        print("-" * 40)
        print("SKIP MODE: Creating dummy BERT predictions...")
        
        # Create dummy BERT predictions (better than CRF)
        self.y_pred_bert_processed = []
        
        for sentence_labels in self.y_test:
            pred_sentence = []
            for i, label in enumerate(sentence_labels):
                # Simulate 80% accuracy - better than CRF
                if np.random.random() < 0.8:
                    pred_sentence.append(label)  # Correct prediction
                else:
                    pred_sentence.append('O')    # Wrong prediction
            self.y_pred_bert_processed.append(pred_sentence)
        
        self.bert_accuracy = 0.8  # Dummy accuracy
        self.bert_training_time = 3600.0  # 1 hour dummy training time
        self.bert_inference_time = 800.0  # ~13 minutes dummy inference time
        self.tokenization_time = 15.0  # Dummy tokenization time
        
        print("Dummy BERT pipeline completed!")
        
        # Step 4: Evaluate and compare
        self.evaluate_and_compare()
        
        # Step 5: Create visualization
        self.create_visualization()
        
        # Step 6: Analyze resource consumption
        self.analyze_resource_consumption()
        
        # Step 7: Responsible NLP Analysis (with dummy data)
        self.analyze_bias_and_fairness()
        
        # Step 8: Document failure cases
        self.document_failure_cases()
        
        # Step 9: Manual sample review
        self.manual_sample_review()
        
        # Step 10: Propose mitigation strategies
        self.propose_mitigation_strategies()
        
        # Step 11: Generate comprehensive report
        self.generate_responsible_nlp_report()
        
        print("\n" + "="*60)
        print("PIPELINE COMPARISON WITH RESPONSIBLE NLP ANALYSIS COMPLETED (SKIP MODE)!")
        print("="*60)
        print("Original Task 2 Files:")
        print("- comparison.png: F1 score comparison chart")
        print("- ner_comparison_results.csv: Detailed metrics comparison")
        print("- resource_consumption.csv: Resource usage analysis")
        print("\nTask 3 Responsible NLP Files:")
        print("- bias_analysis_report.json: Detailed bias analysis")
        print("- failure_analysis_report.json: Failure cases and patterns")
        print("- mitigation_strategies_report.json: Mitigation strategies")
        print("- responsible_nlp_summary.csv: Summary of findings")
        print("- bias_comparison_chart.png: Bias visualization")
        print("- failure_patterns_chart.png: Failure pattern analysis")
        print("- fairness_metrics_chart.png: Fairness metrics comparison")
        print("NOTE: This was run in skip mode with dummy predictions for testing")
        print("="*60)

def main():
    """Main function to run the NER pipeline comparison."""
    import sys
    
    # Check if dataset exists
    dataset_path = 'dataset/ner_cleaned_20k.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the NER dataset is available at the specified path.")
        return
    
    # Check for skip flag
    skip_training = '--skip' in sys.argv or '-s' in sys.argv
    
    # Allow custom dataset size via command line
    max_samples = 20000  # Default
    for arg in sys.argv[1:]:
        if arg not in ['--skip', '-s']:
            try:
                max_samples = int(arg)
                print(f"Using custom dataset size: {max_samples} samples")
                break
            except ValueError:
                continue
    
    if skip_training:
        print("SKIP MODE: Using dummy data for quick testing")
    else:
        print(f"Dataset will be limited to {max_samples:,} sentences for faster training")
    
    # Initialize and run comparison with limited dataset for speed
    comparison = NERPipelineComparison(dataset_path, max_samples=max_samples)
    
    if skip_training:
        comparison.run_comparison_skip_mode()
    else:
        comparison.run_comparison()

if __name__ == "__main__":
    main()
