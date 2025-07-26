"""
Text Summarization Module using Hugging Face Transformers
Implements BART model for content summarization with text chunking capabilities
"""

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import re
from typing import List, Optional
import logging
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class TextSummarizer:
    """
    A class to handle text summarization using Facebook's BART model.
    
    Features:
    - Automatic text chunking for long documents
    - Configurable summary lengths
    - GPU support if available
    - Error handling and validation
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        """
        Initialize the summarizer with the specified model.
        
        Args:
            model_name (str): Hugging Face model identifier
        """
        self.model_name = model_name
        self.max_input_length = 1024  # BART's maximum input length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self._load_model()
        
        # Summary length configurations
        self.length_configs = {
            "short": {"max_length": 50, "min_length": 10},
            "medium": {"max_length": 130, "min_length": 30},
            "long": {"max_length": 200, "min_length": 50}
        }
    
    def _load_model(self):
        """Load the BART model and tokenizer"""
        try:
            print(f"Loading model: {self.model_name}")
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess the input text.
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters that might interfere with tokenization
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        
        # Ensure text ends with proper punctuation
        if text and text[-1] not in '.!?':
            text += '.'
        
        return text
    
    def _chunk_text(self, text: str, overlap: int = 50) -> List[str]:
        """
        Split long text into chunks that fit within model's input limit.
        
        Args:
            text (str): Text to chunk
            overlap (int): Number of tokens to overlap between chunks
            
        Returns:
            List[str]: List of text chunks
        """
        # Tokenize the full text to check length
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
        # If text fits within limit, return as single chunk
        if len(tokens) <= self.max_input_length:
            return [text]
        
        # Split text into sentences for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            sentence_length = len(sentence_tokens)
            
            # If adding this sentence would exceed limit, finalize current chunk
            if current_length + sentence_length > self.max_input_length - 100:  # Leave some buffer
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    # Start new chunk with overlap
                    if overlap > 0 and len(current_chunk) > 1:
                        overlap_sentences = current_chunk[-min(overlap//50, len(current_chunk)):]
                        current_chunk = overlap_sentences + [sentence]
                        current_length = len(self.tokenizer.encode(' '.join(current_chunk), add_special_tokens=False))
                    else:
                        current_chunk = [sentence]
                        current_length = sentence_length
                else:
                    # Single sentence is too long, truncate it
                    truncated_tokens = sentence_tokens[:self.max_input_length - 100]
                    truncated_text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
                    chunks.append(truncated_text)
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _generate_summary(self, text: str, max_length: int, min_length: int) -> str:
        """
        Generate summary for a single text chunk.
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum summary length
            min_length (int): Minimum summary length
            
        Returns:
            str: Generated summary
        """
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True,
                add_special_tokens=True
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,  # Beam search for better quality
                    length_penalty=2.0,  # Encourage longer summaries
                    early_stopping=True,
                    no_repeat_ngram_size=3,  # Avoid repetition
                    do_sample=False  # Deterministic output
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return summary.strip()
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate summary: {str(e)}")
    
    def _merge_summaries(self, summaries: List[str], target_length: str) -> str:
        """
        Merge multiple chunk summaries into a coherent final summary.
        
        Args:
            summaries (List[str]): List of individual summaries
            target_length (str): Target length configuration
            
        Returns:
            str: Final merged summary
        """
        if len(summaries) == 1:
            return summaries[0]
        
        # Combine all summaries
        combined_text = ' '.join(summaries)
        
        # If combined text is still reasonable, summarize it again
        config = self.length_configs[target_length]
        
        # Adjust parameters for final summarization
        final_max_length = min(config["max_length"], len(combined_text.split()) // 2)
        final_min_length = min(config["min_length"], final_max_length // 2)
        
        try:
            final_summary = self._generate_summary(
                combined_text,
                final_max_length,
                final_min_length
            )
            return final_summary
        except Exception:
            # Fallback: return first summary if merging fails
            return summaries[0]
    
    def summarize(self, text: str, length: str = "medium") -> str:
        """
        Main method to summarize text with automatic chunking.
        
        Args:
            text (str): Text to summarize
            length (str): Summary length ("short", "medium", "long")
            
        Returns:
            str: Generated summary
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If summarization fails
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
        
        if length not in self.length_configs:
            raise ValueError(f"Length must be one of {list(self.length_configs.keys())}")
        
        if len(text.strip()) < 50:
            raise ValueError("Text too short for meaningful summarization (minimum 50 characters)")
        
        try:
            # Clean the input text
            cleaned_text = self._clean_text(text)
            
            # Get length configuration
            config = self.length_configs[length]
            
            # Chunk the text if necessary
            chunks = self._chunk_text(cleaned_text)
            
            print(f"Processing {len(chunks)} chunk(s) for summarization...")
            
            # Generate summaries for each chunk
            summaries = []
            for i, chunk in enumerate(chunks):
                print(f"Summarizing chunk {i+1}/{len(chunks)}...")
                summary = self._generate_summary(
                    chunk,
                    config["max_length"],
                    config["min_length"]
                )
                summaries.append(summary)
            
            # Merge summaries if multiple chunks
            if len(summaries) > 1:
                print("Merging chunk summaries...")
                final_summary = self._merge_summaries(summaries, length)
            else:
                final_summary = summaries[0]
            
            # Final cleanup
            final_summary = final_summary.strip()
            
            # Ensure summary is not empty
            if not final_summary:
                raise RuntimeError("Generated summary is empty")
            
            print("Summary generation completed successfully!")
            return final_summary
            
        except ValueError as e:
            raise e
        except Exception as e:
            raise RuntimeError(f"Summarization failed: {str(e)}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_input_length": self.max_input_length,
            "available_lengths": list(self.length_configs.keys())
        }

# Example usage and testing
if __name__ == "__main__":
    # Test the summarizer
    test_text = """
    Artificial intelligence has become one of the most transformative technologies of the 21st century. 
    From healthcare to finance, AI is revolutionizing how we work, live, and interact with technology. 
    Machine learning algorithms can now process vast amounts of data and identify patterns that would be 
    impossible for humans to detect. Deep learning, a subset of machine learning, has enabled breakthroughs 
    in computer vision, natural language processing, and speech recognition. Companies worldwide are 
    investing billions of dollars in AI research and development, recognizing its potential to drive 
    innovation and competitive advantage. However, the rapid advancement of AI also raises important 
    questions about ethics, privacy, and the future of work. As AI systems become more sophisticated, 
    society must grapple with how to harness their benefits while mitigating potential risks.
    """
    
    try:
        summarizer = TextSummarizer()
        summary = summarizer.summarize(test_text, length="medium")
        print(f"Original text length: {len(test_text)} characters")
        print(f"Summary length: {len(summary)} characters")
        print(f"Summary: {summary}")
    except Exception as e:
        print(f"Error during testing: {e}")
