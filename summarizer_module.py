import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re
import fitz  # PyMuPDF
from typing import List

class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.max_input_tokens = 1024
        self.length_configs = {
            "short": {"max_length": 60, "min_length": 20},
            "medium": {"max_length": 130, "min_length": 40},
            "long": {"max_length": 200, "min_length": 60}
        }

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text.strip())

    def _chunk_text(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current = ""

        for sentence in sentences:
            current += sentence + " "
            tokenized = self.tokenizer.encode(current, add_special_tokens=False)
            if len(tokenized) > self.max_input_tokens:
                chunks.append(current.strip())
                current = sentence + " "
        if current:
            chunks.append(current.strip())
        return chunks

    def _generate_summary(self, text: str, max_len: int, min_len: int) -> str:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True,
            padding="longest"
        ).to(self.device)

        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    def summarize(self, raw_text: str, length: str = "medium") -> str:
        if not raw_text or len(raw_text.strip()) < 50:
            raise ValueError("Input too short for summarization.")

        config = self.length_configs.get(length, self.length_configs["medium"])
        cleaned = self._clean_text(raw_text)
        chunks = self._chunk_text(cleaned)

        partial_summaries = []
        for chunk in chunks:
            summary = self._generate_summary(chunk, config["max_length"], config["min_length"])
            partial_summaries.append(summary)

        final = " ".join(partial_summaries)
        return self._generate_summary(final, config["max_length"], config["min_length"]) if len(partial_summaries) > 1 else partial_summaries[0]

# ✅ Standalone helper function for streamlit_app
def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
        all_text = ""
        for page in doc:
            page_text = page.get_text("text")
            all_text += page_text + "\n"
        return all_text.strip()
    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        return ""
