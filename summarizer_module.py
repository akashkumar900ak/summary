import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import re
import logging
import warnings
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

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
        text = re.sub(r'\s+', ' ', text.strip())
        return text

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

    def _extract_figures_and_tables(self, text: str) -> str:
        lines = text.splitlines()
        figure_lines = [line.strip() for line in lines if re.search(r'\b(Fig\.|Figure|Table|Diagram)\b', line, re.IGNORECASE)]
        return "\n".join(figure_lines)

    def summarize(self, raw_text: str, length: str = "medium") -> str:
        if not raw_text or len(raw_text.strip()) < 50:
            raise ValueError("Input too short for summarization.")

        config = self.length_configs.get(length, self.length_configs["medium"])
        cleaned = self._clean_text(raw_text)
        chunks = self._chunk_text(cleaned)

        partial_summaries = []
        for i, chunk in enumerate(chunks):
            summary = self._generate_summary(chunk, config["max_length"], config["min_length"])
            partial_summaries.append(summary)

        if len(partial_summaries) > 1:
            combined = " ".join(partial_summaries)
            final_summary = self._generate_summary(combined, config["max_length"], config["min_length"])
        else:
            final_summary = partial_summaries[0]

        # Extract figure/table mentions
        figure_summary = self._extract_figures_and_tables(raw_text)

        return final_summary + ("\n\n📊 Key Figures/Tables:\n" + figure_summary if figure_summary else "")

    def summarize_pdf_with_images(self, pdf_path: str, length: str = "medium") -> str:
        doc = fitz.open(pdf_path)
        text = ""
        image_texts = []

        for page_num, page in enumerate(doc):
            # Extract text
            text += page.get_text()

            # Extract and OCR images
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    ocr_text = pytesseract.image_to_string(image)
                    if ocr_text.strip():
                        image_texts.append(f"[Page {page_num+1} - Image {img_index+1}]\n{ocr_text.strip()}")
                except Exception as e:
                    continue

        full_text = text + "\n".join(image_texts)
        return self.summarize(full_text, length)

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "max_input_length": self.max_input_tokens,
            "available_lengths": list(self.length_configs.keys())
        }

# 🔎 Optional: Run test directly
if __name__ == "__main__":
    sample_text = """
    Artificial Intelligence is shaping the future. This book covers chapters on vision, language, and robotics.
    Figure 1.2 illustrates how transformers process sequences. Table 3 shows performance across datasets.
    """
    summarizer = TextSummarizer()
    print("Text Summary:\n", summarizer.summarize(sample_text, length="medium"))
