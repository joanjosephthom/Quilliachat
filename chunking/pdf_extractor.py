"""Module for extracting text from PDFs using multiple methods."""

import os
import io
import fitz  # PyMuPDF
import pdfplumber
from pdfminer.high_level import extract_text as pdfminer_extract_text
from typing import Optional

class PDFExtractor:
    """Extract text from PDFs using multiple methods."""
    
    @staticmethod
    def extract_text_with_pymupdf(file_path: str) -> str:
        """Extract text using PyMuPDF."""
        try:
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"PyMuPDF extraction error: {e}")
            return ""
    
    @staticmethod
    def extract_text_with_pdfplumber(file_path: str) -> str:
        """Extract text using pdfplumber."""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
            return text
        except Exception as e:
            print(f"pdfplumber extraction error: {e}")
            return ""
    
    @staticmethod
    def extract_text_with_pdfminer(file_path: str) -> str:
        """Extract text using pdfminer."""
        try:
            return pdfminer_extract_text(file_path)
        except Exception as e:
            print(f"pdfminer extraction error: {e}")
            return ""
    
    @classmethod
    def extract_text(cls, file_path: str) -> str:
        """Try multiple methods to extract text from a PDF."""
        # Try PyMuPDF first (fastest)
        text = cls.extract_text_with_pymupdf(file_path)
        if text.strip():
            print(f"Successfully extracted text with PyMuPDF: {len(text)} characters")
            return text
            
        # Try pdfplumber next
        text = cls.extract_text_with_pdfplumber(file_path)
        if text.strip():
            print(f"Successfully extracted text with pdfplumber: {len(text)} characters")
            return text
            
        # Try pdfminer as a last resort
        text = cls.extract_text_with_pdfminer(file_path)
        if text.strip():
            print(f"Successfully extracted text with pdfminer: {len(text)} characters")
            return text
            
        # If all methods fail
        print("All text extraction methods failed")
        return ""
