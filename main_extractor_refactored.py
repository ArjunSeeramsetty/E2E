#!/usr/bin/env python3
"""
===================================================================
main_extractor.py (Refactored Orchestrator)
===================================================================
This script orchestrates the PDF table extraction pipeline. It extracts
raw tables and dispatches them to specialized processors for structuring,
validation, and standardization using a local LLM.
"""

import os
import sys
import json
import time
import re
import tempfile
import subprocess
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
import fitz  # PyMuPDF
from PyPDF2 import PdfMerger
from loguru import logger
import tabula

# --- 1. CONFIGURATION ---

# Add the project root to the Python path to import the processors module
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Import the dynamically created processor map
from src.processors import get_processor

# --- File Paths ---
INPUT_PDF_PATH = "data/raw/19.04.25_NLDC_PSP.pdf"
FINAL_OUTPUT_DIR = "data/finalized_for_db"

# --- Logging Configuration ---
Path("logs").mkdir(exist_ok=True)
Path(FINAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/main_extractor.log", rotation="10 MB", retention="7 days", level="DEBUG")


# --- 2. CORE EXTRACTION AND DISPATCH LOGIC ---

def check_dependencies():
    """Checks for external dependencies like Tesseract."""
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, text=True, check=True)
        logger.info("✅ Tesseract OCR is available.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("❌ Tesseract not found. The 'Enhanced OCR' mode will fail.")
        logger.info("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def extract_report_date_from_filename(pdf_path: str) -> str:
    """Extracts date from filename like 'DD.MM.YY_...' and returns 'YYYY-MM-DD'."""
    filename = Path(pdf_path).name
    match = re.search(r'(\d{2})\.(\d{2})\.(\d{2})', filename)
    if match:
        day, month, year = match.groups()
        return f"20{year}-{month}-{day}"
    return datetime.now().strftime("%Y-%m-%d")

def extract_tables_standard(pdf_path: str) -> List[pd.DataFrame]:
    """Extracts tables using tabula-py with a robust, multi-strategy approach."""
    logger.info(f"Running Standard Extraction on {pdf_path}")
    try:
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, lattice=True, stream=True, silent=True)
        valid_tables = [df for df in tables if isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] > 1]
        logger.info(f"Standard extraction found {len(valid_tables)} valid tables.")
        return valid_tables
    except Exception as e:
        logger.error(f"Standard extraction failed: {e}")
        return []

def extract_tables_enhanced_ocr(pdf_path: str) -> List[pd.DataFrame]:
    """A robust pipeline that OCRs the PDF and then extracts tables."""
    logger.info(f"Running Enhanced OCR Extraction on {pdf_path}...")
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            doc = fitz.open(pdf_path)
            image_files = []
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img_path = os.path.join(temp_dir, f"page_{i}.png")
                pix.save(img_path)
                image_files.append(img_path)
            doc.close()

            merger = PdfMerger()
            for img_path in image_files:
                pdf_part_path = img_path.replace('.png', '')
                subprocess.run(['tesseract', img_path, pdf_part_path, 'pdf'], check=True, capture_output=True)
                merger.append(f"{pdf_part_path}.pdf")

            merged_pdf_path = os.path.join(temp_dir, "ocr_final.pdf")
            merger.write(merged_pdf_path)
            merger.close()
            return extract_tables_standard(merged_pdf_path)
        except Exception as e:
            logger.error(f"Enhanced OCR extraction failed: {e}")
            return []

# --- 3. MAIN WORKFLOW ---

def main():
    """Main function to drive the extraction and standardization pipeline."""
    check_dependencies()
    report_date = extract_report_date_from_filename(INPUT_PDF_PATH)

    logger.info("\n" + "="*50)
    logger.info("  PDF Table Extraction and LLM Standardization Pipeline")
    logger.info(f"  Processing PDF: {Path(INPUT_PDF_PATH).name}")
    logger.info(f"  Report Date: {report_date}")
    logger.info("="*50)
    logger.info("Choose an extraction method:")
    logger.info("  1. Standard Extraction (Fast, `tabula-py` only)")
    logger.info("  2. Enhanced OCR Extraction (Slow, `Tesseract` + `tabula-py`)")
    logger.info("  3. Exit")
    choice = input("Enter your choice (1/2/3): ")

    raw_tables = []
    if choice == '1':
        raw_tables = extract_tables_standard(INPUT_PDF_PATH)
    elif choice == '2':
        raw_tables = extract_tables_enhanced_ocr(INPUT_PDF_PATH)
    elif choice == '3':
        logger.info("Exiting.")
        return
    else:
        logger.error("Invalid choice.")
        return

    if not raw_tables:
        logger.error("No tables were extracted.")
        return

    logger.info(f"\n🤖 Starting processor-based processing for {len(raw_tables)} raw tables...")
    final_structured_data = {}

    for i, table_df in enumerate(raw_tables):
        logger.info(f"--- Processing Raw Table {i+1}/{len(raw_tables)} ---")
        
        # Get the appropriate processor for the detected table type
        processor = get_processor(table_df)

        if not processor:
            logger.warning(f"No suitable processor found for Table {i+1}. Skipping.")
            continue

        # Use the processor to get structured, validated data
        structured_result = processor.process(table_df, report_date)

        if structured_result:
            table_type = processor.TABLE_TYPE
            if table_type not in final_structured_data:
                final_structured_data[table_type] = []
            
            if isinstance(structured_result, list):
                final_structured_data[table_type].extend(structured_result)
            else:
                final_structured_data[table_type].append(structured_result)

    logger.info("\n--- Saving Final Structured Data ---")
    for table_type, data_list in final_structured_data.items():
        output_path = Path(FINAL_OUTPUT_DIR) / f"{table_type}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved {len(data_list)} records to {output_path}")

    logger.info("\nWorkflow complete. Final JSON files are ready for database ingestion.")

if __name__ == "__main__":
    main() 