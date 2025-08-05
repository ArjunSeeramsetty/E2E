#!/usr/bin/env python3
"""
National Power Supply Data Warehouse - Main Extractor
This version uses a local LLM to intelligently structure and standardize
tables extracted from PDF reports.
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

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# --- File Paths ---
INPUT_PDF_PATH = "data/raw/19.04.25_NLDC_PSP.pdf"
FINAL_OUTPUT_DIR = "data/finalized_for_db"

# --- LLM Configuration ---
LLM_ENDPOINT = "http://localhost:11434/api/generate"  # Ollama endpoint
LLM_MODEL = "llama3.2:3b"  # Recommended: Use a powerful model like llama3 or phi3
LLM_TIMEOUT = 120  # Increased timeout for complex tables
LLM_MAX_RETRIES = 2

# --- Logging Configuration ---
Path("logs").mkdir(exist_ok=True)
Path(FINAL_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO")
logger.add("logs/main_extractor.log", rotation="10 MB", retention="7 days", level="DEBUG")


# --- 2. LLM PROMPTS AND TABLE CLASSIFICATION ---

# Keywords to classify each table type from its raw text content
TABLE_TYPE_PATTERNS = {
    'regional_daily_summary': ['peak demand', 'energy met', 'all india', 'regional summary'],
    'state_daily_summary': ['state', 'states', 'maximum demand', 'drawal schedule', 'od/ud'],
    'generation_by_source': ['sourcewise', 'gross generation', 'coal', 'hydro', 'nuclear', 'res'],
    'generation_outages': ['outage', 'central sector', 'state sector'],
    'power_exchanges': ['exchange', 'transnational', 'import/export', 'scheduled', 'actual'],
    'frequency_profile': ['frequency', 'fvi', 'frequency variation index'],
    'scada_timeseries': ['scada', 'time', 'instantaneous', 'frequency (hz)', 'demand met (mw)']
}

# One-shot prompts instructing the LLM on how to structure the data for each table type.
# This is the core of the intelligent standardization process.
TABLE_PROMPTS = {
    'regional_daily_summary': """
You are an expert data extraction agent. Your task is to analyze the following raw table text from a power supply report and convert it into a structured JSON array. Adhere strictly to the specified format and column names.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each region (NR, WR, SR, ER, NER, All India).
2. For each region, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers (int or float).

**Target JSON Schema:**
- `region_code`: (String) Region abbreviation.
- `power_peak_demand_met_evng_mw`: (Number) Demand Met during Evening Peak hrs(MW).
- `power_peak_shortage_evng_mw`: (Number) Peak Shortage (MW).
- `energy_met_total_mu`: (Number) Energy Met (MU).
- `energy_shortage_total_mu`: (Number) Energy Shortage (MU).
- `energy_generation_hydro_mu`: (Number) Hydro Gen (MU).
- `energy_generation_wind_mu`: (Number) Wind Gen (MU).
- `energy_generation_solar_mu`: (Number) Solar Gen (MU).
- `power_max_demand_met_day_mw`: (Number) Maximum Demand Met During the Day (MW).
- `time_of_max_demand_day`: (String) Time Of Maximum Demand Met.
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
""",
    'state_daily_summary': """
You are an expert data extraction agent. Your task is to analyze the following raw table text of state-wise power data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each state.
2. For each state, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `state_name`: (String) Name of the state.
- `power_state_max_demand_met_mw`: (Number) Max.Demand Met during the day (MW).
- `power_state_shortage_at_max_demand_mw`: (Number) Shortage during maximum Demand (MW).
- `energy_state_met_mu`: (Number) Energy Met (MU).
- `energy_state_drawal_schedule_mu`: (Number) Drawal Schedule (MU).
- `energy_state_over_under_drawal_mu`: (Number) OD(+)/UD(-)(MU).
- `power_state_max_overdrawal_mw`: (Number) Max OD (MW).
- `energy_state_shortage_mu`: (Number) Energy Shortage (MU).
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
""",
    'generation_by_source': """
You are an expert data extraction agent. Your task is to analyze the following raw table text of source-wise generation data and convert it into a structured JSON array.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each source type (Coal, Hydro, Nuclear, etc.).
2. For each source, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `source_type`: (String) Type of generation source (e.g., "COAL", "HYDRO", "NUCLEAR").
- `energy_gross_generation_mu`: (Number) Gross generation in MU.
- `energy_res_share_total_gen_pct`: (Number) RES share percentage.
- `energy_non_fossil_share_total_gen_pct`: (Number) Non-fossil share percentage.
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
""",
    'generation_outages': """
You are an expert data extraction agent. Your task is to analyze the following raw table text of generation outage data and convert it into a structured JSON array.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify outage data for each sector/region.
2. For each outage record, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `sector`: (String) Sector type (e.g., "CENTRAL", "STATE").
- `power_outage_mw`: (Number) Outage capacity in MW.
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
""",
    'power_exchanges': """
You are an expert data extraction agent. Your task is to analyze the following raw table text of power exchange data and convert it into a structured JSON array.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify exchange data for each entity.
2. For each exchange record, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `exchange_type`: (String) Type of exchange (e.g., "INTER_REGIONAL", "TRANSNATIONAL").
- `source_entity`: (String) Source entity (e.g., "WR", "BHUTAN").
- `destination_entity`: (String) Destination entity (e.g., "SR", "BANGLADESH").
- `energy_scheduled_exchange_mu`: (Number) Scheduled exchange in MU.
- `energy_actual_exchange_mu`: (Number) Actual exchange in MU.
- `energy_deviation_mu`: (Number) Deviation in MU.
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
""",
    'scada_timeseries': """
You are an expert data extraction agent. Your task is to analyze the following raw table text of SCADA time-series data and convert it into a structured JSON array.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify time-series data points.
2. For each data point, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `timestamp`: (String) Timestamp of the reading.
- `frequency_hz`: (Number) Grid frequency in Hz.
- `power_demand_met_mw`: (Number) Demand met in MW.
- `power_gen_nuclear_mw`: (Number) Nuclear generation in MW.
- `power_gen_wind_mw`: (Number) Wind generation in MW.
- `power_gen_solar_mw`: (Number) Solar generation in MW.
- `power_gen_hydro_mw`: (Number) Hydro generation in MW.
- `power_gen_gas_mw`: (Number) Gas generation in MW.
- `power_gen_thermal_mw`: (Number) Thermal generation in MW.
- `power_gen_others_mw`: (Number) Other generation in MW.
- `power_gen_total_mw`: (Number) Total generation in MW.
- `power_net_transnational_exchange_mw`: (Number) Net transnational exchange in MW.
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""
}


# --- 3. CORE FUNCTIONS ---

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
        # This single call with both lattice and stream is often the most effective.
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
            # 1. Convert PDF to images
            doc = fitz.open(pdf_path)
            image_files = []
            for i, page in enumerate(doc):
                pix = page.get_pixmap(dpi=300)
                img_path = os.path.join(temp_dir, f"page_{i}.png")
                pix.save(img_path)
                image_files.append(img_path)
            doc.close()

            # 2. OCR images to create a new searchable PDF
            merger = PdfMerger()
            for img_path in image_files:
                pdf_part_path = img_path.replace('.png', '')
                subprocess.run(['tesseract', img_path, pdf_part_path, 'pdf'], check=True, capture_output=True)
                merger.append(f"{pdf_part_path}.pdf")

            merged_pdf_path = os.path.join(temp_dir, "ocr_final.pdf")
            merger.write(merged_pdf_path)
            merger.close()

            # 3. Extract tables from the new OCR'd PDF
            return extract_tables_standard(merged_pdf_path)
        except Exception as e:
            logger.error(f"Enhanced OCR extraction failed: {e}")
            return []

def detect_table_type(table_df: pd.DataFrame) -> str:
    """Detects table type based on keywords in its content."""
    if table_df.empty:
        return 'unknown'
    # Combine headers and a sample of the data for better context
    content_sample = " ".join(map(str, table_df.columns)) + " " + " ".join(map(str, table_df.head(2).values.flatten()))
    content_sample = content_sample.lower()

    scores = {table_type: sum(1 for pattern in patterns if pattern in content_sample)
              for table_type, patterns in TABLE_TYPE_PATTERNS.items()}

    best_type = max(scores, key=scores.get)
    return best_type if scores[best_type] > 0 else 'unknown'

def call_llm_with_retry(prompt: str) -> Optional[str]:
    """Calls the local LLM with a retry mechanism for robustness."""
    for attempt in range(LLM_MAX_RETRIES):
        try:
            payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
            response = requests.post(LLM_ENDPOINT, json=payload, timeout=LLM_TIMEOUT)
            response.raise_for_status()
            
            # Parse the response
            response_data = response.json()
            return response_data.get('response', '')
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    return None

def process_table_with_llm(table_df: pd.DataFrame, report_date: str) -> Optional[Dict]:
    """Identifies, prompts, and processes a single raw table using the LLM."""
    table_type = detect_table_type(table_df)
    logger.info(f"Detected table type: '{table_type}'")

    if table_type == 'unknown' or table_type not in TABLE_PROMPTS:
        logger.warning("Skipping LLM processing for unknown table type.")
        return None

    # Convert raw table DataFrame to a clean string format for the prompt
    input_table_str = table_df.to_string(index=False, header=True)
    prompt_template = TABLE_PROMPTS[table_type]
    prompt = prompt_template.format(input_table=input_table_str, report_date=report_date)

    logger.debug(f"Sending prompt for '{table_type}' to LLM...")
    llm_response = call_llm_with_retry(prompt)

    if not llm_response:
        logger.error(f"LLM failed to process table of type '{table_type}'.")
        return None

    try:
        # The LLM is instructed to return JSON, so we parse it.
        structured_data = json.loads(llm_response)
        logger.info(f"Successfully parsed LLM response for '{table_type}'.")
        return {"type": table_type, "data": structured_data}
    except json.JSONDecodeError:
        logger.error(f"Failed to decode JSON from LLM response for '{table_type}'. Response: {llm_response[:200]}...")
        return None

# --- 4. MAIN WORKFLOW ---

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
        logger.error("No tables were extracted. Please check the PDF or try the other method.")
        return

    logger.info(f"\n🤖 Starting LLM processing for {len(raw_tables)} raw tables...")
    final_structured_data = {}

    for i, table_df in enumerate(raw_tables):
        logger.info(f"--- Processing Raw Table {i+1}/{len(raw_tables)} ---")
        structured_result = process_table_with_llm(table_df, report_date)

        if structured_result:
            table_type = structured_result["type"]
            if table_type not in final_structured_data:
                final_structured_data[table_type] = []
            # If the result is a list, extend the list. If it's a dict, append it.
            if isinstance(structured_result["data"], list):
                final_structured_data[table_type].extend(structured_result["data"])
            else:
                final_structured_data[table_type].append(structured_result["data"])

    logger.info("\n--- Saving Final Structured Data ---")
    for table_type, data_list in final_structured_data.items():
        output_path = Path(FINAL_OUTPUT_DIR) / f"{table_type}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved {len(data_list)} records to {output_path}")

    logger.info("\nWorkflow complete. Final JSON files are ready for database ingestion.")

if __name__ == "__main__":
    main() 