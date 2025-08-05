#!/usr/bin/env python3
"""
National Power Supply Data Warehouse - Hybrid Extractor
Refactored version with logical processor first, LLM fallback approach
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import re
import tempfile
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import functools
import io
import json
import time
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import tabula directly without any JVM configuration
import tabula

# Import other required libraries
import fitz  # PyMuPDF
import PyPDF2
from PIL import Image
import cv2
import numpy as np

# Import the processor system
from src.processors import get_processor
from src.processors import get_processor_info
from src.processors import mark_table_type_as_used, reset_used_table_types
from loguru import logger

# LLM Configuration
LLM_ENDPOINT = "http://localhost:11434/api/generate"  # Ollama endpoint
LLM_MODEL = "llama3.2:3b"  # Using available model
LLM_TIMEOUT = 300  # Increased timeout for complex tables
LLM_MAX_RETRIES = 3

# Table type detection patterns (for fallback)
TABLE_TYPE_PATTERNS = {
    'regional_daily_summary': [
        'demand met during evening peak', 'peak shortage', 'energy met', 'hydro gen', 'wind gen', 'solar gen', 'energy shortage', 'maximum demand met during the day', 'time of maximum demand met'
    ],
    'state_daily_summary': [
        'state_name', 'region_code', 'max_demand_met_day_mw', 'state_max_demand_met_mw', 'energy_met_total_mu', 'state_drawal_schedule_mu'
    ],
    'generation_by_source': [
        'source', 'sourcewise', 'gross generation', 'fuel type', 'coal', 'hydro', 'nuclear', 'thermal'
    ],
    'generation_outages': [
        'central sector', 'state sector', 'total', 'res_share_total_gen_pct'
    ],
    'inter_regional_transmission': [
        'inter-regional', 'voltage_level', 'line_details', 'sl_no', 'no_of_circuit', 'max import', 'max export', 'import', 'net'
    ],
    'transnational_transmission': [
        'transnational', 'state', 'region', 'line_name', 'max', 'min', 'avg', 'energy exchange'
    ],
    'frequency_profile': [
        'frequency_variation_index', 'frequency variation index', '49_7', '49_8', '49_9', '50_05', 'fvi'
    ],
    'solar_non_solar_peak': [
        'solar hr', 'non-solar hr', 'max demand met', 'peak demand', 'solar hour', 'non-solar hour'
    ],
    'scada_timeseries': [
        'scada', 'time', 'timestamp', 'instantaneous', '15-minute', 'all india scada', 'frequency', 'demand'
    ]
}

# One-shot prompts for each table type (for LLM fallback)
TABLE_PROMPTS = {
    'regional_daily_summary': """
You are a data processing expert. Convert the following table into a structured JSON format for Regional Power Supply Summary.

Input Table:
{input_table}

IMPORTANT: This table may have a confusing structure where the first column contains labels like "Schedule(MU)", "Actual(MU)", "O/D/U/D(MU)" and the actual data is in subsequent columns. You need to:

1. Identify the data structure - look for row labels vs column headers
2. Extract the actual numerical data, not the labels
3. Map the data correctly to the expected fields

Expected Output Format (JSON):
{{
    "region_code": "NR",
    "peak_demand_met_evng_mw": 12345.67,
    "energy_met_total_mu": 1234.56,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for ALL regions (NR, WR, SR, ER, NER, All India) if present in the table. Return an array of objects, one for each region with data.
""",
    'state_daily_summary': """
You are a data processing expert. Convert the following table into a structured JSON format for State Power Supply Summary.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
    "state_name": "Maharashtra",
    "region_code": "WR",
    "max_demand_met_day_mw": 12345.67,
    "state_max_demand_met_mw": 12345.67,
    "energy_met_total_mu": 1234.56,
    "state_drawal_schedule_mu": 1234.56,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for ALL states present in the table. Return an array of objects, one for each state with data.
""",
    'generation_by_source': """
You are a data processing expert. Convert the following table into a structured JSON format for Generation by Source.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
    "region_name": "NR",
    "source_name": "Coal",
    "energy_generation_mu": 1234.56,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for ALL regions (NR, WR, SR, ER, NER, All India) and ALL sources (Coal, Lignite, Hydro, Nuclear, Gas, Naptha & Diesel, RES, Total) if present in the table. Return an array of objects, one for each region-source combination with data.
""",
    'generation_outages': """
You are a data processing expert. Convert the following table into a structured JSON format for Generation Outages.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
    "region_name": "NR",
    "central_sector_mw": 1234.56,
    "state_sector_mw": 1234.56,
    "total_mw": 2469.12,
    "res_share_total_gen_pct": 12.34,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for ALL regions (NR, WR, SR, ER, NER, All India) if present in the table. Return an array of objects, one for each region with data.
""",
    'inter_regional_transmission': """
You are a data processing expert. Convert the following table into a structured JSON format for Inter-Regional Transmission.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
    "voltage_level": "765 kV",
    "line_details": "NR-WR",
    "sl_no": 1,
    "no_of_circuit": 2,
    "max_import_mw": 1234.56,
    "max_export_mw": 1234.56,
    "import_mw": 1234.56,
    "export_mw": 1234.56,
    "net_mw": 0.0,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for ALL transmission lines present in the table. Return an array of objects, one for each line with data.
""",
    'transnational_transmission': """
You are a data processing expert. Convert the following table into a structured JSON format for Transnational Transmission.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
    "state": "West Bengal",
    "region": "ER",
    "line_name": "India-Bangladesh",
    "max_mw": 1234.56,
    "min_mw": 123.45,
    "avg_mw": 678.90,
    "energy_exchange_mu": 123.45,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for ALL transnational lines present in the table. Return an array of objects, one for each line with data.
""",
    'frequency_profile': """
You are a data processing expert. Convert the following table into a structured JSON format for Frequency Profile.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
    "frequency_band": "Frequency (<49.7)",
    "percentage_time": 12.34,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for ALL frequency bands present in the table. Return an array of objects, one for each band with data.
""",
    'solar_non_solar_peak': """
You are a data processing expert. Convert the following table into a structured JSON format for Solar/Non-Solar Peak Demand.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
    "period_type": "Solar hr",
    "power_max_demand_met_mw": 12345.67,
    "timestamp": "14:30",
    "power_shortage_at_max_demand_mw": 123.45,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for BOTH Solar hr and Non-Solar hr periods if present in the table. Return an array of objects, one for each period with data.
""",
    'scada_timeseries': """
You are a data processing expert. Convert the following table into a structured JSON format for SCADA Time Series.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
    "timestamp": "2024-01-15T14:30:00",
    "frequency_hz": 50.02,
    "demand_mw": 123456.78,
    "report_date": "2024-01-15"
}}

IMPORTANT: Extract data for ALL time points present in the table. Return an array of objects, one for each time point with data.
"""
}

def extract_tables_standard(pdf_path):
    """
    Extract tables from PDF using tabula-py.
    """
    logger.info(f"Extracting tables from {pdf_path}")
    
    try:
        # Extract tables from all pages
        tables = tabula.read_pdf(
            pdf_path,
            pages='all',
            multiple_tables=True,
            guess=False,
            stream=True,
            pandas_options={'header': None}
        )
        
        logger.info(f"Extracted {len(tables)} raw tables from PDF")
        return tables
        
    except Exception as e:
        logger.error(f"Error extracting tables from PDF: {e}")
        return []

def detect_table_type_fallback(table_df: pd.DataFrame) -> str:
    """
    Fallback table type detection using pattern matching.
    """
    # Convert table to string for pattern matching
    table_str = table_df.to_string().lower()
    
    for table_type, patterns in TABLE_TYPE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in table_str:
                logger.info(f"Detected table type '{table_type}' using pattern '{pattern}'")
                return table_type
    
    logger.warning("Could not detect table type using patterns")
    return "unknown"

def call_llm_with_retry(prompt: str, model: str = LLM_MODEL, max_retries: int = LLM_MAX_RETRIES) -> Optional[str]:
    """
    Call LLM with retry logic for reliability.
    """
    for attempt in range(max_retries):
        try:
            response = requests.post(
                LLM_ENDPOINT,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "max_tokens": 4000
                    }
                },
                timeout=LLM_TIMEOUT
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.warning(f"LLM request failed (attempt {attempt + 1}/{max_retries}): {response.status_code}")
                
        except Exception as e:
            logger.warning(f"LLM request error (attempt {attempt + 1}/{max_retries}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error("All LLM retry attempts failed")
    return None

def process_table_with_llm_fallback(table_df: pd.DataFrame, table_index: int, report_date: str) -> Dict[str, Any]:
    """
    Process table using LLM as fallback when logical processor fails.
    """
    logger.info(f"Processing table {table_index} with LLM fallback...")
    
    try:
        # Detect table type using fallback method
        table_type = detect_table_type_fallback(table_df)
        
        if table_type == "unknown":
            return {
                'table_index': table_index,
                'table_type': 'unknown',
                'structured_data': None,
                'processing_method': 'llm_fallback',
                'error': 'Could not detect table type'
            }
        
        # Get prompt for this table type
        prompt_template = TABLE_PROMPTS.get(table_type)
        if not prompt_template:
            return {
                'table_index': table_index,
                'table_type': table_type,
                'structured_data': None,
                'processing_method': 'llm_fallback',
                'error': f'No prompt template for table type: {table_type}'
            }
        
        # Prepare table data for LLM
        table_str = table_df.to_string(index=False)
        prompt = prompt_template.format(input_table=table_str)
        
        # Call LLM
        logger.info(f"Calling LLM for table {table_index} ({table_type})...")
        llm_response = call_llm_with_retry(prompt)
        
        if not llm_response:
            return {
                'table_index': table_index,
                'table_type': table_type,
                'structured_data': None,
                'processing_method': 'llm_fallback',
                'error': 'LLM call failed'
            }
        
        # Parse LLM response
        try:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON array or object
                json_match = re.search(r'\[.*\]|\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    json_str = llm_response
            
            structured_data = json.loads(json_str)
            
            # Ensure it's a list
            if isinstance(structured_data, dict):
                structured_data = [structured_data]
            
            return {
                'table_index': table_index,
                'table_type': table_type,
                'structured_data': structured_data,
                'processing_method': 'llm_fallback',
                'error': None
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            return {
                'table_index': table_index,
                'table_type': table_type,
                'structured_data': None,
                'processing_method': 'llm_fallback',
                'error': f'JSON parsing failed: {e}'
            }
            
    except Exception as e:
        logger.error(f"Error processing table {table_index} with LLM: {e}")
        return {
            'table_index': table_index,
            'table_type': 'unknown',
            'structured_data': None,
            'processing_method': 'llm_fallback',
            'error': str(e)
        }

def process_table_hybrid(table_df: pd.DataFrame, report_date: str = "2025-04-19", detected_type: str = None) -> Dict[str, Any]:
    """
    Process a single table using the unified detection and processor selection system.
    Args:
        table_df: DataFrame containing the table data
        report_date: Report date string
        detected_type: Pre-detected table type (if None, will detect automatically)
    Returns:
        Dictionary containing processing results
    """
    try:
        # Use pre-detected type if provided, otherwise detect
        if detected_type is None:
            processor_info = get_processor_info(table_df)
            detected_type = processor_info['detected_type']
            confidence = processor_info['confidence']
        else:
            confidence = 1.0  # Assume high confidence for pre-detected types
        
        logger.info(f"Table detection: {detected_type} (confidence: {confidence:.2f})")

        # Mark this table type as used to prevent re-detection
        if detected_type != "unknown":
            mark_table_type_as_used(detected_type)

        # Get the appropriate processor (prefer logical processors) using the detected type
        processor = get_processor(table_df, prefer_logical=True, detected_type=detected_type)

        if processor is None:
            logger.error("No suitable processor found for table")
            return {
                "success": False,
                "error": "No suitable processor found",
                "detected_type": detected_type,
                "confidence": confidence
            }

        # Process the table
        logger.info(f"Processing with {processor.__class__.__name__}")
        processed_data = processor.process(table_df, report_date)

        if processed_data:
            # Validate the processed data
            if hasattr(processor, 'validate') and processor.validate(processed_data):
                logger.info(f"Successfully processed {len(processed_data)} records")

                # Generate summary if available
                summary = {}
                if hasattr(processor, 'get_summary'):
                    summary = processor.get_summary(processed_data)

                return {
                    "success": True,
                    "processor_type": processor.__class__.__name__,
                    "detected_type": detected_type,
                    "confidence": confidence,
                    "records_count": len(processed_data),
                    "data": processed_data,
                    "summary": summary
                }
            else:
                logger.error("Validation failed for processed data")
                return {
                    "success": False,
                    "error": "Validation failed",
                    "processor_type": processor.__class__.__name__,
                    "detected_type": detected_type,
                    "confidence": confidence
                }
        else:
            logger.error("No data extracted from table")
            return {
                "success": False,
                "error": "No data extracted",
                "processor_type": processor.__class__.__name__,
                "detected_type": detected_type,
                "confidence": confidence
            }

    except Exception as e:
        logger.error(f"Error processing table: {e}")
        return {
            "success": False,
            "error": str(e),
            "detected_type": "unknown",
            "confidence": 0.0
        }

def process_all_tables_hybrid(tables: List[pd.DataFrame], report_date: str = None) -> List[Dict[str, Any]]:
    """
    Process all tables using hybrid approach: logical processors first, LLM fallback.
    """
    logger.info(f"Starting hybrid table processing for {len(tables)} tables...")
    
    # Reset used table types at the beginning
    reset_used_table_types()
    
    if not report_date:
        report_date = datetime.now().strftime("%Y-%m-%d")
    
    results = []
    for i, table in enumerate(tables):
        logger.info(f"Processing table {i+1}/{len(tables)}...")
        
        # Add delay between LLM calls to avoid overwhelming the service
        if i > 0:
            time.sleep(1)
        
        result = process_table_hybrid(table, report_date)
        results.append(result)
        
        # Log progress
        if result['error']:
            logger.warning(f"Table {i+1}: {result['error']}")
        else:
            processing_method = result.get('processing_method', 'unknown')
            logger.info(f"Table {i+1}: Successfully processed as {result['table_type']} using {processing_method}")
    
    return results

def save_hybrid_results(results: List[Dict[str, Any]], output_dir: str, extraction_method: str):
    """
    Save hybrid processing results to JSON files.
    """
    if not results:
        logger.warning("No processing results to save")
        return
    
    # Create processed results directory
    processed_output_dir = os.path.join(output_dir, "processed_tables")
    Path(processed_output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving hybrid processing results to {processed_output_dir}")
    
    # Track processing methods
    logical_count = 0
    llm_fallback_count = 0
    hybrid_error_count = 0
    
    # Save individual table results
    for result in results:
        if result['structured_data'] is not None:
            table_index = result['table_index']
            table_type = result['table_type']
            processing_method = result.get('processing_method', 'unknown')
            
            # Track processing method
            if processing_method == 'logical':
                logical_count += 1
            elif 'llm_fallback' in processing_method:
                llm_fallback_count += 1
            elif processing_method == 'hybrid_error':
                hybrid_error_count += 1
            
            # Create filename
            filename = f"{extraction_method}_table_{table_index}_{table_type}_{processing_method}.json"
            filepath = os.path.join(processed_output_dir, filename)
            
            # Save structured data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {processing_method} result for table {table_index} ({table_type}) to {filepath}")
    
    # Save summary report
    summary = {
        'total_tables': len(results),
        'successful_tables': sum(1 for r in results if r['structured_data'] is not None),
        'failed_tables': sum(1 for r in results if r['structured_data'] is None),
        'logical_processed': logical_count,
        'llm_fallback_processed': llm_fallback_count,
        'hybrid_errors': hybrid_error_count,
        'table_types': {},
        'errors': []
    }
    
    # Count table types
    for result in results:
        table_type = result['table_type']
        if table_type not in summary['table_types']:
            summary['table_types'][table_type] = 0
        summary['table_types'][table_type] += 1
    
    # Collect errors
    for result in results:
        if result['error']:
            summary['errors'].append({
                'table_index': result['table_index'],
                'table_type': result['table_type'],
                'error': result['error']
            })
    
    # Save summary
    summary_filepath = os.path.join(processed_output_dir, f"{extraction_method}_hybrid_summary.json")
    with open(summary_filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved hybrid processing summary to {summary_filepath}")
    logger.info(f"Summary: {summary['logical_processed']} logical, {summary['llm_fallback_processed']} LLM fallback, {summary['hybrid_errors']} errors")

def main():
    """
    Main function implementing the hybrid workflow.
    """
    # Configure logging
    logger.add("logs/hybrid_extractor.log", rotation="1 day", retention="7 days")
    
    # Get PDF file path
    if len(sys.argv) < 2:
        print("Usage: python main_extractor_refactored.py <pdf_file_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = "data/processed"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting hybrid extraction workflow for {pdf_path}")
    
    try:
        # Step 1: Extract raw tables
        logger.info("Step 1: Extracting raw tables from PDF...")
        raw_tables = extract_tables_standard(pdf_path)
        
        if not raw_tables:
            logger.error("No tables extracted from PDF")
            sys.exit(1)
        
        logger.info(f"Extracted {len(raw_tables)} raw tables")
        
        # Step 2: Save raw tables to CSV
        logger.info("Step 2: Saving raw tables to CSV...")
        raw_tables_dir = os.path.join(output_dir, "raw_tables")
        Path(raw_tables_dir).mkdir(parents=True, exist_ok=True)
        
        for i, table in enumerate(raw_tables):
            if not table.empty:
                csv_path = os.path.join(raw_tables_dir, f"table_{i+1:02d}_page_{i+1:02d}.csv")
                table.to_csv(csv_path, index=False, header=True)
                logger.info(f"Saved raw table {i+1} to {csv_path}")
        
        # Step 3: Process tables with hybrid approach
        logger.info("Step 3: Processing tables with hybrid approach...")
        report_date = datetime.now().strftime("%Y-%m-%d")
        results = process_all_tables_hybrid(raw_tables, report_date)
        
        # Step 4: Save results
        logger.info("Step 4: Saving hybrid processing results...")
        save_hybrid_results(results, output_dir, "hybrid")
        
        # Final summary
        successful = sum(1 for r in results if r['structured_data'] is not None)
        logger.info(f"✅ Hybrid workflow completed successfully!")
        logger.info(f"   - Raw tables extracted: {len(raw_tables)}")
        logger.info(f"   - Tables processed: {len(results)}")
        logger.info(f"   - Successful: {successful}")
        logger.info(f"   - Failed: {len(results) - successful}")
        
    except Exception as e:
        logger.error(f"Error in hybrid workflow: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 