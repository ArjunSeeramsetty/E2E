#!/usr/bin/env python3
"""
Improved Regional Summary Table Extraction
This script specifically targets the actual regional summary table structure.
"""

import os
import sys
import json
import time
import re
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import pandas as pd
from loguru import logger
import tabula

# --- Configuration ---
INPUT_PDF_PATH = "data/raw/19.04.25_NLDC_PSP.pdf"
OUTPUT_DIR = "data/improved_regional_summary"

# --- LLM Configuration ---
LLM_ENDPOINT = "http://localhost:11434/api/generate"
LLM_MODEL = "llama3.2:3b"
LLM_TIMEOUT = 120
LLM_MAX_RETRIES = 3

# --- Logging ---
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO")

def extract_regional_summary_table(pdf_path: str) -> Optional[pd.DataFrame]:
    """Extract the specific regional summary table from the PDF."""
    logger.info(f"Extracting regional summary table from {pdf_path}")
    
    try:
        # Extract all tables
        tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, lattice=True, stream=True, silent=True)
        valid_tables = [df for df in tables if isinstance(df, pd.DataFrame) and not df.empty and df.shape[1] > 1]
        
        # Find the specific regional summary table
        for i, table in enumerate(valid_tables):
            # Check if this is the regional summary table with state-wise data
            content = " ".join(map(str, table.columns)).lower() + " " + " ".join(map(str, table.head(3).values.flatten())).lower()
            
            # Look for specific indicators of the regional summary table - more flexible matching
            indicators = ['max.demand', 'energy met', 'drawal', 'od', 'ud']
            matches = sum(1 for indicator in indicators if indicator in content)
            
            if matches >= 3:  # At least 3 indicators must match
                logger.info(f"Found regional summary table {i+1}: Shape {table.shape}")
                logger.info(f"Columns: {list(table.columns)}")
                logger.info(f"First few rows:\n{table.head()}")
                return table
        
        # If not found with strict criteria, try to find any table with regional data
        for i, table in enumerate(valid_tables):
            content = " ".join(map(str, table.columns)).lower() + " " + " ".join(map(str, table.head(3).values.flatten())).lower()
            
            # Look for regional indicators
            if any(region in content for region in ['nr', 'wr', 'sr', 'er', 'ner']) and 'region' in content:
                logger.info(f"Found potential regional summary table {i+1}: Shape {table.shape}")
                logger.info(f"Columns: {list(table.columns)}")
                logger.info(f"First few rows:\n{table.head()}")
                return table
        
        logger.error("Regional summary table not found!")
        return None
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return None

def call_llm_with_retry(prompt: str) -> Optional[str]:
    """Calls the local LLM with retry mechanism."""
    for attempt in range(LLM_MAX_RETRIES):
        try:
            payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
            response = requests.post(LLM_ENDPOINT, json=payload, timeout=LLM_TIMEOUT)
            response.raise_for_status()
            
            response_data = response.json()
            return response_data.get('response', '')
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt < LLM_MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None

def extract_json_from_response(response: str) -> Optional[str]:
    """Extract JSON from LLM response that might contain explanatory text."""
    # First, try to extract JSON from markdown code blocks
    code_block_pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
    code_block_matches = re.findall(code_block_pattern, response, re.DOTALL)
    if code_block_matches:
        try:
            json.loads(code_block_matches[0])
            return code_block_matches[0]
        except json.JSONDecodeError:
            pass
    
    # Look for JSON array starting with [ and ending with ]
    # This handles cases where the LLM adds explanatory text before/after
    json_array_pattern = r'\[\s*\{.*?\}\s*\]'
    matches = re.findall(json_array_pattern, response, re.DOTALL)
    if matches:
        # Try to parse the longest match
        longest_match = max(matches, key=len)
        try:
            json.loads(longest_match)
            return longest_match
        except json.JSONDecodeError:
            pass
    
    # Fallback: look for any JSON array
    json_patterns = [
        r'\[.*\]',  # JSON array
        r'\{.*\}',  # JSON object
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            # Try to parse the longest match
            longest_match = max(matches, key=len)
            try:
                json.loads(longest_match)
                return longest_match
            except json.JSONDecodeError:
                continue
    
    return None

def process_regional_summary_with_llm(table_df: pd.DataFrame, report_date: str) -> Optional[Dict]:
    """Process regional summary table with specific structure understanding."""
    
    # Convert table to string for LLM processing
    table_str = table_df.to_string(index=False, header=True)
    
    # Enhanced prompt specifically for regional summary table
    prompt = f"""
You are an expert data extraction agent for power supply reports. Your task is to extract ALL values from the regional summary table and convert them into structured JSON.

**Raw Table Data:**
```
{table_str}
```

**Table Structure Analysis:**
This is a regional summary table with the following columns:
- Region: The region name (NR, WR, SR, ER, NER)
- States: Individual states within each region
- Max.Demand Met during the day (MW): Maximum demand met in MW
- Shortage during maximum Demand (MW): Shortage during peak demand
- Energy Met (MU): Energy met in MU
- Drawal Schedule (MU): Scheduled drawal in MU
- OD(+)/UD(-) (MU): Overdrawal/Underdrawal in MU
- Max OD (MW): Maximum overdrawal in MW
- Energy Shortage (MU): Energy shortage in MU

**Instructions:**
1. Extract ALL numerical values from the table, including zeros and small numbers
2. For each state/region combination, create a complete JSON object
3. Use the provided report_date: {report_date}
4. Convert all text numbers to actual numbers
5. If a value is missing or not present, use null
6. Pay special attention to small decimal values and ensure they are captured
7. Map the columns correctly to the target schema

**Required JSON Schema:**
- region_code: (String) Region abbreviation (NR, WR, SR, ER, NER)
- state_name: (String) State name
- power_max_demand_met_day_mw: (Number) Max.Demand Met during the day (MW)
- power_shortage_at_max_demand_mw: (Number) Shortage during maximum Demand (MW)
- energy_met_mu: (Number) Energy Met (MU)
- energy_drawal_schedule_mu: (Number) Drawal Schedule (MU)
- energy_over_under_drawal_mu: (Number) OD(+)/UD(-)(MU)
- power_max_overdrawal_mw: (Number) Max OD (MW)
- energy_shortage_mu: (Number) Energy Shortage (MU)
- report_date: (String) "{report_date}"

**IMPORTANT:** 
- Extract ALL values from the table, including small numbers like 0.1, 0.5, etc.
- Do not skip any data
- Map the columns correctly to the target schema
- Include all states and regions

Output ONLY the JSON array. No explanations or additional text.
"""

    logger.info("Sending enhanced prompt to LLM...")
    llm_response = call_llm_with_retry(prompt)
    
    if not llm_response:
        logger.error("LLM failed to respond")
        return None
    
    # Extract JSON from response
    logger.debug(f"LLM Response: {llm_response[:500]}...")
    json_str = extract_json_from_response(llm_response)
    if not json_str:
        logger.error(f"Could not extract JSON from response: {llm_response[:200]}...")
        return None
    
    try:
        structured_data = json.loads(json_str)
        logger.info(f"Successfully parsed regional summary data with {len(structured_data)} records")
        return {"type": "regional_daily_summary", "data": structured_data}
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        logger.error(f"Response: {llm_response[:300]}...")
        return None

def main():
    """Main function to test improved regional summary extraction."""
    logger.info("="*60)
    logger.info("  IMPROVED REGIONAL SUMMARY TABLE EXTRACTION")
    logger.info("="*60)
    
    # Extract regional summary table
    regional_table = extract_regional_summary_table(INPUT_PDF_PATH)
    
    if regional_table is None:
        logger.error("Regional summary table not found!")
        return
    
    report_date = "2025-04-19"  # Extract from filename
    
    logger.info(f"\n--- Processing Regional Summary Table ---")
    logger.info(f"Table shape: {regional_table.shape}")
    logger.info(f"Table columns: {list(regional_table.columns)}")
    
    # Save raw table for inspection
    raw_table_path = Path(OUTPUT_DIR) / "raw_regional_summary.csv"
    regional_table.to_csv(raw_table_path, index=False)
    logger.info(f"Saved raw table to: {raw_table_path}")
    
    # Process with LLM
    result = process_regional_summary_with_llm(regional_table, report_date)
    
    if result:
        # Save structured result
        output_path = Path(OUTPUT_DIR) / "structured_regional_summary.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result["data"], f, indent=2, ensure_ascii=False)
        logger.info(f"✅ Saved structured data to: {output_path}")
        
        # Display first few records
        logger.info("First few structured records:")
        for j, record in enumerate(result["data"][:5]):
            logger.info(f"Record {j+1}: {record}")
        
        # Display summary statistics
        logger.info(f"\n📊 Summary:")
        logger.info(f"Total records extracted: {len(result['data'])}")
        
        # Count by region
        region_counts = {}
        for record in result["data"]:
            region = record.get("region_code", "Unknown")
            region_counts[region] = region_counts.get(region, 0) + 1
        
        logger.info("Records by region:")
        for region, count in region_counts.items():
            logger.info(f"  {region}: {count} records")
            
    else:
        logger.error("Failed to process regional summary table")
    
    logger.info("\n" + "="*60)
    logger.info("  EXTRACTION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main() 