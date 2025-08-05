#!/usr/bin/env python3
"""
Robust Regional Summary Table Extraction
This script handles structural inconsistencies in PDF tables.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

from loguru import logger
import tabula

# --- Configuration ---
INPUT_PDF_PATH = "data/raw/19.04.25_NLDC_PSP.pdf"
OUTPUT_DIR = "data/robust_regional_summary"

# --- Logging ---
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO")

def extract_regional_summary_table(pdf_path: str) -> pd.DataFrame:
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
                return table
        
        # If not found with strict criteria, try to find any table with regional data
        for i, table in enumerate(valid_tables):
            content = " ".join(map(str, table.columns)).lower() + " " + " ".join(map(str, table.head(3).values.flatten())).lower()
            
            # Look for regional indicators
            if any(region in content for region in ['nr', 'wr', 'sr', 'er', 'ner']) and 'region' in content:
                logger.info(f"Found potential regional summary table {i+1}: Shape {table.shape}")
                logger.info(f"Columns: {list(table.columns)}")
                return table
        
        raise ValueError("Regional summary table not found!")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise

def detect_row_structure(row: pd.Series) -> Tuple[str, str, int]:
    """
    Detect the structure of a row and determine:
    - region_code: The region this row belongs to
    - state_name: The state name
    - data_start_col: The column index where data starts (to handle shifted data)
    """
    # Clean the row data
    row_clean = [str(cell).strip() if pd.notna(cell) else "" for cell in row]
    
    # Check if this is a region header row
    if row_clean[0] in ['NR', 'WR', 'SR', 'ER', 'NER']:
        region_code = row_clean[0]
        state_name = row_clean[1] if row_clean[1] else ""
        data_start_col = 2  # Data starts from column 2
        return region_code, state_name, data_start_col
    
    # Check if this is a state row with region in first column
    elif row_clean[0] in ['NR', 'WR', 'SR', 'ER', 'NER'] and row_clean[1]:
        region_code = row_clean[0]
        state_name = row_clean[1]
        data_start_col = 2
        return region_code, state_name, data_start_col
    
    # Check if this is a state row with empty first column (under a region)
    elif not row_clean[0] and row_clean[1]:
        # This is a state row under the current region
        # We need to determine the region from context
        region_code = "UNKNOWN"  # Will be set by caller
        state_name = row_clean[1]
        data_start_col = 2
        return region_code, state_name, data_start_col
    
    # Check for shifted data (when state name is in column 0 but no region)
    elif row_clean[0] and row_clean[0] not in ['NR', 'WR', 'SR', 'ER', 'NER']:
        # This might be a state row with shifted data
        region_code = "UNKNOWN"  # Will be set by caller
        state_name = row_clean[0]
        data_start_col = 1  # Data starts from column 1 (shifted)
        return region_code, state_name, data_start_col
    
    # Default case
    else:
        region_code = "UNKNOWN"
        state_name = ""
        data_start_col = 0
        return region_code, state_name, data_start_col

def extract_numeric_value(cell_value: Any) -> float:
    """Safely extract numeric value from a cell."""
    if pd.isna(cell_value):
        return None
    
    try:
        # Convert to string and clean
        cell_str = str(cell_value).strip()
        if not cell_str or cell_str == 'nan':
            return None
        
        # Try to convert to float
        return float(cell_str)
    except (ValueError, TypeError):
        return None

def process_regional_summary_robustly(table_df: pd.DataFrame, report_date: str) -> List[Dict]:
    """Robustly process the regional summary table structure handling inconsistencies."""
    logger.info("Processing regional summary table robustly...")
    
    # Clean column names
    table_df.columns = [col.replace('\r', ' ').replace('\n', ' ') for col in table_df.columns]
    
    # Initialize result list
    results = []
    current_region = None
    
    for index, row in table_df.iterrows():
        # Skip completely empty rows
        if row.isna().all():
            continue
        
        # Detect row structure
        region_code, state_name, data_start_col = detect_row_structure(row)
        
        # Update current region if we found a new one
        if region_code != "UNKNOWN":
            current_region = region_code
        elif current_region:
            region_code = current_region
        else:
            # Skip rows where we can't determine the region
            continue
        
        # Skip if no valid state name
        if not state_name or state_name == 'nan' or state_name == 'States':
            continue
        
        # Extract values based on detected structure
        try:
            # Calculate column indices based on data_start_col
            col_indices = {
                'power_max_demand_met_day_mw': data_start_col,
                'power_shortage_at_max_demand_mw': data_start_col + 1,
                'energy_met_mu': data_start_col + 2,
                'energy_drawal_schedule_mu': data_start_col + 3,
                'energy_over_under_drawal_mu': data_start_col + 4,
                'power_max_overdrawal_mw': data_start_col + 5,
                'energy_shortage_mu': data_start_col + 6
            }
            
            # Extract values
            record = {
                "region_code": region_code,
                "state_name": state_name,
                "power_max_demand_met_day_mw": extract_numeric_value(row.iloc[col_indices['power_max_demand_met_day_mw']]) if col_indices['power_max_demand_met_day_mw'] < len(row) else None,
                "power_shortage_at_max_demand_mw": extract_numeric_value(row.iloc[col_indices['power_shortage_at_max_demand_mw']]) if col_indices['power_shortage_at_max_demand_mw'] < len(row) else None,
                "energy_met_mu": extract_numeric_value(row.iloc[col_indices['energy_met_mu']]) if col_indices['energy_met_mu'] < len(row) else None,
                "energy_drawal_schedule_mu": extract_numeric_value(row.iloc[col_indices['energy_drawal_schedule_mu']]) if col_indices['energy_drawal_schedule_mu'] < len(row) else None,
                "energy_over_under_drawal_mu": extract_numeric_value(row.iloc[col_indices['energy_over_under_drawal_mu']]) if col_indices['energy_over_under_drawal_mu'] < len(row) else None,
                "power_max_overdrawal_mw": extract_numeric_value(row.iloc[col_indices['power_max_overdrawal_mw']]) if col_indices['power_max_overdrawal_mw'] < len(row) else None,
                "energy_shortage_mu": extract_numeric_value(row.iloc[col_indices['energy_shortage_mu']]) if col_indices['energy_shortage_mu'] < len(row) else None,
                "report_date": report_date
            }
            
            # Log the extraction for debugging
            logger.debug(f"Row {index}: {state_name} in {region_code}, data_start_col={data_start_col}")
            logger.debug(f"  Raw row: {list(row)}")
            logger.debug(f"  Extracted: {record}")
            
            results.append(record)
            
        except (ValueError, TypeError, IndexError) as e:
            logger.warning(f"Failed to process row {index}: {e}")
            logger.warning(f"Row data: {list(row)}")
            continue
    
    logger.info(f"Successfully processed {len(results)} records")
    return results

def main():
    """Main function to test robust regional summary extraction."""
    logger.info("="*60)
    logger.info("  ROBUST REGIONAL SUMMARY TABLE EXTRACTION")
    logger.info("="*60)
    
    try:
        # Extract regional summary table
        regional_table = extract_regional_summary_table(INPUT_PDF_PATH)
        
        report_date = "2025-04-19"  # Extract from filename
        
        logger.info(f"\n--- Processing Regional Summary Table ---")
        logger.info(f"Table shape: {regional_table.shape}")
        logger.info(f"Table columns: {list(regional_table.columns)}")
        
        # Save raw table for inspection
        raw_table_path = Path(OUTPUT_DIR) / "raw_regional_summary.csv"
        regional_table.to_csv(raw_table_path, index=False)
        logger.info(f"Saved raw table to: {raw_table_path}")
        
        # Process robustly
        results = process_regional_summary_robustly(regional_table, report_date)
        
        if results:
            # Save structured result
            output_path = Path(OUTPUT_DIR) / "structured_regional_summary.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved structured data to: {output_path}")
            
            # Display first few records
            logger.info("First few structured records:")
            for j, record in enumerate(results[:5]):
                logger.info(f"Record {j+1}: {record}")
            
            # Display summary statistics
            logger.info(f"\n📊 Summary:")
            logger.info(f"Total records extracted: {len(results)}")
            
            # Count by region
            region_counts = {}
            for record in results:
                region = record.get("region_code", "Unknown")
                region_counts[region] = region_counts.get(region, 0) + 1
            
            logger.info("Records by region:")
            for region, count in region_counts.items():
                logger.info(f"  {region}: {count} records")
            
            # Show some problematic records for verification
            logger.info("\n🔍 Sample records for verification:")
            er_records = [r for r in results if r['region_code'] == 'ER'][:3]
            ner_records = [r for r in results if r['region_code'] == 'NER'][:3]
            
            logger.info("ER records:")
            for record in er_records:
                logger.info(f"  {record['state_name']}: Max Demand={record['power_max_demand_met_day_mw']}, Energy Met={record['energy_met_mu']}")
            
            logger.info("NER records:")
            for record in ner_records:
                logger.info(f"  {record['state_name']}: Max Demand={record['power_max_demand_met_day_mw']}, Energy Met={record['energy_met_mu']}")
                
        else:
            logger.error("Failed to process regional summary table")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("  EXTRACTION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main() 