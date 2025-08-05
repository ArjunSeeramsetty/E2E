#!/usr/bin/env python3
"""
Manual Regional Summary Table Extraction
This script directly processes the regional summary table structure.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from loguru import logger
import tabula

# --- Configuration ---
INPUT_PDF_PATH = "data/raw/19.04.25_NLDC_PSP.pdf"
OUTPUT_DIR = "data/manual_regional_summary"

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

def process_regional_summary_manually(table_df: pd.DataFrame, report_date: str) -> List[Dict]:
    """Manually process the regional summary table structure."""
    logger.info("Processing regional summary table manually...")
    
    # Clean column names
    table_df.columns = [col.replace('\r', ' ').replace('\n', ' ') for col in table_df.columns]
    
    # Initialize result list
    results = []
    current_region = None
    
    for index, row in table_df.iterrows():
        # Skip header rows
        if pd.isna(row.iloc[0]) and pd.isna(row.iloc[1]):
            continue
            
        # Check if this is a region row (has region code)
        if pd.notna(row.iloc[0]) and str(row.iloc[0]).strip() in ['NR', 'WR', 'SR', 'ER', 'NER']:
            current_region = str(row.iloc[0]).strip()
            state_name = str(row.iloc[1]).strip()
        else:
            # This is a state row under the current region
            state_name = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else str(row.iloc[1]).strip()
        
        # Skip if no valid state name
        if not state_name or state_name == 'nan' or state_name == 'States':
            continue
            
        # Extract values
        try:
            record = {
                "region_code": current_region,
                "state_name": state_name,
                "power_max_demand_met_day_mw": float(row.iloc[2]) if pd.notna(row.iloc[2]) else None,
                "power_shortage_at_max_demand_mw": float(row.iloc[3]) if pd.notna(row.iloc[3]) else None,
                "energy_met_mu": float(row.iloc[4]) if pd.notna(row.iloc[4]) else None,
                "energy_drawal_schedule_mu": float(row.iloc[5]) if pd.notna(row.iloc[5]) else None,
                "energy_over_under_drawal_mu": float(row.iloc[6]) if pd.notna(row.iloc[6]) else None,
                "power_max_overdrawal_mw": float(row.iloc[7]) if pd.notna(row.iloc[7]) else None,
                "energy_shortage_mu": float(row.iloc[8]) if pd.notna(row.iloc[8]) else None,
                "report_date": report_date
            }
            results.append(record)
            logger.debug(f"Processed: {state_name} in {current_region}")
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to process row {index}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(results)} records")
    return results

def main():
    """Main function to test manual regional summary extraction."""
    logger.info("="*60)
    logger.info("  MANUAL REGIONAL SUMMARY TABLE EXTRACTION")
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
        
        # Process manually
        results = process_regional_summary_manually(regional_table, report_date)
        
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
                
        else:
            logger.error("Failed to process regional summary table")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("  EXTRACTION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main() 