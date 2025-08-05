#!/usr/bin/env python3
"""
Display Processed JSONs for Each Table - Concise Version

This script processes all tables and displays a concise summary of the JSON output
for each, along with analysis of data completeness.
"""

import os
import pandas as pd
import json
from typing import List, Dict, Any
from loguru import logger
import sys

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.processors import classify_all_tables, TABLE_STRUCTURE_PATTERNS
from main_extractor_refactored import process_table_hybrid

def load_all_tables() -> List[pd.DataFrame]:
    """Load all CSV files from the raw_tables directory."""
    tables = []
    csv_dir = "data/processed/raw_tables"

    if not os.path.exists(csv_dir):
        logger.error(f"Directory {csv_dir} not found!")
        return tables

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    csv_files.sort()  # Sort to ensure consistent order

    logger.info(f"Found {len(csv_files)} CSV files")

    for csv_file in csv_files:
        file_path = os.path.join(csv_dir, csv_file)
        try:
            df = pd.read_csv(file_path)
            tables.append(df)
            logger.info(f"  - Loaded {csv_file}: {df.shape[0]} rows, {df.shape[1]} cols")
        except Exception as e:
            logger.error(f"Failed to load {csv_file}: {e}")

    return tables

def analyze_data_completeness(raw_df: pd.DataFrame, processed_data: List[Dict], table_type: str) -> Dict[str, Any]:
    """Analyze whether all values from raw table are captured in processed JSON."""
    analysis = {
        "raw_rows": len(raw_df),
        "raw_cols": len(raw_df.columns),
        "raw_cells": len(raw_df) * len(raw_df.columns),
        "processed_records": len(processed_data),
        "missing_data": [],
        "data_coverage": 0.0
    }
    
    # Count non-null values in raw table
    non_null_raw = raw_df.notna().sum().sum()
    analysis["non_null_raw_cells"] = non_null_raw
    
    # Count non-null values in processed data
    non_null_processed = 0
    for record in processed_data:
        for value in record.values():
            if value is not None and value != "":
                non_null_processed += 1
    
    analysis["non_null_processed_values"] = non_null_processed
    
    # Calculate coverage
    if non_null_raw > 0:
        analysis["data_coverage"] = (non_null_processed / non_null_raw) * 100
    
    # Table-specific analysis
    if table_type == "regional_summary":
        expected_regions = ["NR", "WR", "SR", "ER", "NER", "TOTAL"]
        found_regions = set(record.get("region_code") for record in processed_data)
        missing_regions = set(expected_regions) - found_regions
        if missing_regions:
            analysis["missing_data"].append(f"Missing regions: {missing_regions}")
    
    elif table_type == "state_summary":
        # Check if all states are captured
        raw_states = set()
        for idx, row in raw_df.iterrows():
            if not pd.isna(row.iloc[0]):
                raw_states.add(str(row.iloc[0]).strip())
        
        processed_states = set(record.get("state_name") for record in processed_data)
        missing_states = raw_states - processed_states
        if missing_states:
            analysis["missing_data"].append(f"Missing states: {list(missing_states)[:5]}...")  # Show first 5
    
    elif table_type == "generation_by_source":
        expected_sources = ["Coal", "Lignite", "Hydro", "Nuclear", "Gas/Naptha/Diesel", "RES (Wind+Solar+Biomass)", "Total"]
        found_sources = set(record.get("source_name") for record in processed_data)
        missing_sources = set(expected_sources) - found_sources
        if missing_sources:
            analysis["missing_data"].append(f"Missing sources: {missing_sources}")
    
    elif table_type == "generation_outages":
        expected_sectors = ["Central Sector", "State Sector", "Total"]
        found_sectors = set(record.get("sector_name") for record in processed_data)
        missing_sectors = set(expected_sectors) - found_sectors
        if missing_sectors:
            analysis["missing_data"].append(f"Missing sectors: {missing_sectors}")
    
    elif table_type == "share":
        # Check if both measures are captured
        expected_measures = [
            "Share of RES in total generation (%)",
            "Share of Non-fossil fuel (Hydro,Nuclear and RES) in total generation(%)"
        ]
        found_measures = set(record.get("measure") for record in processed_data)
        missing_measures = set(expected_measures) - found_measures
        if missing_measures:
            analysis["missing_data"].append(f"Missing measures: {missing_measures}")
    
    return analysis

def display_sample_records(data: List[Dict], max_samples: int = 3) -> str:
    """Display sample records from the processed data."""
    if not data:
        return "No data"
    
    samples = data[:max_samples]
    sample_str = json.dumps(samples, indent=2, ensure_ascii=False)
    
    if len(data) > max_samples:
        sample_str += f"\n... and {len(data) - max_samples} more records"
    
    return sample_str

def display_processed_jsons():
    """Display processed JSONs for each table with completeness analysis."""
    logger.info("🔍 Displaying Processed JSONs for Each Table - CONCISE VERSION")
    logger.info("=" * 80)

    # Load all tables
    tables = load_all_tables()
    if not tables:
        logger.error("No tables loaded!")
        return

    # Classify all tables
    classifications = classify_all_tables(tables)

    # Process each table and display results
    for table_idx, detected_type, confidence in classifications:
        table_name = f"table_{table_idx+1:02d}"
        table_df = tables[table_idx]
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📋 TABLE {table_idx+1}: {table_name}")
        logger.info(f"📊 Detected Type: {detected_type} (confidence: {confidence:.2f})")
        logger.info(f"📏 Raw Table Shape: {table_df.shape}")
        logger.info(f"{'='*80}")

        if detected_type == "unknown":
            logger.warning("❌ Garbage table - no processing attempted")
            continue

        # Process the table
        try:
            result = process_table_hybrid(table_df, report_date="2025-04-19", detected_type=detected_type)

            if result["success"]:
                logger.info(f"✅ Successfully processed {result['records_count']} records")
                logger.info(f"🔧 Processor: {result['processor_type']}")
                
                # Display sample processed JSON
                logger.info(f"\n📄 SAMPLE PROCESSED JSON (first 3 records):")
                logger.info("-" * 40)
                
                sample_json = display_sample_records(result['data'], max_samples=3)
                print(sample_json)
                
                # Analyze data completeness
                analysis = analyze_data_completeness(table_df, result['data'], detected_type)
                
                logger.info(f"\n📊 DATA COMPLETENESS ANALYSIS:")
                logger.info("-" * 40)
                logger.info(f"Raw table cells: {analysis['raw_cells']}")
                logger.info(f"Non-null raw cells: {analysis['non_null_raw_cells']}")
                logger.info(f"Processed records: {analysis['processed_records']}")
                logger.info(f"Non-null processed values: {analysis['non_null_processed_values']}")
                logger.info(f"Data coverage: {analysis['data_coverage']:.1f}%")
                
                if analysis['missing_data']:
                    logger.warning("⚠️  MISSING DATA:")
                    for missing in analysis['missing_data']:
                        logger.warning(f"   - {missing}")
                else:
                    logger.info("✅ All expected data appears to be captured")
                
                # Display summary if available
                if result.get('summary'):
                    logger.info(f"\n📈 SUMMARY:")
                    logger.info("-" * 40)
                    summary_str = json.dumps(result['summary'], indent=2, ensure_ascii=False)
                    print(summary_str)
                
            else:
                logger.error(f"❌ Processing failed: {result['error']}")
                
        except Exception as e:
            logger.error(f"❌ Exception during processing: {e}")

        logger.info(f"\n{'_'*80}")

if __name__ == "__main__":
    display_processed_jsons() 