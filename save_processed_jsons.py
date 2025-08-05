#!/usr/bin/env python3
"""
Save Processed JSONs

This script processes all tables and saves the JSON output for each table
to the data/processed/jsons/ directory.
"""

import os
import pandas as pd
import json
from typing import List, Dict, Any
from loguru import logger
import sys
from datetime import datetime

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

def save_processed_jsons():
    """Process all tables and save JSONs to the specified directory."""
    logger.info("🔍 Processing Tables and Saving JSONs")
    logger.info("=" * 80)

    # Create output directory
    output_dir = "data/processed/jsons"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Load all tables
    tables = load_all_tables()
    if not tables:
        logger.error("No tables loaded!")
        return

    # Classify all tables
    classifications = classify_all_tables(tables)

    # Process each table and save results
    successful_tables = 0
    failed_tables = 0
    
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
            failed_tables += 1
            continue

        # Process the table
        try:
            result = process_table_hybrid(table_df, report_date="2025-04-19", detected_type=detected_type)

            if result["success"]:
                logger.info(f"✅ Successfully processed {result['records_count']} records")
                logger.info(f"🔧 Processor: {result['processor_type']}")
                
                # Create output filename
                output_filename = f"{table_name}_{detected_type}.json"
                output_path = os.path.join(output_dir, output_filename)
                
                # Prepare the complete result data
                output_data = {
                    "table_info": {
                        "table_number": table_idx + 1,
                        "table_name": table_name,
                        "detected_type": detected_type,
                        "confidence": confidence,
                        "raw_shape": list(table_df.shape),
                        "processor_type": result['processor_type'],
                        "records_count": result['records_count'],
                        "processing_timestamp": datetime.now().isoformat()
                    },
                    "data": result['data'],
                    "summary": result.get('summary', {}),
                    "metadata": {
                        "raw_cells": len(table_df) * len(table_df.columns),
                        "non_null_raw_cells": table_df.notna().sum().sum(),
                        "processed_records": len(result['data']),
                        "non_null_processed_values": sum(1 for record in result['data'] for value in record.values() if value is not None and value != ""),
                        "data_coverage_percentage": round((sum(1 for record in result['data'] for value in record.values() if value is not None and value != "") / table_df.notna().sum().sum()) * 100, 1) if table_df.notna().sum().sum() > 0 else 0
                    }
                }
                
                # Save to JSON file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"💾 Saved JSON to: {output_path}")
                logger.info(f"📊 Data coverage: {output_data['metadata']['data_coverage_percentage']}%")
                
                successful_tables += 1
                
            else:
                logger.error(f"❌ Processing failed: {result['error']}")
                failed_tables += 1
                
                # Save error information
                error_filename = f"{table_name}_{detected_type}_error.json"
                error_path = os.path.join(output_dir, error_filename)
                
                error_data = {
                    "table_info": {
                        "table_number": table_idx + 1,
                        "table_name": table_name,
                        "detected_type": detected_type,
                        "confidence": confidence,
                        "raw_shape": list(table_df.shape),
                        "processing_timestamp": datetime.now().isoformat()
                    },
                    "error": result['error'],
                    "status": "failed"
                }
                
                with open(error_path, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, indent=2, ensure_ascii=False, default=str)
                
                logger.info(f"💾 Saved error info to: {error_path}")
                
        except Exception as e:
            logger.error(f"❌ Exception during processing: {e}")
            failed_tables += 1
            
            # Save exception information
            error_filename = f"{table_name}_{detected_type}_exception.json"
            error_path = os.path.join(output_dir, error_filename)
            
            error_data = {
                "table_info": {
                    "table_number": table_idx + 1,
                    "table_name": table_name,
                    "detected_type": detected_type,
                    "confidence": confidence,
                    "raw_shape": list(table_df.shape),
                    "processing_timestamp": datetime.now().isoformat()
                },
                "exception": str(e),
                "status": "exception"
            }
            
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Saved exception info to: {error_path}")

        logger.info(f"\n{'_'*80}")

    # Summary
    logger.info(f"\n📈 PROCESSING SUMMARY:")
    logger.info("=" * 80)
    logger.info(f"Total tables: {len(tables)}")
    logger.info(f"Successfully processed: {successful_tables}")
    logger.info(f"Failed/Not processed: {failed_tables}")
    logger.info(f"Success rate: {(successful_tables/len(tables))*100:.1f}%")
    logger.info(f"JSONs saved to: {output_dir}")
    
    # List all saved files
    logger.info(f"\n📁 SAVED FILES:")
    logger.info("-" * 40)
    saved_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
    saved_files.sort()
    for file in saved_files:
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path)
        logger.info(f"  {file} ({file_size:,} bytes)")

if __name__ == "__main__":
    save_processed_jsons() 