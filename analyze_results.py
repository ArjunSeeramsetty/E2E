#!/usr/bin/env python3
"""
Results Analysis Script

This script analyzes the test results and provides detailed insights about
table processing success/failure patterns.
"""

import sys
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append('src')

from processors import get_processor
from main_extractor_refactored import process_table_hybrid

def analyze_table_processing():
    """Analyze all tables and provide detailed insights."""
    print("🔍 Analyzing Table Processing Results")
    print("=" * 60)
    
    # Load tables
    raw_tables_dir = Path('data/processed/raw_tables')
    csv_files = sorted(raw_tables_dir.glob('*.csv'))
    
    results = []
    
    for csv_file in csv_files:
        table_name = csv_file.stem
        df = pd.read_csv(csv_file)
        
        # Get processor
        processor = get_processor(df)
        processor_name = processor.__class__.__name__ if processor else "None"
        table_type = processor.TABLE_TYPE if processor else "unknown"
        
        # Test processing
        try:
            report_date = "2025-04-19"  # Default date for testing
            result = process_table_hybrid(df, table_name, report_date)
            success = result["success"]
            method = result.get("method", "unknown")
            record_count = len(result.get("data", []))
            error = result.get("error", None)
        except Exception as e:
            success = False
            method = "exception"
            record_count = 0
            error = str(e)
        
        results.append({
            "table_name": table_name,
            "shape": f"{df.shape[0]}x{df.shape[1]}",
            "processor": processor_name,
            "table_type": table_type,
            "success": success,
            "method": method,
            "record_count": record_count,
            "error": error
        })
    
    # Display results
    print(f"\n📊 Processing Results Summary:")
    print(f"{'='*60}")
    
    # Group by table type
    type_stats = {}
    for result in results:
        table_type = result["table_type"]
        if table_type not in type_stats:
            type_stats[table_type] = {"count": 0, "success": 0, "failed": 0}
        
        type_stats[table_type]["count"] += 1
        if result["success"]:
            type_stats[table_type]["success"] += 1
        else:
            type_stats[table_type]["failed"] += 1
    
    print(f"\n📈 Table Type Statistics:")
    for table_type, stats in type_stats.items():
        success_rate = (stats["success"] / stats["count"]) * 100 if stats["count"] > 0 else 0
        print(f"  {table_type}: {stats['success']}/{stats['count']} ({success_rate:.1f}%)")
    
    # Show detailed results
    print(f"\n📋 Detailed Results:")
    print(f"{'='*80}")
    print(f"{'Table':<20} {'Type':<25} {'Processor':<30} {'Status':<10} {'Method':<15} {'Records':<8}")
    print(f"{'-'*80}")
    
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{result['table_name']:<20} {result['table_type']:<25} {result['processor']:<30} {status:<10} {result['method']:<15} {result['record_count']:<8}")
    
    # Analyze failures
    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"\n❌ Failed Tables Analysis:")
        print(f"{'='*40}")
        for failure in failures:
            print(f"  {failure['table_name']}: {failure['error']}")
    
    # Analyze successes
    successes = [r for r in results if r["success"]]
    if successes:
        print(f"\n✅ Successful Tables Analysis:")
        print(f"{'='*40}")
        
        # Group by method
        method_stats = {}
        for success in successes:
            method = success["method"]
            if method not in method_stats:
                method_stats[method] = {"count": 0, "total_records": 0}
            method_stats[method]["count"] += 1
            method_stats[method]["total_records"] += success["record_count"]
        
        for method, stats in method_stats.items():
            avg_records = stats["total_records"] / stats["count"] if stats["count"] > 0 else 0
            print(f"  {method}: {stats['count']} tables, {stats['total_records']} total records ({avg_records:.1f} avg)")
    
    # Check for unexpected patterns
    print(f"\n🔍 Pattern Analysis:")
    print(f"{'='*30}")
    
    # Check for multiple tables of same type
    type_counts = {}
    for result in results:
        table_type = result["table_type"]
        type_counts[table_type] = type_counts.get(table_type, 0) + 1
    
    unexpected_types = []
    for table_type, count in type_counts.items():
        if count > 1 and table_type != "transnational_exchange":  # transnational_exchange handles 3 tables
            unexpected_types.append((table_type, count))
    
    if unexpected_types:
        print(f"  ⚠️  Multiple tables detected as same type:")
        for table_type, count in unexpected_types:
            print(f"    - {table_type}: {count} tables")
    else:
        print(f"  ✅ No unexpected multiple table types detected")
    
    # Check for missing table types
    expected_types = [
        "regional_daily_summary", "state_daily_summary", "generation_by_source",
        "generation_outages", "inter_regional_transmission", "transnational_transmission",
        "transnational_exchange", "transnational_summary", "regional_import_export_summary",
        "share", "solar_non_solar_peak", "frequency_profile", "scada_timeseries", "ddf"
    ]
    
    found_types = set(type_counts.keys())
    missing_types = set(expected_types) - found_types
    unexpected_found = found_types - set(expected_types)
    
    if missing_types:
        print(f"  ⚠️  Missing expected table types: {missing_types}")
    if unexpected_found:
        print(f"  ⚠️  Unexpected table types found: {unexpected_found}")
    if not missing_types and not unexpected_found:
        print(f"  ✅ All expected table types found")

if __name__ == "__main__":
    analyze_table_processing() 