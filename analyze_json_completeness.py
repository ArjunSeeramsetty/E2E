#!/usr/bin/env python3
"""
Analyze JSON Completeness from Output

This script analyzes the processed JSONs output to provide a summary
of data completeness for each table.
"""

import re

def analyze_output_file(filename):
    """Analyze the output file to extract completeness information."""
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filename, 'r', encoding='latin-1') as f:
            content = f.read()
    
    print("📊 JSON COMPLETENESS ANALYSIS SUMMARY")
    print("=" * 80)
    
    # Extract table information using simpler patterns
    table_sections = content.split("TABLE ")
    
    total_tables = 0
    successful_tables = 0
    coverages = []
    
    for section in table_sections[1:]:  # Skip first empty section
        lines = section.split('\n')
        if len(lines) < 3:
            continue
            
        table_num = lines[0].split(':')[0]
        
        # Extract detected type
        detected_type = "unknown"
        confidence = "0.0"
        for line in lines:
            if "Detected Type:" in line:
                match = re.search(r"Detected Type: (\w+) \(confidence: ([\d.]+)\)", line)
                if match:
                    detected_type = match.group(1)
                    confidence = match.group(2)
                break
        
        # Extract shape
        shape = "0x0"
        for line in lines:
            if "Raw Table Shape:" in line:
                match = re.search(r"Raw Table Shape: \((\d+), (\d+)\)", line)
                if match:
                    shape = f"{match.group(1)}x{match.group(2)}"
                break
        
        # Extract completeness info
        raw_cells = "0"
        non_null_raw = "0"
        processed_records = "0"
        non_null_processed = "0"
        coverage = "0.0"
        
        for line in lines:
            if "Raw table cells:" in line:
                match = re.search(r"Raw table cells: (\d+)", line)
                if match:
                    raw_cells = match.group(1)
            elif "Non-null raw cells:" in line:
                match = re.search(r"Non-null raw cells: (\d+)", line)
                if match:
                    non_null_raw = match.group(1)
            elif "Processed records:" in line:
                match = re.search(r"Processed records: (\d+)", line)
                if match:
                    processed_records = match.group(1)
            elif "Non-null processed values:" in line:
                match = re.search(r"Non-null processed values: (\d+)", line)
                if match:
                    non_null_processed = match.group(1)
            elif "Data coverage:" in line:
                match = re.search(r"Data coverage: ([\d.]+)%", line)
                if match:
                    coverage = match.group(1)
        
        # Check for missing data
        missing_data = []
        for line in lines:
            if "Missing" in line and ":" in line:
                missing_data.append(line.strip())
        
        # Print table summary
        print(f"\n📋 TABLE {table_num}: {detected_type.upper()}")
        print(f"   Raw Shape: {shape}")
        print(f"   Confidence: {float(confidence):.2f}")
        print(f"   📊 COMPLETENESS:")
        print(f"      Raw table cells: {raw_cells}")
        print(f"      Non-null raw cells: {non_null_raw}")
        print(f"      Processed records: {processed_records}")
        print(f"      Non-null processed values: {non_null_processed}")
        print(f"      Data coverage: {coverage}%")
        
        if missing_data:
            print(f"      ⚠️  MISSING DATA:")
            for missing in missing_data:
                print(f"         {missing}")
        else:
            print(f"      ✅ All expected data captured")
        
        print("-" * 60)
        
        total_tables += 1
        if float(coverage) >= 100:
            successful_tables += 1
        coverages.append(float(coverage))
    
    # Summary statistics
    print(f"\n📈 OVERALL SUMMARY:")
    print("=" * 80)
    
    failed_tables = total_tables - successful_tables
    
    print(f"Total tables processed: {total_tables}")
    print(f"Successfully processed: {successful_tables}")
    print(f"Failed/Incomplete: {failed_tables}")
    if total_tables > 0:
        print(f"Success rate: {(successful_tables/total_tables)*100:.1f}%")
    
    if coverages:
        avg_coverage = sum(coverages) / len(coverages)
        print(f"Average data coverage: {avg_coverage:.1f}%")

if __name__ == "__main__":
    analyze_output_file("processed_jsons_output.txt") 