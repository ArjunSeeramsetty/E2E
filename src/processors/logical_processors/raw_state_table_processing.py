#!/usr/bin/env python3
"""
Raw State Table Processing Module

This module contains the robust processing logic for state-wise power supply position tables.
It handles structural inconsistencies and data shifting issues commonly found in PDF-extracted tables.
"""

import pandas as pd
from typing import List, Dict, Any, Tuple
from loguru import logger


def detect_row_structure_robust(row: pd.Series) -> Tuple[str, str, int]:
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


def extract_numeric_value_robust(cell_value: Any) -> float:
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


def process_state_table_robustly(table_df: pd.DataFrame, report_date: str) -> List[Dict]:
    """
    Robustly process the state table structure handling inconsistencies.
    
    Args:
        table_df: DataFrame containing the state table data
        report_date: Report date in YYYY-MM-DD format
        
    Returns:
        List of dictionaries containing processed state records
    """
    logger.info("Processing state table robustly...")
    
    # Clean column names
    table_df.columns = [col.replace('\r', ' ').replace('\n', ' ') for col in table_df.columns]
    
    # Initialize result list
    results = []
    current_region = None
    
    for index, row in table_df.iterrows():
        # Skip completely empty rows
        if row.isna().all():
            continue
        
        # Get row data as list
        row_data = list(row)
        
        # Check if this is a region header row (has region code in first column)
        if len(row_data) > 0 and pd.notna(row_data[0]) and str(row_data[0]).strip() in ['NR', 'WR', 'SR', 'ER', 'NER']:
            current_region = str(row_data[0]).strip()
            # If second column has state name, process it
            if len(row_data) > 1 and pd.notna(row_data[1]) and str(row_data[1]).strip():
                state_name = str(row_data[1]).strip()
                if state_name != 'nan' and state_name != 'States':
                    try:
                        record = {
                            "region_code": current_region,
                            "state_name": state_name,
                            "power_max_demand_met_day_mw": extract_numeric_value_robust(row_data[2]) if len(row_data) > 2 else None,
                            "power_shortage_at_max_demand_mw": extract_numeric_value_robust(row_data[3]) if len(row_data) > 3 else None,
                            "energy_met_mu": extract_numeric_value_robust(row_data[4]) if len(row_data) > 4 else None,
                            "energy_drawal_schedule_mu": extract_numeric_value_robust(row_data[5]) if len(row_data) > 5 else None,
                            "energy_over_under_drawal_mu": extract_numeric_value_robust(row_data[6]) if len(row_data) > 6 else None,
                            "power_max_overdrawal_mw": extract_numeric_value_robust(row_data[7]) if len(row_data) > 7 else None,
                            "energy_shortage_mu": extract_numeric_value_robust(row_data[8]) if len(row_data) > 8 else None,
                            "report_date": report_date
                        }
                        results.append(record)
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(f"Failed to process region row {index}: {e}")
        
        # Check if this is a state row (empty first column, state name in second column)
        elif len(row_data) > 1 and (pd.isna(row_data[0]) or str(row_data[0]).strip() == '') and pd.notna(row_data[1]):
            state_name = str(row_data[1]).strip()
            if state_name != 'nan' and state_name != 'States' and current_region:
                try:
                    record = {
                        "region_code": current_region,
                        "state_name": state_name,
                        "power_max_demand_met_day_mw": extract_numeric_value_robust(row_data[2]) if len(row_data) > 2 else None,
                        "power_shortage_at_max_demand_mw": extract_numeric_value_robust(row_data[3]) if len(row_data) > 3 else None,
                        "energy_met_mu": extract_numeric_value_robust(row_data[4]) if len(row_data) > 4 else None,
                        "energy_drawal_schedule_mu": extract_numeric_value_robust(row_data[5]) if len(row_data) > 5 else None,
                        "energy_over_under_drawal_mu": extract_numeric_value_robust(row_data[6]) if len(row_data) > 6 else None,
                        "power_max_overdrawal_mw": extract_numeric_value_robust(row_data[7]) if len(row_data) > 7 else None,
                        "energy_shortage_mu": extract_numeric_value_robust(row_data[8]) if len(row_data) > 8 else None,
                        "report_date": report_date
                    }
                    results.append(record)
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Failed to process state row {index}: {e}")
        
        # Check for shifted data structure (state name in first column, no region code)
        elif len(row_data) > 0 and pd.notna(row_data[0]) and str(row_data[0]).strip() and str(row_data[0]).strip() not in ['NR', 'WR', 'SR', 'ER', 'NER'] and current_region:
            state_name = str(row_data[0]).strip()
            if state_name != 'nan' and state_name != 'States':
                try:
                    # Data is shifted left by one column
                    record = {
                        "region_code": current_region,
                        "state_name": state_name,
                        "power_max_demand_met_day_mw": extract_numeric_value_robust(row_data[1]) if len(row_data) > 1 else None,
                        "power_shortage_at_max_demand_mw": extract_numeric_value_robust(row_data[2]) if len(row_data) > 2 else None,
                        "energy_met_mu": extract_numeric_value_robust(row_data[3]) if len(row_data) > 3 else None,
                        "energy_drawal_schedule_mu": extract_numeric_value_robust(row_data[4]) if len(row_data) > 4 else None,
                        "energy_over_under_drawal_mu": extract_numeric_value_robust(row_data[5]) if len(row_data) > 5 else None,
                        "power_max_overdrawal_mw": extract_numeric_value_robust(row_data[6]) if len(row_data) > 6 else None,
                        "energy_shortage_mu": extract_numeric_value_robust(row_data[7]) if len(row_data) > 7 else None,
                        "report_date": report_date
                    }
                    results.append(record)
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Failed to process shifted state row {index}: {e}")
    
    logger.info(f"Successfully processed {len(results)} records robustly")
    return results


def validate_state_table_data(processed_records: List[Dict]) -> Dict[str, Any]:
    """
    Validate the processed state table data and provide summary statistics.
    
    Args:
        processed_records: List of processed state records
        
    Returns:
        Dictionary containing validation summary
    """
    if not processed_records:
        return {
            "valid": False,
            "total_records": 0,
            "regions": {},
            "errors": ["No records processed"]
        }
    
    # Count records by region
    region_counts = {}
    for record in processed_records:
        region = record.get('region_code', 'Unknown')
        region_counts[region] = region_counts.get(region, 0) + 1
    
    # Check for required fields
    required_fields = ['region_code', 'state_name', 'power_max_demand_met_day_mw', 'energy_met_mu']
    missing_fields = []
    
    for record in processed_records:
        for field in required_fields:
            if field not in record or record[field] is None:
                missing_fields.append(f"Record {record.get('state_name', 'Unknown')}: missing {field}")
    
    validation_result = {
        "valid": len(missing_fields) == 0,
        "total_records": len(processed_records),
        "regions": region_counts,
        "missing_fields": missing_fields,
        "expected_regions": ['NR', 'WR', 'SR', 'ER', 'NER'],
        "found_regions": list(region_counts.keys())
    }
    
    return validation_result


def get_state_table_summary(processed_records: List[Dict]) -> Dict[str, Any]:
    """
    Generate a summary of the processed state table data.
    
    Args:
        processed_records: List of processed state records
        
    Returns:
        Dictionary containing summary statistics
    """
    if not processed_records:
        return {"error": "No records to summarize"}
    
    # Calculate totals by region
    region_totals = {}
    for record in processed_records:
        region = record.get('region_code', 'Unknown')
        if region not in region_totals:
            region_totals[region] = {
                'power_max_demand_met_day_mw': 0,
                'energy_met_mu': 0,
                'energy_shortage_mu': 0,
                'state_count': 0
            }
        
        region_totals[region]['state_count'] += 1
        region_totals[region]['power_max_demand_met_day_mw'] += record.get('power_max_demand_met_day_mw', 0) or 0
        region_totals[region]['energy_met_mu'] += record.get('energy_met_mu', 0) or 0
        region_totals[region]['energy_shortage_mu'] += record.get('energy_shortage_mu', 0) or 0
    
    # Calculate national totals
    national_totals = {
        'total_power_max_demand_met_day_mw': sum(r.get('power_max_demand_met_day_mw', 0) or 0 for r in processed_records),
        'total_energy_met_mu': sum(r.get('energy_met_mu', 0) or 0 for r in processed_records),
        'total_energy_shortage_mu': sum(r.get('energy_shortage_mu', 0) or 0 for r in processed_records),
        'total_states': len(processed_records)
    }
    
    return {
        "national_totals": national_totals,
        "region_totals": region_totals,
        "record_count": len(processed_records)
    } 