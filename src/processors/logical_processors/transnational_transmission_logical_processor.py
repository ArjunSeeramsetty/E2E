#!/usr/bin/env python3
"""
Transnational Transmission Logical Processor

This processor handles transnational transmission tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class TransnationalTransmissionLogicalProcessor(BaseProcessor):
    """Logical processor for transnational transmission tables using rule-based parsing."""
    TABLE_TYPE = "transnational_transmission"
    KEYWORDS = ['international exchanges', 'state', 'region', 'line name', 'max (mw)', 'min (mw)', 'avg (mw)', 'energy exchange (mu)', 'transnational transmission', 'international transmission', 'line details', 'voltage level', 'no. of circuit', 'bhutan', 'nepal', 'bangladesh']
    REQUIRED_COLUMNS = ["state", "region", "line_name", "power_max_mw", "power_min_mw", "power_avg_mw", "energy_exchange_mu", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process transnational transmission table using robust rule-based logic."""
        logger.info("Processing with TransnationalTransmissionLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract transmission data
            records = []
            
            # Expected columns for transmission data
            expected_columns = ['State', 'Region', 'Line Name', 'Max (MW)', 'Min (MW)', 'Avg (MW)', 'Energy Exchange (MU)']
            
            # Check if we have the right number of columns
            if len(df.columns) != len(expected_columns):
                logger.error(f"Column count mismatch. Expected {len(expected_columns)}, got {len(df.columns)}")
                return []
            
            # Assign column names
            df.columns = expected_columns
            
            # Process each row
            for idx, row in df.iterrows():
                state = str(row.get('State', '')).strip()
                region = str(row.get('Region', '')).strip()
                line_name = str(row.get('Line Name', '')).strip()
                
                # Skip empty rows or header rows
                if not line_name or 'state' in state.lower() or 'region' in region.lower():
                    continue
                
                # Extract numeric values
                max_mw = self._extract_numeric_value(row.get('Max (MW)'))
                min_mw = self._extract_numeric_value(row.get('Min (MW)'))
                avg_mw = self._extract_numeric_value(row.get('Avg (MW)'))
                energy_exchange = self._extract_numeric_value(row.get('Energy Exchange (MU)'))
                
                # Handle state propagation (if state is empty, use previous state)
                if not state and idx > 0:
                    prev_state = records[-1].get('state') if records else None
                    if prev_state:
                        state = prev_state
                
                # Handle region propagation (if region is empty, use previous region)
                if not region and idx > 0:
                    prev_region = records[-1].get('region') if records else None
                    if prev_region:
                        region = prev_region
                
                record = {
                    "state": state,
                    "region": region,
                    "line_name": line_name,
                    "power_max_mw": max_mw,
                    "power_min_mw": min_mw,
                    "power_avg_mw": avg_mw,
                    "energy_exchange_mu": energy_exchange,
                    "report_date": report_date
                }
                
                records.append(record)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} transmission records")
                return records
            else:
                logger.error("Validation failed for processed data for transnational_transmission.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing transnational transmission table: {e}")
            return []
    
    def _clean_column_names(self, columns):
        """Clean column names by removing special characters and normalizing."""
        cleaned = []
        for col in columns:
            if pd.isna(col):
                cleaned.append("Unnamed")
            else:
                # Remove special characters and normalize
                col_str = str(col).strip()
                col_str = col_str.replace('\r', ' ').replace('\n', ' ')
                col_str = ' '.join(col_str.split())  # Normalize whitespace
                cleaned.append(col_str)
        return cleaned
    
    def _extract_numeric_value(self, value) -> float:
        """Extract numeric value from various formats."""
        try:
            if pd.isna(value):
                return None
            
            # Convert to string and clean
            value_str = str(value).strip()
            
            # Remove common non-numeric characters
            value_str = value_str.replace(',', '').replace('%', '').replace('(', '').replace(')', '')
            
            # Try to convert to float
            return float(value_str)
        except (ValueError, TypeError):
            return None
    
    def validate(self, data: List[Dict]) -> bool:
        """Validate the processed transnational transmission data."""
        if not super().validate(data):
            return False
        
        # Check for expected regions
        expected_regions = {"NR", "WR", "SR", "ER", "NER"}
        found_regions = {record.get("region") for record in data}
        
        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")
        
        # Check for plausible numeric values
        for record in data:
            numeric_fields = ['power_max_mw', 'power_min_mw', 'power_avg_mw', 'energy_exchange_mu']
            
            for field in numeric_fields:
                value = record.get(field)
                if value is not None and not isinstance(value, (int, float)):
                    logger.error(f"Validation failed for {self.TABLE_TYPE}: '{field}' is not a number for line {record.get('line_name')}.")
                    return False
        
        logger.info(f"Validation passed for {self.TABLE_TYPE}.")
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the processed data."""
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "states_found": list(set(record.get("state") for record in data)),
            "regions_found": list(set(record.get("region") for record in data)),
            "lines_found": list(set(record.get("line_name") for record in data)),
            "total_power_max": sum(record.get("power_max_mw", 0) or 0 for record in data),
            "total_power_min": sum(record.get("power_min_mw", 0) or 0 for record in data),
            "total_power_avg": sum(record.get("power_avg_mw", 0) or 0 for record in data),
            "total_energy_exchange": sum(record.get("energy_exchange_mu", 0) or 0 for record in data)
        }
        
        # Calculate averages
        max_values = [record.get("power_max_mw") for record in data if record.get("power_max_mw") is not None]
        min_values = [record.get("power_min_mw") for record in data if record.get("power_min_mw") is not None]
        avg_values = [record.get("power_avg_mw") for record in data if record.get("power_avg_mw") is not None]
        exchange_values = [record.get("energy_exchange_mu") for record in data if record.get("energy_exchange_mu") is not None]
        
        if max_values:
            summary["average_power_max"] = sum(max_values) / len(max_values)
        
        if min_values:
            summary["average_power_min"] = sum(min_values) / len(min_values)
        
        if avg_values:
            summary["average_power_avg"] = sum(avg_values) / len(avg_values)
        
        if exchange_values:
            summary["average_energy_exchange"] = sum(exchange_values) / len(exchange_values)
            
        return summary 