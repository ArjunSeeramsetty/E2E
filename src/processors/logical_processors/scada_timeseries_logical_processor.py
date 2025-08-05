#!/usr/bin/env python3
"""
SCADA Timeseries Logical Processor

This processor handles SCADA timeseries tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import re
from loguru import logger

class ScadaTimeseriesLogicalProcessor(BaseProcessor):
    """Logical processor for SCADA timeseries tables using rule-based parsing."""
    TABLE_TYPE = "scada_timeseries"
    KEYWORDS = ['15 min (instantaneous)', 'all india grid frequency', 'generation & demand met', 'scada data', 'time', 'frequency (hz)', 'demand met (mw)', 'nuclear (mw)', 'wind (mw)', 'solar (mw)', 'hydro** (mw)', 'gas (mw)', 'thermal (mw)', 'others* (mw)', 'net demand met (mw)', 'total generation (mw)', 'net transnational exchange (mw)', 'scada', 'timeseries', 'instantaneous']
    REQUIRED_COLUMNS = ["timestamp", "frequency_hz", "power_demand_mw", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process SCADA timeseries table using robust rule-based logic."""
        logger.info("Processing with ScadaTimeseriesLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Find the data start row (first row with time pattern)
            data_start_row = self._find_data_start_row(df)
            if data_start_row == -1:
                logger.error("Could not find data start row in SCADA timeseries table")
                return []
            
            # Extract data starting from the identified row
            df = df.iloc[data_start_row:].reset_index(drop=True)
            
            # Assign standard column names
            standard_columns = [
                'TIME', 'FREQUENCY (Hz)', 'DEMAND MET (MW)', 'NUCLEAR (MW)', 'WIND (MW)', 'SOLAR (MW)',
                'HYDRO** (MW)', 'GAS (MW)', 'THERMAL (MW)', 'OTHERS* (MW)', 'NET DEMAND MET (MW)',
                'TOTAL GENERATION (MW)', 'NET TRANSNATIONAL EXCHANGE (MW)'
            ]
            
            if len(standard_columns) == len(df.columns):
                df.columns = standard_columns
            else:
                logger.error(f"Column count mismatch. Expected {len(standard_columns)}, got {len(df.columns)}")
                return []
            
            # Process each time series record
            records = []
            for idx, row in df.iterrows():
                record = self._extract_timeseries_record(row, report_date)
                if record:
                    records.append(record)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} timeseries records")
                return records
            else:
                logger.error("Validation failed for processed data for scada_timeseries.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing SCADA timeseries table: {e}")
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
    
    def _find_data_start_row(self, df: pd.DataFrame) -> int:
        """Find the first row that contains time data (HH:MM format)."""
        for i in range(min(10, len(df))):
            time_col = df.iloc[i, 0] if len(df.columns) > 0 else None
            if time_col is not None:
                time_str = str(time_col).strip()
                # Check if this looks like a time value (HH:MM format)
                if re.match(r'\d{1,2}:\d{2}', time_str):
                    return i
        return -1
    
    def _extract_timeseries_record(self, row: pd.Series, report_date: str) -> Dict[str, Any]:
        """Extract a single timeseries record from a row."""
        try:
            # Extract time
            time_value = str(row.get('TIME', '')).strip()
            if not time_value or not re.match(r'\d{1,2}:\d{2}', time_value):
                return None
            
            # Format time to HH:MM format (remove seconds if present)
            time_value = time_value[:5] if ':' in time_value else time_value
            
            # Extract frequency
            frequency = self._extract_numeric_value(row.get('FREQUENCY (Hz)', None))
            
            # Extract demand met
            demand_met = self._extract_numeric_value(row.get('DEMAND MET (MW)', None))
            
            # Extract other generation values
            nuclear = self._extract_numeric_value(row.get('NUCLEAR (MW)', None))
            wind = self._extract_numeric_value(row.get('WIND (MW)', None))
            solar = self._extract_numeric_value(row.get('SOLAR (MW)', None))
            hydro = self._extract_numeric_value(row.get('HYDRO** (MW)', None))
            gas = self._extract_numeric_value(row.get('GAS (MW)', None))
            thermal = self._extract_numeric_value(row.get('THERMAL (MW)', None))
            others = self._extract_numeric_value(row.get('OTHERS* (MW)', None))
            
            # Extract net demand and total generation
            net_demand = self._extract_numeric_value(row.get('NET DEMAND MET (MW)', None))
            total_generation = self._extract_numeric_value(row.get('TOTAL GENERATION (MW)', None))
            
            # Extract transnational exchange
            transnational = self._extract_numeric_value(row.get('NET TRANSNATIONAL EXCHANGE (MW)', None))
            
            return {
                "timestamp": time_value,
                "frequency_hz": frequency,
                "power_demand_mw": demand_met,
                "nuclear_generation_mw": nuclear,
                "wind_generation_mw": wind,
                "solar_generation_mw": solar,
                "hydro_generation_mw": hydro,
                "gas_generation_mw": gas,
                "thermal_generation_mw": thermal,
                "others_generation_mw": others,
                "net_demand_met_mw": net_demand,
                "total_generation_mw": total_generation,
                "transnational_exchange_mw": transnational,
                "report_date": report_date
            }
            
        except Exception as e:
            logger.error(f"Error extracting timeseries record: {e}")
            return None
    
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
        """Validate the processed SCADA timeseries data."""
        if not super().validate(data):
            return False
        
        # Check for expected time patterns
        time_pattern = re.compile(r'\d{1,2}:\d{2}')
        for record in data:
            timestamp = record.get("timestamp")
            if not timestamp or not time_pattern.match(str(timestamp)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: Invalid timestamp format: {timestamp}")
                return False
        
        # Check for plausible numeric values
        for record in data:
            frequency = record.get("frequency_hz")
            demand = record.get("power_demand_mw")
            
            if frequency is not None and not isinstance(frequency, (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'frequency_hz' is not a number for timestamp {record.get('timestamp')}.")
                return False
            
            if demand is not None and not isinstance(demand, (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'power_demand_mw' is not a number for timestamp {record.get('timestamp')}.")
                return False
        
        logger.info(f"Validation passed for {self.TABLE_TYPE}.")
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the processed data."""
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "time_range": {
                "start": min(record.get("timestamp", "") for record in data if record.get("timestamp")),
                "end": max(record.get("timestamp", "") for record in data if record.get("timestamp"))
            },
            "total_demand": sum(record.get("power_demand_mw", 0) or 0 for record in data),
            "total_generation": sum(record.get("total_generation_mw", 0) or 0 for record in data)
        }
        
        # Calculate averages
        frequency_values = [record.get("frequency_hz") for record in data if record.get("frequency_hz") is not None]
        demand_values = [record.get("power_demand_mw") for record in data if record.get("power_demand_mw") is not None]
        
        if frequency_values:
            summary["average_frequency"] = sum(frequency_values) / len(frequency_values)
        
        if demand_values:
            summary["average_demand"] = sum(demand_values) / len(demand_values)
            
        return summary 