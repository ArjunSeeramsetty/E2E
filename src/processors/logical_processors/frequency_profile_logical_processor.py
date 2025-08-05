#!/usr/bin/env python3
"""
Frequency Profile Logical Processor

This processor handles frequency profile tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class FrequencyProfileLogicalProcessor(BaseProcessor):
    """Logical processor for frequency profile tables using rule-based parsing."""
    TABLE_TYPE = "frequency_profile"
    KEYWORDS = ['fvi', '49.7', '49.8', '49.9', '50.05', 'frequency variation index', 'frequency_variation_index', 'frequency', 'region']
    REQUIRED_COLUMNS = ["region", "frequency_variation_index", "percentage_time_less_49_7", "percentage_time_49_7_to_49_8", "percentage_time_49_8_to_49_9", "percentage_time_less_49_9", "percentage_time_49_9_to_50_05", "percentage_time_greater_50_05", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process frequency profile table using robust rule-based logic."""
        logger.info("Processing with FrequencyProfileLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract frequency profile data
            records = []
            
            # Expected regions
            regions = ['All India']
            
            # Process each region
            for region in regions:
                region_data = self._extract_region_data(df, region, report_date)
                if region_data:
                    records.append(region_data)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} frequency profile records")
                return records
            else:
                logger.error("Validation failed for processed data for frequency_profile.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing frequency profile table: {e}")
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
    
    def _extract_region_data(self, df: pd.DataFrame, region: str, report_date: str) -> Dict[str, Any]:
        """Extract data for a specific region."""
        try:
            # Find the row with region data - look for the data row (not header)
            region_row = None
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                # Skip header row and look for data row
                if first_col.lower() == region.lower():
                    region_row = row
                    break
            
            if region_row is None:
                logger.warning(f"Could not find data row for region: {region}")
                return None
            
            # Extract frequency variation index (FVI)
            fvi = self._extract_numeric_value(region_row.iloc[1])  # FVI column
            
            # Extract percentage values for each frequency band
            less_49_7 = self._extract_numeric_value(region_row.iloc[2])  # < 49.7
            band_49_7_to_49_8 = self._extract_numeric_value(region_row.iloc[3])  # 49.7 - 49.8
            band_49_8_to_49_9 = self._extract_numeric_value(region_row.iloc[4])  # 49.8 - 49.9
            less_49_9 = self._extract_numeric_value(region_row.iloc[5])  # < 49.9
            band_49_9_to_50_05 = self._extract_numeric_value(region_row.iloc[6])  # 49.9 - 50.05
            greater_50_05 = self._extract_numeric_value(region_row.iloc[7])  # > 50.05
            
            return {
                "region": region,
                "frequency_variation_index": fvi,
                "percentage_time_less_49_7": less_49_7,
                "percentage_time_49_7_to_49_8": band_49_7_to_49_8,
                "percentage_time_49_8_to_49_9": band_49_8_to_49_9,
                "percentage_time_less_49_9": less_49_9,
                "percentage_time_49_9_to_50_05": band_49_9_to_50_05,
                "percentage_time_greater_50_05": greater_50_05,
                "report_date": report_date
            }
            
        except Exception as e:
            logger.error(f"Error extracting data for region {region}: {e}")
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
        """Validate the processed frequency profile data."""
        if not super().validate(data):
            return False
        
        # Check for expected regions
        expected_regions = {"All India"}
        found_regions = {record.get("region") for record in data}
        
        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")
        
        # Check for plausible numeric values
        for record in data:
            numeric_fields = [
                'frequency_variation_index', 'percentage_time_less_49_7', 
                'percentage_time_49_7_to_49_8', 'percentage_time_49_8_to_49_9',
                'percentage_time_less_49_9', 'percentage_time_49_9_to_50_05',
                'percentage_time_greater_50_05'
            ]
            
            for field in numeric_fields:
                value = record.get(field)
                if value is not None and not isinstance(value, (int, float)):
                    logger.error(f"Validation failed for {self.TABLE_TYPE}: '{field}' is not a number for region {record.get('region')}.")
                    return False
        
        logger.info(f"Validation passed for {self.TABLE_TYPE}.")
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the processed data."""
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "regions_found": list(set(record.get("region") for record in data)),
            "total_fvi": sum(record.get("frequency_variation_index", 0) or 0 for record in data),
            "total_percentage_time": sum(record.get("percentage_time_less_49_7", 0) or 0 for record in data) + 
                                   sum(record.get("percentage_time_49_7_to_49_8", 0) or 0 for record in data) +
                                   sum(record.get("percentage_time_49_8_to_49_9", 0) or 0 for record in data) +
                                   sum(record.get("percentage_time_less_49_9", 0) or 0 for record in data) +
                                   sum(record.get("percentage_time_49_9_to_50_05", 0) or 0 for record in data) +
                                   sum(record.get("percentage_time_greater_50_05", 0) or 0 for record in data)
        }
        
        # Calculate averages
        fvi_values = [record.get("frequency_variation_index") for record in data if record.get("frequency_variation_index") is not None]
        
        if fvi_values:
            summary["average_fvi"] = sum(fvi_values) / len(fvi_values)
            
        return summary 