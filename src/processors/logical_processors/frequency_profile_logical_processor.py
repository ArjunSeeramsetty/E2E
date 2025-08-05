#!/usr/bin/env python3
"""
Frequency Profile Logical Processor

This processor handles frequency profile tables using rule-based parsing
instead of LLM processing. It extracts frequency variation data and
structures it for database storage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from ..base_processor import BaseProcessor
from loguru import logger


class FrequencyProfileLogicalProcessor(BaseProcessor):
    """Logical processor for frequency profile tables using rule-based parsing."""
    
    TABLE_TYPE = "frequency_profile"
    KEYWORDS = ['fvi', '49.7', '49.8', '49.9', '50.05', 'frequency variation index', 'frequency_variation_index']
    REQUIRED_COLUMNS = ["frequency_band", "percentage_time", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> Optional[List[Dict]]:
        """
        Process frequency profile table using rule-based parsing.
        """
        logger.info(f"Processing with {self.__class__.__name__}...")
        
        try:
            result = []
            
            # Expected frequency bands
            expected_bands = [
                "FVI",
                "Frequency (<49.7)",
                "Frequency (49.7 - 49.8)",
                "Frequency (49.8 - 49.9)",
                "Frequency (< 49.9)",
                "Frequency (49.9 - 50.05)",
                "Frequency (> 50.05)"
            ]
            
            # Clean and normalize the dataframe
            df = table_df.copy()
            df.columns = [str(col).strip() for col in df.columns]
            
            # Find the frequency band column (usually first column)
            band_col = df.columns[0]
            
            # Process each row to extract frequency data
            for idx, row in df.iterrows():
                band_name = str(row[band_col]).strip()
                
                # Skip empty rows or headers
                if pd.isna(band_name) or band_name == '' or 'nan' in band_name.lower():
                    continue
                
                # Normalize band name
                band_name = self._normalize_band_name(band_name)
                
                # Skip if band is not recognized
                if band_name not in expected_bands:
                    continue
                
                # Look for All India row or total values
                all_india_value = None
                
                # Check if there's an "All India" column or row
                for col_idx in range(1, len(df.columns)):
                    if col_idx >= len(row):
                        continue
                        
                    col_name = str(df.columns[col_idx]).strip()
                    value = row.iloc[col_idx]
                    
                    # Look for All India values
                    if 'all india' in col_name.lower() or 'total' in col_name.lower():
                        try:
                            numeric_value = pd.to_numeric(str(value), errors='coerce')
                            if not pd.isna(numeric_value):
                                all_india_value = numeric_value
                                break
                        except:
                            continue
                
                # If no All India column found, use the first numeric value
                if all_india_value is None:
                    for col_idx in range(1, len(df.columns)):
                        if col_idx >= len(row):
                            continue
                        value = row.iloc[col_idx]
                        try:
                            numeric_value = pd.to_numeric(str(value), errors='coerce')
                            if not pd.isna(numeric_value):
                                all_india_value = numeric_value
                                break
                        except:
                            continue
                
                if all_india_value is not None:
                    # Create record
                    record = {
                        "frequency_band": band_name,
                        "percentage_time": all_india_value,
                        "report_date": report_date
                    }
                    
                    result.append(record)
            
            if self.validate(result):
                logger.info(f"Successfully processed and validated data for {self.TABLE_TYPE}.")
                return result
            else:
                logger.error(f"Validation failed for processed data for {self.TABLE_TYPE}.")
                return None
                
        except Exception as e:
            logger.error(f"Error processing frequency profile table: {e}")
            return None
    
    def _normalize_band_name(self, band_name: str) -> str:
        """
        Normalize frequency band names to match expected values.
        """
        band_lower = band_name.lower()
        
        if 'fvi' in band_lower or 'frequency variation index' in band_lower:
            return "FVI"
        elif '<49.7' in band_lower or '49.7' in band_lower:
            return "Frequency (<49.7)"
        elif '49.7 - 49.8' in band_lower:
            return "Frequency (49.7 - 49.8)"
        elif '49.8 - 49.9' in band_lower:
            return "Frequency (49.8 - 49.9)"
        elif '< 49.9' in band_lower:
            return "Frequency (< 49.9)"
        elif '49.9 - 50.05' in band_lower:
            return "Frequency (49.9 - 50.05)"
        elif '> 50.05' in band_lower:
            return "Frequency (> 50.05)"
        
        return band_name
    
    def validate(self, data: List[Dict]) -> bool:
        """Validate the extracted frequency profile data."""
        if not data:
            return False
            
        for record in data:
            # Check required fields
            if not all(field in record for field in self.REQUIRED_COLUMNS):
                return False
                
            # Validate frequency_band
            if not isinstance(record.get('frequency_band'), str):
                return False
                
            # Validate percentage_time is numeric and between 0-100
            try:
                percentage = float(record.get('percentage_time', 0))
                if percentage < 0 or percentage > 100:
                    return False
            except (ValueError, TypeError):
                return False
                
            # Validate report_date format
            if not isinstance(record.get('report_date'), str):
                return False
                
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict:
        """Generate a summary of the frequency profile data."""
        if not data:
            return {"error": "No data to summarize"}
            
        summary = {
            "total_records": len(data),
            "frequency_bands": [],
            "total_percentage": 0.0,
            "fvi_value": None
        }
        
        total_percentage = 0.0
        
        for record in data:
            band = record.get('frequency_band', '')
            percentage = record.get('percentage_time', 0)
            
            if band and band not in summary["frequency_bands"]:
                summary["frequency_bands"].append(band)
            if percentage:
                total_percentage += float(percentage)
                
            # Extract FVI value
            if 'fvi' in band.lower():
                summary["fvi_value"] = percentage
        
        summary["total_percentage"] = total_percentage
            
        return summary 