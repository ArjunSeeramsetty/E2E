#!/usr/bin/env python3
"""
Share Logical Processor

This processor handles share tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class ShareLogicalProcessor(BaseProcessor):
    """Logical processor for share tables using rule-based parsing."""
    TABLE_TYPE = "share"
    KEYWORDS = ['share percentage', 'share data', 'generation share', 'share of res', 'share of non-fossil', 'share of hydro', 'share of nuclear', '% share', 'percentage share', 'res share', 'non-fossil share']
    REQUIRED_COLUMNS = ["measure", "region_name", "share_percentage", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process share table using robust rule-based logic."""
        logger.info("Processing with ShareLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract share data
            records = []
            
            # Expected regions (columns 2-7)
            regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'ALL India']
            
            # Process each measure (each row)
            for idx, row in df.iterrows():
                measure = str(row.iloc[0]).strip()
                
                # Skip empty rows
                if pd.isna(measure) or measure == '' or 'nan' in measure.lower():
                    continue
                
                # Extract data for each region
                for i, region in enumerate(regions):
                    if i + 1 < len(row):  # +1 because first column is measure name
                        share_percentage = self._extract_numeric_value(row.iloc[i + 1])
                        
                        if share_percentage is not None:
                            records.append({
                                "region_name": region,
                                "measure": measure,
                                "share_percentage": share_percentage,
                                "report_date": report_date
                            })
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} share records")
                return records
            else:
                logger.error("Validation failed for processed data for share.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing share table: {e}")
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
    
    def _extract_measure_data(self, df: pd.DataFrame, measure: str, regions: List[str], report_date: str) -> List[Dict[str, Any]]:
        """Extract data for a specific measure across all regions."""
        try:
            records = []
            
            # Find the row with measure data
            measure_row = None
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                if measure.lower() in first_col.lower():
                    measure_row = row
                    break
            
            if measure_row is None:
                return []
            
            # Extract data for each region
            for i, region in enumerate(regions):
                if i + 1 < len(measure_row):  # +1 because first column is measure name
                    share_percentage = self._extract_numeric_value(measure_row.iloc[i + 1])
                    
                    if share_percentage is not None:
                        records.append({
                            "region_name": region,
                            "measure": measure,
                            "share_percentage": share_percentage,
                            "report_date": report_date
                        })
            
            return records
            
        except Exception as e:
            logger.error(f"Error extracting data for measure {measure}: {e}")
            return []
    
    def _determine_measure_type(self, text: str) -> str:
        """Determine the measure type from the text."""
        text_lower = text.lower()
        
        if 'res' in text_lower:
            return 'RES'
        elif 'non-fossil' in text_lower or 'non fossil' in text_lower:
            return 'Non-fossil'
        elif 'hydro' in text_lower:
            return 'Hydro'
        elif 'nuclear' in text_lower:
            return 'Nuclear'
        else:
            return 'Unknown'
    
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
        """Validate the processed share data."""
        if not super().validate(data):
            return False
        
        # Check for expected measures and regions
        expected_measures = {"RES", "Non-fossil", "Hydro", "Nuclear"}
        expected_regions = {"NR", "WR", "SR", "ER", "NER", "All India"}
        
        found_measures = {record.get("measure") for record in data}
        found_regions = {record.get("region_name") for record in data}
        
        if not found_measures.intersection(expected_measures):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected measures found.")
        
        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")
        
        # Check for plausible numeric values (share percentages should be between 0 and 100)
        for record in data:
            share_percentage = record.get("share_percentage")
            
            if share_percentage is not None:
                if not isinstance(share_percentage, (int, float)):
                    logger.error(f"Validation failed for {self.TABLE_TYPE}: 'share_percentage' is not a number for measure {record.get('measure')} and region {record.get('region_name')}.")
                    return False
                
                if share_percentage < 0 or share_percentage > 100:
                    logger.warning(f"Validation warning for {self.TABLE_TYPE}: Share percentage {share_percentage} is outside expected range (0-100) for measure {record.get('measure')} and region {record.get('region_name')}.")
        
        logger.info(f"Validation passed for {self.TABLE_TYPE}.")
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the processed data."""
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "measures_found": list(set(record.get("measure") for record in data)),
            "regions_found": list(set(record.get("region_name") for record in data)),
            "total_share_percentage": sum(record.get("share_percentage", 0) or 0 for record in data)
        }
        
        # Calculate averages by measure
        measure_totals = {}
        for record in data:
            measure = record.get("measure")
            percentage = record.get("share_percentage", 0) or 0
            if measure not in measure_totals:
                measure_totals[measure] = []
            measure_totals[measure].append(percentage)
        
        for measure, values in measure_totals.items():
            if values:
                summary[f"avg_{measure.lower().replace('-', '_')}_share"] = sum(values) / len(values)
            
        return summary 