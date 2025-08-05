#!/usr/bin/env python3
"""
Regional Import Export Summary Logical Processor

This processor handles regional import export summary tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class RegionalImportExportSummaryLogicalProcessor(BaseProcessor):
    """Logical processor for regional import export summary tables using rule-based parsing."""
    TABLE_TYPE = "regional_import_export_summary"
    KEYWORDS = ['schedule(mu)', 'actual(mu)', 'o/d/u/d(mu)', 'regional import export summary', 'schedule', 'actual', 'o/d/u/d', 'regional', 'import', 'export', 'overdrawal', 'underdrawal']
    REQUIRED_COLUMNS = ["region", "energy_schedule_mu", "energy_actual_mu", "energy_overdrawal_mu", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process regional import export summary table using robust rule-based logic."""
        logger.info("Processing with RegionalImportExportSummaryLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract regional import export data
            records = []
            
            # Expected regions
            regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India', 'TOTAL']
            
            # Process each region
            for region in regions:
                if region in df.columns:
                    region_data = self._extract_region_data(df, region, report_date)
                    if region_data:
                        records.append(region_data)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} regional import export records")
                return records
            else:
                logger.error("Validation failed for processed data for regional_import_export_summary.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing regional import export summary table: {e}")
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
            # Find the row with "Schedule (MU)"
            schedule_row = None
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                if 'schedule' in first_col.lower() and 'mu' in first_col.lower():
                    schedule_row = row
                    break
            
            # Find the row with "Actual (MU)"
            actual_row = None
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                if 'actual' in first_col.lower() and 'mu' in first_col.lower():
                    actual_row = row
                    break
            
            # Find the row with "O/D/U/D (MU)"
            overdraft_row = None
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                if ('o/d/u/d' in first_col.lower() or 'overdrawal' in first_col.lower() or 'underdrawal' in first_col.lower()) and 'mu' in first_col.lower():
                    overdraft_row = row
                    break
            
            # Get the values for this region
            if region in df.columns:
                region_idx = df.columns.get_loc(region)
                
                energy_schedule = None
                if schedule_row is not None and region_idx < len(schedule_row):
                    energy_schedule = self._extract_numeric_value(schedule_row.iloc[region_idx])
                
                energy_actual = None
                if actual_row is not None and region_idx < len(actual_row):
                    energy_actual = self._extract_numeric_value(actual_row.iloc[region_idx])
                
                energy_overdrawal = None
                if overdraft_row is not None and region_idx < len(overdraft_row):
                    energy_overdrawal = self._extract_numeric_value(overdraft_row.iloc[region_idx])
                
                return {
                    "region": region,
                    "energy_schedule_mu": energy_schedule,
                    "energy_actual_mu": energy_actual,
                    "energy_overdrawal_mu": energy_overdrawal,
                    "report_date": report_date
                }
            
            return None
            
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
        """Validate the processed regional import export summary data."""
        if not super().validate(data):
            return False
        
        # Check for expected regions
        expected_regions = {"NR", "WR", "SR", "ER", "NER", "All India", "TOTAL"}
        found_regions = {record.get("region") for record in data}
        
        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")
        
        # Check for plausible numeric values
        for record in data:
            numeric_fields = ['energy_schedule_mu', 'energy_actual_mu', 'energy_overdrawal_mu']
            
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
            "total_energy_schedule": sum(record.get("energy_schedule_mu", 0) or 0 for record in data),
            "total_energy_actual": sum(record.get("energy_actual_mu", 0) or 0 for record in data),
            "total_energy_overdrawal": sum(record.get("energy_overdrawal_mu", 0) or 0 for record in data)
        }
        
        # Calculate averages
        schedule_values = [record.get("energy_schedule_mu") for record in data if record.get("energy_schedule_mu") is not None]
        actual_values = [record.get("energy_actual_mu") for record in data if record.get("energy_actual_mu") is not None]
        overdraft_values = [record.get("energy_overdrawal_mu") for record in data if record.get("energy_overdrawal_mu") is not None]
        
        if schedule_values:
            summary["average_energy_schedule"] = sum(schedule_values) / len(schedule_values)
        
        if actual_values:
            summary["average_energy_actual"] = sum(actual_values) / len(actual_values)
        
        if overdraft_values:
            summary["average_energy_overdrawal"] = sum(overdraft_values) / len(overdraft_values)
            
        return summary 