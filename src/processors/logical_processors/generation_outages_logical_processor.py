#!/usr/bin/env python3
"""
Generation Outages Logical Processor

This processor handles generation outages tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class GenerationOutagesLogicalProcessor(BaseProcessor):
    """Logical processor for generation outages tables using rule-based parsing."""
    TABLE_TYPE = "generation_outages"
    KEYWORDS = ['outage', 'generation outage', 'sector', 'central sector', 'state sector', 'mw', 'central sector', 'state sector', 'total', '% share', 'capacity', 'outage capacity']
    REQUIRED_COLUMNS = ["sector_name", "region_name", "power_outage_capacity_mw", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process generation outages table using robust rule-based logic."""
        logger.info("Processing with GenerationOutagesLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract generation outages data
            records = []
            
            # Expected regions
            regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'TOTAL']
            
            # Process each sector
            sectors = ['Central Sector', 'State Sector', 'Total']
            
            for sector in sectors:
                sector_data = self._extract_sector_data(df, sector, regions, report_date)
                if sector_data:
                    records.extend(sector_data)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} generation outages records")
                return records
            else:
                logger.error("Validation failed for processed data for generation_outages.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing generation outages table: {e}")
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
    
    def _extract_sector_data(self, df: pd.DataFrame, sector: str, regions: List[str], report_date: str) -> List[Dict[str, Any]]:
        """Extract data for a specific sector across all regions."""
        try:
            records = []
            
            # Find the row with sector data
            sector_row = None
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                if sector.lower() in first_col.lower():
                    sector_row = row
                    break
            
            if sector_row is None:
                return []
            
            # Extract data for each region
            for i, region in enumerate(regions):
                if i + 1 < len(sector_row):  # +1 because first column is sector name
                    outage_capacity = self._extract_numeric_value(sector_row.iloc[i + 1])
                    
                    if outage_capacity is not None:
                        records.append({
                            "region_name": region,
                            "sector_name": sector,
                            "power_outage_capacity_mw": outage_capacity,
                            "report_date": report_date
                        })
            
            return records
            
        except Exception as e:
            logger.error(f"Error extracting data for sector {sector}: {e}")
            return []
    
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
        """Validate the processed generation outages data."""
        if not super().validate(data):
            return False
        
        # Check for expected sectors and regions
        expected_sectors = {"Central Sector", "State Sector", "Total"}
        expected_regions = {"NR", "WR", "SR", "ER", "NER", "All India"}
        
        found_sectors = {record.get("sector_name") for record in data}
        found_regions = {record.get("region_name") for record in data}
        
        if not found_sectors.intersection(expected_sectors):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected sectors found.")
        
        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")
        
        # Check for plausible numeric values
        for record in data:
            outage_capacity = record.get("power_outage_capacity_mw")
            
            if outage_capacity is not None and not isinstance(outage_capacity, (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'power_outage_capacity_mw' is not a number for sector {record.get('sector_name')} and region {record.get('region_name')}.")
                return False
        
        logger.info(f"Validation passed for {self.TABLE_TYPE}.")
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the processed data."""
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "sectors_found": list(set(record.get("sector_name") for record in data)),
            "regions_found": list(set(record.get("region_name") for record in data)),
            "total_outage_capacity": sum(record.get("power_outage_capacity_mw", 0) or 0 for record in data)
        }
        
        # Calculate averages by sector
        sector_totals = {}
        for record in data:
            sector = record.get("sector_name")
            capacity = record.get("power_outage_capacity_mw", 0) or 0
            if sector not in sector_totals:
                sector_totals[sector] = []
            sector_totals[sector].append(capacity)
        
        for sector, values in sector_totals.items():
            if values:
                summary[f"avg_{sector.lower().replace(' ', '_')}_capacity"] = sum(values) / len(values)
            
        return summary 