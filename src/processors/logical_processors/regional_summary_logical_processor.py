#!/usr/bin/env python3
"""
Regional Summary Logical Processor

This processor handles regional summary tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class RegionalSummaryLogicalProcessor(BaseProcessor):
    """Logical processor for regional summary tables using rule-based parsing."""
    TABLE_TYPE = "regional_daily_summary"
    KEYWORDS = ['demand met during evening peak', 'peak shortage', 'energy met', 'hydro gen', 'wind gen', 'solar gen', 'energy shortage', 'maximum demand met during the day', 'time of maximum demand met', 'regional daily summary', 'evening peak hrs', 'from rldcs']
    REQUIRED_COLUMNS = ["region_code", "peak_demand_met_evng_mw", "energy_met_total_mu", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process regional summary table using robust rule-based logic."""
        logger.info("Processing with RegionalSummaryLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract regional data
            records = []
            
            # Process each region (NR, WR, SR, ER, NER, TOTAL)
            regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'TOTAL']
            
            for region in regions:
                if region in df.columns:
                    region_data = self._extract_region_data(df, region, report_date)
                    if region_data:
                        records.append(region_data)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} regional records")
                return records
            else:
                logger.error("Validation failed for processed data for regional_daily_summary.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing regional summary table: {e}")
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
            # Find all the required rows
            evening_peak_row = None
            peak_shortage_row = None
            energy_met_row = None
            energy_shortage_row = None
            hydro_gen_row = None
            wind_gen_row = None
            solar_gen_row = None
            max_demand_row = None
            time_max_demand_row = None
            
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                
                if 'demand met during evening peak' in first_col.lower():
                    evening_peak_row = row
                elif 'peak shortage' in first_col.lower():
                    peak_shortage_row = row
                elif 'energy met' in first_col.lower() and 'shortage' not in first_col.lower():
                    energy_met_row = row
                elif 'energy shortage' in first_col.lower():
                    energy_shortage_row = row
                elif 'hydro gen' in first_col.lower():
                    hydro_gen_row = row
                elif 'wind gen' in first_col.lower():
                    wind_gen_row = row
                elif 'solar gen' in first_col.lower():
                    solar_gen_row = row
                elif 'maximum demand met during the day' in first_col.lower():
                    max_demand_row = row
                elif 'time of maximum demand met' in first_col.lower():
                    time_max_demand_row = row
            
            if region in df.columns:
                region_idx = df.columns.get_loc(region)
                
                # Extract all metrics
                peak_demand = self._extract_numeric_value(evening_peak_row.iloc[region_idx] if evening_peak_row is not None and region_idx < len(evening_peak_row) else None)
                peak_shortage = self._extract_numeric_value(peak_shortage_row.iloc[region_idx] if peak_shortage_row is not None and region_idx < len(peak_shortage_row) else None)
                energy_met = self._extract_numeric_value(energy_met_row.iloc[region_idx] if energy_met_row is not None and region_idx < len(energy_met_row) else None)
                energy_shortage = self._extract_numeric_value(energy_shortage_row.iloc[region_idx] if energy_shortage_row is not None and region_idx < len(energy_shortage_row) else None)
                hydro_gen = self._extract_numeric_value(hydro_gen_row.iloc[region_idx] if hydro_gen_row is not None and region_idx < len(hydro_gen_row) else None)
                wind_gen = self._extract_numeric_value(wind_gen_row.iloc[region_idx] if wind_gen_row is not None and region_idx < len(wind_gen_row) else None)
                solar_gen = self._extract_numeric_value(solar_gen_row.iloc[region_idx] if solar_gen_row is not None and region_idx < len(solar_gen_row) else None)
                max_demand = self._extract_numeric_value(max_demand_row.iloc[region_idx] if max_demand_row is not None and region_idx < len(max_demand_row) else None)
                time_max_demand = str(time_max_demand_row.iloc[region_idx]) if time_max_demand_row is not None and region_idx < len(time_max_demand_row) else None
                
                # Clean time value
                if time_max_demand and time_max_demand.lower() != 'nan':
                    time_max_demand = time_max_demand.strip()
                else:
                    time_max_demand = None
                
                return {
                    "region_code": region,
                    "peak_demand_met_evng_mw": peak_demand,
                    "peak_shortage_evng_mw": peak_shortage,
                    "energy_met_total_mu": energy_met,
                    "energy_shortage_total_mu": energy_shortage,
                    "generation_hydro_mu": hydro_gen,
                    "generation_wind_mu": wind_gen,
                    "generation_solar_mu": solar_gen,
                    "max_demand_met_day_mw": max_demand,
                    "time_of_max_demand_day": time_max_demand,
                    "report_date": report_date
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting data for region {region}: {e}")
            return None
    
    def _find_energy_met(self, df: pd.DataFrame, region: str) -> float:
        """Find energy met value for the given region."""
        try:
            # Look for "Energy Met (MU)" row
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                if 'energy met' in first_col.lower():
                    if region in df.columns:
                        region_idx = df.columns.get_loc(region)
                        if region_idx < len(row):
                            return self._extract_numeric_value(row.iloc[region_idx])
            return None
        except Exception as e:
            logger.error(f"Error finding energy met for region {region}: {e}")
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
        """Validate the processed regional summary data."""
        if not super().validate(data):
            return False
        
        # Check for expected regions
        expected_regions = {"NR", "WR", "SR", "ER", "NER", "TOTAL"}
        found_regions = {record.get("region_code") for record in data}
        
        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")
        
        # Check for plausible numeric values
        for record in data:
            numeric_fields = [
                'peak_demand_met_evng_mw', 'peak_shortage_evng_mw', 'energy_met_total_mu', 
                'energy_shortage_total_mu', 'generation_hydro_mu', 'generation_wind_mu', 
                'generation_solar_mu', 'max_demand_met_day_mw'
            ]
            
            for field in numeric_fields:
                value = record.get(field)
                if value is not None and not isinstance(value, (int, float)):
                    logger.error(f"Validation failed for {self.TABLE_TYPE}: '{field}' is not a number for region {record.get('region_code')}.")
                    return False
        
        logger.info(f"Validation passed for {self.TABLE_TYPE}.")
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the processed data."""
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "regions_found": list(set(record.get("region_code") for record in data)),
            "total_peak_demand": sum(record.get("peak_demand_met_evng_mw", 0) or 0 for record in data),
            "total_energy_met": sum(record.get("energy_met_total_mu", 0) or 0 for record in data)
        }
        
        # Calculate averages
        peak_values = [record.get("peak_demand_met_evng_mw") for record in data if record.get("peak_demand_met_evng_mw") is not None]
        energy_values = [record.get("energy_met_total_mu") for record in data if record.get("energy_met_total_mu") is not None]
        
        if peak_values:
            summary["average_peak_demand"] = sum(peak_values) / len(peak_values)
        
        if energy_values:
            summary["average_energy_met"] = sum(energy_values) / len(energy_values)
            
        return summary 