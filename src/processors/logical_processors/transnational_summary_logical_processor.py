#!/usr/bin/env python3
"""
Transnational Summary Logical Processor

This processor handles transnational summary tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class TransnationalSummaryLogicalProcessor(BaseProcessor):
    """Logical processor for transnational summary tables using rule-based parsing."""
    TABLE_TYPE = "transnational_summary"
    KEYWORDS = ['bhutan', 'nepal', 'bangladesh', 'godda', 'actual', 'day peak', 'mu', 'mw', 'transnational', 'international exchanges', 'country', 'state', 'region', 'line name', 'max (mw)', 'min (mw)', 'avg (mw)', 'energy exchange (mu)']
    REQUIRED_COLUMNS = ["country", "energy_actual_mu", "power_day_peak_mw", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process transnational summary table using robust rule-based logic."""
        logger.info("Processing with TransnationalSummaryLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract transnational data
            records = []
            
            # Expected countries/entities
            countries = ['Bhutan', 'Nepal', 'Bangladesh', 'Godda -> Bangladesh']
            
            # Process each country/entity
            for country in countries:
                if country in df.columns:
                    country_data = self._extract_country_data(df, country, report_date)
                    if country_data:
                        records.append(country_data)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} transnational records")
                return records
            else:
                logger.error("Validation failed for processed data for transnational_summary.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing transnational summary table: {e}")
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
    
    def _extract_country_data(self, df: pd.DataFrame, country: str, report_date: str) -> Dict[str, Any]:
        """Extract data for a specific country/entity."""
        try:
            # Find the row with "Actual (MU)"
            actual_mu_row = None
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                if 'actual' in first_col.lower() and 'mu' in first_col.lower():
                    actual_mu_row = row
                    break
            
            if actual_mu_row is None:
                return None
            
            # Find the row with "Day Peak (MW)"
            day_peak_row = None
            for idx, row in df.iterrows():
                first_col = str(row.iloc[0]).strip()
                if 'day peak' in first_col.lower() and 'mw' in first_col.lower():
                    day_peak_row = row
                    break
            
            # Get the values for this country
            if country in df.columns:
                country_idx = df.columns.get_loc(country)
                
                energy_actual = None
                if actual_mu_row is not None and country_idx < len(actual_mu_row):
                    energy_actual = self._extract_numeric_value(actual_mu_row.iloc[country_idx])
                
                power_day_peak = None
                if day_peak_row is not None and country_idx < len(day_peak_row):
                    power_day_peak = self._extract_numeric_value(day_peak_row.iloc[country_idx])
                
                return {
                    "country": country,
                    "energy_actual_mu": energy_actual,
                    "power_day_peak_mw": power_day_peak,
                    "report_date": report_date
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting data for country {country}: {e}")
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
        """Validate the processed transnational summary data."""
        if not super().validate(data):
            return False
        
        # Check for expected countries/entities
        expected_countries = {"Bhutan", "Nepal", "Bangladesh", "Godda -> Bangladesh"}
        found_countries = {record.get("country") for record in data}
        
        if not found_countries.intersection(expected_countries):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected countries found.")
        
        # Check for plausible numeric values
        for record in data:
            energy_actual = record.get("energy_actual_mu")
            power_day_peak = record.get("power_day_peak_mw")
            
            if energy_actual is not None and not isinstance(energy_actual, (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_actual_mu' is not a number for country {record.get('country')}.")
                return False
            
            if power_day_peak is not None and not isinstance(power_day_peak, (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'power_day_peak_mw' is not a number for country {record.get('country')}.")
                return False
        
        logger.info(f"Validation passed for {self.TABLE_TYPE}.")
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the processed data."""
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "countries_found": list(set(record.get("country") for record in data)),
            "total_energy_actual": sum(record.get("energy_actual_mu", 0) or 0 for record in data),
            "total_power_day_peak": sum(record.get("power_day_peak_mw", 0) or 0 for record in data)
        }
        
        # Calculate averages
        energy_values = [record.get("energy_actual_mu") for record in data if record.get("energy_actual_mu") is not None]
        power_values = [record.get("power_day_peak_mw") for record in data if record.get("power_day_peak_mw") is not None]
        
        if energy_values:
            summary["average_energy_actual"] = sum(energy_values) / len(energy_values)
        
        if power_values:
            summary["average_power_day_peak"] = sum(power_values) / len(power_values)
            
        return summary 