#!/usr/bin/env python3
"""
Transnational Summary Logical Processor

This processor handles transnational summary tables using rule-based parsing
instead of LLM processing. It extracts transnational summary data with countries.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from ..base_processor import BaseProcessor
from loguru import logger


class TransnationalSummaryLogicalProcessor(BaseProcessor):
    """Logical processor for transnational summary tables using rule-based parsing."""
    
    TABLE_TYPE = "transnational_summary"
    KEYWORDS = ['bhutan', 'nepal', 'bangladesh', 'godda', 'actual', 'day peak', 'mu', 'mw', 'transnational']
    REQUIRED_COLUMNS = ["country", "energy_actual_mu", "power_day_peak_mw", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> Optional[List[Dict]]:
        """
        Process transnational summary table using rule-based parsing.
        """
        logger.info(f"Processing with {self.__class__.__name__}...")
        
        try:
            result = []
            
            # Expected countries
            expected_countries = ["Bhutan", "Nepal", "Bangladesh", "Godda -> Bangladesh"]
            
            # Clean and normalize the dataframe
            df = table_df.copy()
            df.columns = [str(col).strip() for col in df.columns]
            
            # This table typically has countries as columns and metrics as rows
            # Look for the metric column (usually first column)
            metric_col = df.columns[0]
            
            # Process each row to extract metric data
            for idx, row in df.iterrows():
                metric_name = str(row[metric_col]).strip()
                
                # Skip empty rows or headers
                if pd.isna(metric_name) or metric_name == '' or 'nan' in metric_name.lower():
                    continue
                
                # Normalize metric name
                metric_type = self._normalize_metric_name(metric_name)
                
                if metric_type not in ["energy_actual_mu", "power_day_peak_mw"]:
                    continue
                
                # Process each country column
                for col_idx in range(1, len(df.columns)):
                    if col_idx >= len(row):
                        continue
                        
                    country_name = str(df.columns[col_idx]).strip()
                    value = row.iloc[col_idx]
                    
                    # Normalize country name
                    country_name = self._normalize_country_name(country_name)
                    
                    # Skip if country is not recognized
                    if country_name not in expected_countries:
                        continue
                    
                    # Convert value to numeric
                    try:
                        numeric_value = pd.to_numeric(str(value), errors='coerce')
                        if pd.isna(numeric_value):
                            continue
                    except:
                        continue
                    
                    # Create record
                    record = {
                        "country": country_name,
                        metric_type: numeric_value,
                        "energy_actual_mu": None if metric_type != "energy_actual_mu" else numeric_value,
                        "power_day_peak_mw": None if metric_type != "power_day_peak_mw" else numeric_value,
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
            logger.error(f"Error processing transnational summary table: {e}")
            return None
    
    def _normalize_metric_name(self, metric_name: str) -> str:
        """
        Normalize metric names to match expected values.
        """
        metric_lower = metric_name.lower()
        
        if 'actual' in metric_lower and ('mu' in metric_lower or 'energy' in metric_lower):
            return "energy_actual_mu"
        elif 'day peak' in metric_lower or 'peak' in metric_lower:
            return "power_day_peak_mw"
        
        return metric_name
    
    def _normalize_country_name(self, country_name: str) -> str:
        """
        Normalize country names to match expected values.
        """
        country_lower = country_name.lower()
        
        if 'bhutan' in country_lower:
            return "Bhutan"
        elif 'nepal' in country_lower:
            return "Nepal"
        elif 'bangladesh' in country_lower:
            return "Bangladesh"
        elif 'godda' in country_lower:
            return "Godda -> Bangladesh"
        
        return country_name
    
    def validate(self, data: List[Dict]) -> bool:
        """Validate the extracted transnational summary data."""
        if not data:
            return False
            
        for record in data:
            # Check required fields
            if not all(field in record for field in self.REQUIRED_COLUMNS):
                return False
                
            # Validate country
            if not isinstance(record.get('country'), str):
                return False
                
            # Validate numeric values
            for field in ['energy_actual_mu', 'power_day_peak_mw']:
                if record.get(field) is not None:
                    try:
                        float(record[field])
                    except (ValueError, TypeError):
                        return False
                        
            # Validate report_date format
            if not isinstance(record.get('report_date'), str):
                return False
                
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict:
        """Generate a summary of the transnational summary data."""
        if not data:
            return {"error": "No data to summarize"}
            
        summary = {
            "total_records": len(data),
            "countries": [],
            "total_energy": 0.0,
            "total_power": 0.0
        }
        
        total_energy = 0.0
        total_power = 0.0
        
        for record in data:
            country = record.get('country', '')
            energy = record.get('energy_actual_mu')
            power = record.get('power_day_peak_mw')
            
            if country and country not in summary["countries"]:
                summary["countries"].append(country)
            if energy is not None:
                total_energy += float(energy)
            if power is not None:
                total_power += float(power)
        
        summary["total_energy"] = total_energy
        summary["total_power"] = total_power
            
        return summary 