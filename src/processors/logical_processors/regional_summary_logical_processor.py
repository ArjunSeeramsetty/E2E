#!/usr/bin/env python3
"""
Regional Summary Logical Processor

This processor handles regional summary tables using rule-based parsing
instead of LLM processing. It extracts regional power supply data and
structures it for database storage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from ..base_processor import BaseProcessor
from loguru import logger


class RegionalSummaryLogicalProcessor(BaseProcessor):
    """Logical processor for regional summary tables using rule-based parsing."""
    
    TABLE_TYPE = "regional_daily_summary"
    KEYWORDS = ['demand met during evening peak', 'peak shortage', 'energy met', 'hydro gen', 'wind gen', 'solar gen', 'energy shortage', 'maximum demand met during the day', 'time of maximum demand met']
    REQUIRED_COLUMNS = ["region_code", "peak_demand_met_evng_mw", "energy_met_total_mu", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> Optional[List[Dict]]:
        """
        Process regional summary table using rule-based parsing.
        """
        logger.info(f"Processing with {self.__class__.__name__}...")
        
        try:
            result = []
            
            # Expected regions
            expected_regions = ["NR", "WR", "SR", "ER", "NER", "All India"]
            
            # Clean and normalize the dataframe
            df = table_df.copy()
            df.columns = [str(col).strip() for col in df.columns]
            
            # Find the metrics column (usually first column)
            metrics_col = df.columns[0]
            
            # Process each row to extract metrics
            for idx, row in df.iterrows():
                metric_name = str(row[metrics_col]).strip()
                
                # Skip empty rows or headers
                if pd.isna(metric_name) or metric_name == '' or 'nan' in metric_name.lower():
                    continue
                
                # Process each region column
                for col_idx in range(1, len(df.columns)):
                    if col_idx >= len(row):
                        continue
                        
                    region_name = str(df.columns[col_idx]).strip()
                    value = row.iloc[col_idx]
                    
                    # Skip if region is not recognized
                    if region_name not in expected_regions:
                        continue
                    
                    # Convert value to numeric
                    try:
                        numeric_value = pd.to_numeric(str(value), errors='coerce')
                        if pd.isna(numeric_value):
                            continue
                    except:
                        continue
                    
                    # Map metric names to standardized fields
                    record = self._create_record_from_metric(
                        metric_name, region_name, numeric_value, report_date
                    )
                    
                    if record:
                        result.append(record)
            
            if self.validate(result):
                logger.info(f"Successfully processed and validated data for {self.TABLE_TYPE}.")
                return result
            else:
                logger.error(f"Validation failed for processed data for {self.TABLE_TYPE}.")
                return None
                
        except Exception as e:
            logger.error(f"Error processing regional summary table: {e}")
            return None
    
    def _create_record_from_metric(self, metric_name: str, region_name: str, value: float, report_date: str) -> Optional[Dict]:
        """
        Create a standardized record from metric name and value.
        """
        metric_name_lower = metric_name.lower()
        
        # Map metric names to standardized fields
        if 'demand met during evening peak' in metric_name_lower:
            return {
                "region_code": region_name,
                "peak_demand_met_evng_mw": value,
                "energy_met_total_mu": None,
                "report_date": report_date
            }
        elif 'peak shortage' in metric_name_lower:
            return {
                "region_code": region_name,
                "peak_shortage_evng_mw": value,
                "energy_met_total_mu": None,
                "report_date": report_date
            }
        elif 'energy met' in metric_name_lower:
            return {
                "region_code": region_name,
                "energy_met_total_mu": value,
                "peak_demand_met_evng_mw": None,
                "report_date": report_date
            }
        elif 'energy shortage' in metric_name_lower:
            return {
                "region_code": region_name,
                "energy_shortage_total_mu": value,
                "energy_met_total_mu": None,
                "report_date": report_date
            }
        elif 'hydro gen' in metric_name_lower:
            return {
                "region_code": region_name,
                "generation_hydro_mu": value,
                "energy_met_total_mu": None,
                "report_date": report_date
            }
        elif 'wind gen' in metric_name_lower:
            return {
                "region_code": region_name,
                "generation_wind_mu": value,
                "energy_met_total_mu": None,
                "report_date": report_date
            }
        elif 'solar gen' in metric_name_lower:
            return {
                "region_code": region_name,
                "generation_solar_mu": value,
                "energy_met_total_mu": None,
                "report_date": report_date
            }
        elif 'maximum demand met during the day' in metric_name_lower:
            return {
                "region_code": region_name,
                "max_demand_met_day_mw": value,
                "energy_met_total_mu": None,
                "report_date": report_date
            }
        elif 'time of maximum demand met' in metric_name_lower:
            return {
                "region_code": region_name,
                "time_of_max_demand_day": str(value),
                "energy_met_total_mu": None,
                "report_date": report_date
            }
        
        return None
    
    def validate(self, data: List[Dict]) -> bool:
        """Validate the extracted regional summary data."""
        if not data:
            return False
            
        for record in data:
            # Check required fields
            if not all(field in record for field in self.REQUIRED_COLUMNS):
                return False
                
            # Validate region_code
            if not isinstance(record.get('region_code'), str):
                return False
                
            # Validate numeric values
            for field in ['peak_demand_met_evng_mw', 'energy_met_total_mu']:
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
        """Generate a summary of the regional summary data."""
        if not data:
            return {"error": "No data to summarize"}
            
        summary = {
            "total_records": len(data),
            "regions": [],
            "metrics": [],
            "average_energy_met": 0.0
        }
        
        energy_values = []
        
        for record in data:
            region = record.get('region_code', '')
            energy_met = record.get('energy_met_total_mu')
            
            if region and region not in summary["regions"]:
                summary["regions"].append(region)
            if energy_met is not None:
                energy_values.append(float(energy_met))
        
        if energy_values:
            summary["average_energy_met"] = sum(energy_values) / len(energy_values)
            
        return summary 