#!/usr/bin/env python3
"""
Solar/Non-Solar Peak Demand Logical Processor

This processor handles solar/non-solar peak demand tables using rule-based parsing
instead of LLM processing. It extracts peak demand data for solar and non-solar hours.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from ..base_processor import BaseProcessor
from loguru import logger


class SolarNonSolarPeakLogicalProcessor(BaseProcessor):
    """Logical processor for solar/non-solar peak demand tables using rule-based parsing."""
    
    TABLE_TYPE = "solar_non_solar_peak"
    KEYWORDS = ['solar hr', 'non-solar hr', 'max demand met', 'peak demand', 'solar hour', 'non-solar hour']
    REQUIRED_COLUMNS = ["period_type", "power_max_demand_met_mw", "timestamp", "power_shortage_at_max_demand_mw", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> Optional[List[Dict]]:
        """
        Process solar/non-solar peak demand table using rule-based parsing.
        """
        logger.info(f"Processing with {self.__class__.__name__}...")
        
        try:
            result = []
            
            # Clean and normalize the dataframe
            df = table_df.copy()
            df.columns = [str(col).strip() for col in df.columns]
            
            # Expected structure: 2 rows (Solar hr, Non-Solar hr) with multiple columns
            # Look for the period type column (usually first column)
            period_col = df.columns[0]
            
            # Process each row to extract period data
            for idx, row in df.iterrows():
                period_name = str(row[period_col]).strip()
                
                # Skip empty rows or headers
                if pd.isna(period_name) or period_name == '' or 'nan' in period_name.lower():
                    continue
                
                # Normalize period name
                period_type = self._normalize_period_name(period_name)
                
                if period_type not in ["Solar hr", "Non-Solar hr"]:
                    continue
                
                # Extract values from the row
                max_demand = None
                timestamp = None
                shortage = None
                
                # Look for values in the row
                for col_idx in range(1, len(row)):
                    if col_idx >= len(row):
                        continue
                    
                    col_name = str(df.columns[col_idx]).strip()
                    value = row.iloc[col_idx]
                    
                    # Try to identify what each column represents
                    if 'max' in col_name.lower() or 'demand' in col_name.lower():
                        try:
                            max_demand = pd.to_numeric(str(value), errors='coerce')
                        except:
                            pass
                    elif 'time' in col_name.lower() or 'hr' in col_name.lower():
                        timestamp = str(value).strip()
                    elif 'shortage' in col_name.lower():
                        try:
                            shortage = pd.to_numeric(str(value), errors='coerce')
                        except:
                            pass
                    else:
                        # If we haven't found the values yet, try to parse them in order
                        if max_demand is None:
                            try:
                                max_demand = pd.to_numeric(str(value), errors='coerce')
                            except:
                                pass
                        elif timestamp is None:
                            timestamp = str(value).strip()
                        elif shortage is None:
                            try:
                                shortage = pd.to_numeric(str(value), errors='coerce')
                            except:
                                pass
                
                # Create record if we have at least the period type
                if period_type:
                    record = {
                        "period_type": period_type,
                        "power_max_demand_met_mw": max_demand,
                        "timestamp": timestamp,
                        "power_shortage_at_max_demand_mw": shortage,
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
            logger.error(f"Error processing solar/non-solar peak demand table: {e}")
            return None
    
    def _normalize_period_name(self, period_name: str) -> str:
        """
        Normalize period names to match expected values.
        """
        period_lower = period_name.lower()
        
        if 'solar' in period_lower:
            return "Solar hr"
        elif 'non-solar' in period_lower or 'non solar' in period_lower:
            return "Non-Solar hr"
        
        return period_name
    
    def validate(self, data: List[Dict]) -> bool:
        """Validate the extracted solar/non-solar peak demand data."""
        if not data:
            return False
            
        for record in data:
            # Check required fields
            if not all(field in record for field in self.REQUIRED_COLUMNS):
                return False
                
            # Validate period_type
            if not isinstance(record.get('period_type'), str):
                return False
                
            # Validate numeric values
            for field in ['power_max_demand_met_mw', 'power_shortage_at_max_demand_mw']:
                if record.get(field) is not None:
                    try:
                        float(record[field])
                    except (ValueError, TypeError):
                        return False
                        
            # Validate timestamp format (optional)
            timestamp = record.get('timestamp')
            if timestamp is not None and not isinstance(timestamp, str):
                return False
                
            # Validate report_date format
            if not isinstance(record.get('report_date'), str):
                return False
                
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict:
        """Generate a summary of the solar/non-solar peak demand data."""
        if not data:
            return {"error": "No data to summarize"}
            
        summary = {
            "total_records": len(data),
            "period_types": [],
            "max_demand_solar": None,
            "max_demand_non_solar": None,
            "total_shortage": 0.0
        }
        
        total_shortage = 0.0
        
        for record in data:
            period = record.get('period_type', '')
            max_demand = record.get('power_max_demand_met_mw')
            shortage = record.get('power_shortage_at_max_demand_mw')
            
            if period and period not in summary["period_types"]:
                summary["period_types"].append(period)
            
            if period == "Solar hr" and max_demand is not None:
                summary["max_demand_solar"] = max_demand
            elif period == "Non-Solar hr" and max_demand is not None:
                summary["max_demand_non_solar"] = max_demand
                
            if shortage is not None:
                total_shortage += float(shortage)
        
        summary["total_shortage"] = total_shortage
            
        return summary 