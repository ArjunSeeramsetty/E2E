#!/usr/bin/env python3
"""
DDF (Demand Deficiency Factor) Processor

This processor handles table 10 which contains demand deficiency factors
based on regional and state maximum demands.
"""

import pandas as pd
from typing import List, Dict, Optional
from ..base_processor import BaseProcessor
from loguru import logger


class DDFProcessor(BaseProcessor):
    """
    Processor for Demand Deficiency Factor (DDF) tables.
    
    Table 10 contains demand deficiency factors calculated based on:
    - Regional maximum demands
    - State maximum demands
    """
    
    TABLE_TYPE = "ddf"
    KEYWORDS = ['demand deficiency factor', 'ddf', 'regional max demands', 'state max demands', 'based on regional', 'based on state']
    REQUIRED_COLUMNS = ["calculation_basis", "ddf_value", "report_date"]
    
    PROMPT_TEMPLATE = """
    You are processing a Demand Deficiency Factor (DDF) table from a power supply position report.
    
    The table contains demand deficiency factors calculated based on different criteria:
    - Regional maximum demands
    - State maximum demands
    
    Please extract the data and return a JSON array of objects with the following structure:
    [
        {
            "calculation_basis": "Regional Max Demands",
            "ddf_value": 1.057,
            "report_date": "2025-01-27"
        },
        {
            "calculation_basis": "State Max Demands", 
            "ddf_value": 1.082,
            "report_date": "2025-01-27"
        }
    ]
    
    Important:
    - Extract the calculation basis (e.g., "Regional Max Demands", "State Max Demands")
    - Extract the DDF value as a numeric value
    - Include the report_date in every record
    - Return a JSON array, not a single object
    - Ensure all numeric values are properly formatted
    
    Table data:
    {input_table}
    
    Report date: {report_date}
    
    Return only the JSON array, no additional text or formatting.
    """
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> Optional[List[Dict]]:
        """
        Process DDF table using rule-based parsing since the structure is simple.
        """
        logger.info(f"Processing with {self.__class__.__name__}...")
        
        try:
            # Handle the specific structure of table 10
            # The table has unusual structure where the first row is header and second row is data
            result = []
            
            # The table structure is:
            # Column 0: "Based on Regional Max Demands" (header)
            # Column 1: "1.057" (value)
            # Row 0: "Based on State Max Demands" (calculation basis)
            # Row 0: "1.082" (DDF value)
            
            # First record: Regional Max Demands
            regional_basis = table_df.columns[0]  # "Based on Regional Max Demands"
            regional_value = table_df.columns[1]  # "1.057" (this is actually the value)
            
            try:
                regional_ddf = float(regional_value)
                result.append({
                    "calculation_basis": str(regional_basis),
                    "ddf_value": regional_ddf,
                    "report_date": report_date
                })
            except (ValueError, TypeError):
                logger.warning(f"Could not convert regional DDF value '{regional_value}' to float")
            
            # Second record: State Max Demands
            state_basis = table_df.iloc[0, 0]  # "Based on State Max Demands"
            state_value = table_df.iloc[0, 1]  # "1.082"
            
            try:
                state_ddf = float(state_value)
                result.append({
                    "calculation_basis": str(state_basis),
                    "ddf_value": state_ddf,
                    "report_date": report_date
                })
            except (ValueError, TypeError):
                logger.warning(f"Could not convert state DDF value '{state_value}' to float")
            
            if self.validate(result):
                logger.info(f"Successfully processed and validated data for {self.TABLE_TYPE}.")
                return result
            else:
                logger.error(f"Validation failed for processed data for {self.TABLE_TYPE}.")
                return None
                
        except Exception as e:
            logger.error(f"Error processing DDF table: {e}")
            return None
    
    def validate(self, data: List[Dict]) -> bool:
        """
        Validate the extracted DDF data.
        
        Args:
            data: List of dictionaries containing DDF records
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        if not data:
            return False
            
        for record in data:
            # Check required fields
            if not all(field in record for field in self.REQUIRED_COLUMNS):
                return False
                
            # Validate calculation_basis
            if not isinstance(record.get('calculation_basis'), str):
                return False
                
            # Validate ddf_value is numeric
            try:
                ddf_value = float(record.get('ddf_value', 0))
                if ddf_value <= 0:
                    return False
            except (ValueError, TypeError):
                return False
                
            # Validate report_date format
            if not isinstance(record.get('report_date'), str):
                return False
                
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict:
        """
        Generate a summary of the DDF data.
        
        Args:
            data: List of dictionaries containing DDF records
            
        Returns:
            Dict: Summary information
        """
        if not data:
            return {"error": "No data to summarize"}
            
        summary = {
            "total_records": len(data),
            "calculation_bases": [],
            "ddf_values": [],
            "average_ddf": 0.0
        }
        
        ddf_values = []
        
        for record in data:
            calculation_basis = record.get('calculation_basis', '')
            ddf_value = record.get('ddf_value', 0)
            
            if calculation_basis:
                summary["calculation_bases"].append(calculation_basis)
            if ddf_value:
                ddf_values.append(float(ddf_value))
                summary["ddf_values"].append(ddf_value)
        
        if ddf_values:
            summary["average_ddf"] = sum(ddf_values) / len(ddf_values)
            
        return summary 