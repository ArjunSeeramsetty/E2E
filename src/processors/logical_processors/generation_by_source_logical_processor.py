#!/usr/bin/env python3
"""
Generation By Source Logical Processor

This processor handles generation by source tables using rule-based parsing
instead of LLM processing. It extracts source-wise generation data and
structures it for database storage.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from ..base_processor import BaseProcessor
from loguru import logger


class GenerationBySourceLogicalProcessor(BaseProcessor):
    """Logical processor for generation by source tables using rule-based parsing."""
    
    TABLE_TYPE = "generation_by_source"
    KEYWORDS = ['source', 'sourcewise', 'gross generation', 'fuel type', 'coal', 'hydro', 'nuclear', 'thermal']
    REQUIRED_COLUMNS = ["region_name", "source_name", "energy_generation_mu", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> Optional[List[Dict]]:
        """
        Process generation by source table using rule-based parsing.
        """
        logger.info(f"Processing with {self.__class__.__name__}...")
        
        try:
            result = []
            
            # Expected regions and sources
            expected_regions = ["NR", "WR", "SR", "ER", "NER", "All India"]
            expected_sources = ["Coal", "Lignite", "Hydro", "Nuclear", "Gas, Naptha & Diesel", "RES (Wind, Solar, Biomass & Others)", "Total"]
            
            # Clean and normalize the dataframe
            df = table_df.copy()
            df.columns = [str(col).strip() for col in df.columns]
            
            # Find the source column (usually first column)
            source_col = df.columns[0]
            
            # Process each row to extract source data
            for idx, row in df.iterrows():
                source_name = str(row[source_col]).strip()
                
                # Skip empty rows or headers
                if pd.isna(source_name) or source_name == '' or 'nan' in source_name.lower():
                    continue
                
                # Normalize source name
                source_name = self._normalize_source_name(source_name)
                
                # Skip if source is not recognized
                if source_name not in expected_sources:
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
                    
                    # Create record
                    record = {
                        "region_name": region_name,
                        "source_name": source_name,
                        "energy_generation_mu": numeric_value,
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
            logger.error(f"Error processing generation by source table: {e}")
            return None
    
    def _normalize_source_name(self, source_name: str) -> str:
        """
        Normalize source names to match expected values.
        """
        source_lower = source_name.lower()
        
        if 'coal' in source_lower:
            return "Coal"
        elif 'lignite' in source_lower:
            return "Lignite"
        elif 'hydro' in source_lower:
            return "Hydro"
        elif 'nuclear' in source_lower:
            return "Nuclear"
        elif any(x in source_lower for x in ['gas', 'naptha', 'diesel']):
            return "Gas, Naptha & Diesel"
        elif any(x in source_lower for x in ['res', 'wind', 'solar', 'biomass']):
            return "RES (Wind, Solar, Biomass & Others)"
        elif 'total' in source_lower:
            return "Total"
        
        return source_name
    
    def validate(self, data: List[Dict]) -> bool:
        """Validate the extracted generation by source data."""
        if not data:
            return False
            
        for record in data:
            # Check required fields
            if not all(field in record for field in self.REQUIRED_COLUMNS):
                return False
                
            # Validate region_name and source_name
            if not isinstance(record.get('region_name'), str):
                return False
            if not isinstance(record.get('source_name'), str):
                return False
                
            # Validate energy_generation_mu is numeric
            try:
                energy_value = float(record.get('energy_generation_mu', 0))
                if energy_value < 0:
                    return False
            except (ValueError, TypeError):
                return False
                
            # Validate report_date format
            if not isinstance(record.get('report_date'), str):
                return False
                
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict:
        """Generate a summary of the generation by source data."""
        if not data:
            return {"error": "No data to summarize"}
            
        summary = {
            "total_records": len(data),
            "regions": [],
            "sources": [],
            "total_generation": 0.0
        }
        
        total_generation = 0.0
        
        for record in data:
            region = record.get('region_name', '')
            source = record.get('source_name', '')
            energy = record.get('energy_generation_mu', 0)
            
            if region and region not in summary["regions"]:
                summary["regions"].append(region)
            if source and source not in summary["sources"]:
                summary["sources"].append(source)
            if energy:
                total_generation += float(energy)
        
        summary["total_generation"] = total_generation
            
        return summary 