#!/usr/bin/env python3
"""
Generation by Source Logical Processor

This processor handles generation by source tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class GenerationBySourceLogicalProcessor(BaseProcessor):
    """Logical processor for generation by source tables using rule-based parsing."""
    TABLE_TYPE = "generation_by_source"
    KEYWORDS = ['sourcewise generation', 'gross generation', 'fuel type', 'coal', 'lignite', 'hydro', 'nuclear', 'gas naptha diesel', 'res wind solar biomass', 'sourcewise', 'coal', 'lignite', 'hydro', 'nuclear', 'gas, naptha & diesel', 'res (wind, solar, biomass & others)', 'total', '% share', 'generation by source', 'source wise']
    REQUIRED_COLUMNS = ["region_name", "source_name", "energy_generation_mu", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process generation by source table using robust rule-based logic."""
        logger.info("Processing with GenerationBySourceLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract generation data by source and region
            records = []
            
            # Expected regions
            regions = ['NR', 'WR', 'SR', 'ER', 'NER', 'All India']
            
            # Expected sources
            sources = ['Coal', 'Lignite', 'Hydro', 'Nuclear', 'Gas, Naptha & Diesel', 'RES (Wind, Solar, Biomass & Others)', 'Total']
            
            # Process each source row
            for idx, row in df.iterrows():
                source_name = str(row.iloc[0]).strip()
                
                # Skip empty rows or non-source rows
                if pd.isna(source_name) or source_name == '' or 'nan' in source_name.lower():
                    continue
                
                # Normalize source name
                normalized_source = self._normalize_source_name(source_name)
                if normalized_source not in sources:
                    continue
                
                # Process each region column
                for col_idx in range(1, len(df.columns)):
                    if col_idx >= len(row):
                        continue
                    
                    region_name = str(df.columns[col_idx]).strip()
                    value = row.iloc[col_idx]
                    
                    # Skip if region is not recognized
                    if region_name not in regions:
                        continue
                    
                    # Convert value to numeric
                    numeric_value = self._extract_numeric_value(value)
                    if numeric_value is None:
                        continue
                    
                    # Create record
                    record = {
                        "region_name": region_name,
                        "source_name": normalized_source,
                        "energy_generation_mu": numeric_value,
                        "report_date": report_date
                    }
                    
                    records.append(record)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} generation records")
                return records
            else:
                logger.error("Validation failed for processed data for generation_by_source.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing generation by source table: {e}")
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
    
    def _normalize_source_name(self, source_name: str) -> str:
        """Normalize source name to match expected values."""
        source_lower = source_name.lower()
        
        # Map variations to standard names
        if 'coal' in source_lower:
            return 'Coal'
        elif 'lignite' in source_lower:
            return 'Lignite'
        elif 'hydro' in source_lower:
            return 'Hydro'
        elif 'nuclear' in source_lower:
            return 'Nuclear'
        elif any(keyword in source_lower for keyword in ['gas', 'naptha', 'diesel']):
            return 'Gas, Naptha & Diesel'
        elif any(keyword in source_lower for keyword in ['res', 'wind', 'solar', 'biomass']):
            return 'RES (Wind, Solar, Biomass & Others)'
        elif 'total' in source_lower:
            return 'Total'
        
        return source_name
    
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
        """Validate the processed generation by source data."""
        if not super().validate(data):
            return False
        
        # Check for expected regions and sources
        expected_regions = {"NR", "WR", "SR", "ER", "NER", "All India"}
        expected_sources = {"Coal", "Lignite", "Hydro", "Nuclear", "Gas, Naptha & Diesel", "RES (Wind, Solar, Biomass & Others)", "Total"}
        
        found_regions = {record.get("region_name") for record in data}
        found_sources = {record.get("source_name") for record in data}
        
        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")
        
        if not found_sources.intersection(expected_sources):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected sources found.")
        
        # Check for plausible numeric values
        for record in data:
            energy_generation = record.get("energy_generation_mu")
            
            if energy_generation is not None and not isinstance(energy_generation, (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_generation_mu' is not a number for region {record.get('region_name')} and source {record.get('source_name')}.")
                return False
        
        logger.info(f"Validation passed for {self.TABLE_TYPE}.")
        return True
    
    def get_summary(self, data: List[Dict]) -> Dict[str, Any]:
        """Generate summary statistics for the processed data."""
        if not data:
            return {"total_records": 0}
        
        summary = {
            "total_records": len(data),
            "regions_found": list(set(record.get("region_name") for record in data)),
            "sources_found": list(set(record.get("source_name") for record in data)),
            "total_generation": sum(record.get("energy_generation_mu", 0) or 0 for record in data)
        }
        
        # Calculate averages by source
        source_totals = {}
        for record in data:
            source = record.get("source_name")
            energy = record.get("energy_generation_mu", 0) or 0
            if source not in source_totals:
                source_totals[source] = []
            source_totals[source].append(energy)
        
        for source, values in source_totals.items():
            if values:
                summary[f"avg_{source.lower().replace(' ', '_')}"] = sum(values) / len(values)
            
        return summary 