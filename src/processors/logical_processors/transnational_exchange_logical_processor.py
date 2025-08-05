#!/usr/bin/env python3
"""
Transnational Exchange Logical Processor

This processor handles transnational exchange tables using robust rule-based parsing
based on the working logic from modular_psp_parser_old.py.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from loguru import logger

class TransnationalExchangeLogicalProcessor(BaseProcessor):
    """Logical processor for transnational exchange tables using rule-based parsing."""
    TABLE_TYPE = "transnational_exchange"
    KEYWORDS = ['country', 'ppa', 'bilateral', 'dam iex', 'dam pxil', 'dam hpx', 'rtm iex', 'rtm pxil', 'rtm hpx', 'total', 'export', 'import', 'net', 'exchange', 'transnational exchange']
    REQUIRED_COLUMNS = ["country", "ppa", "bilateral", "dam_iex", "dam_pxil", "dam_hpx", "rtm_iex", "rtm_pxil", "rtm_hpx", "total", "exchange_type", "report_date"]
    
    def process(self, table_df: pd.DataFrame, report_date: str) -> List[Dict[str, Any]]:
        """Process transnational exchange table using robust rule-based logic."""
        logger.info("Processing with TransnationalExchangeLogicalProcessor...")
        
        try:
            # Clean column names
            df = table_df.copy()
            df.columns = self._clean_column_names(df.columns)
            
            # Extract exchange data
            records = []
            
            # Expected exchange types
            exchange_types = ['Export', 'Import', 'NET']
            
            # Process each exchange type
            for exchange_type in exchange_types:
                exchange_data = self._extract_exchange_data(df, exchange_type, report_date)
                if exchange_data:
                    records.extend(exchange_data)
            
            # Validate the processed data
            if self.validate(records):
                logger.info(f"Successfully processed {len(records)} exchange records")
                return records
            else:
                logger.error("Validation failed for processed data for transnational_exchange.")
                return []
                
        except Exception as e:
            logger.error(f"Error processing transnational exchange table: {e}")
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
    
    def _extract_exchange_data(self, df: pd.DataFrame, exchange_type: str, report_date: str) -> List[Dict[str, Any]]:
        """Extract data for a specific exchange type."""
        try:
            records = []
            
            # Expected columns for exchange data
            expected_columns = ['Country', 'PPA', 'Bilateral', 'DAM IEX', 'DAM PXIL', 'DAM HPX', 'RTM IEX', 'RTM PXIL', 'RTM HPX', 'Total']
            
            # Skip first 3 rows (header rows)
            if len(df) <= 3:
                logger.warning(f"Not enough rows for {exchange_type} exchange data")
                return []
            
            # Process data starting from row 4
            df_data = df.iloc[3:].reset_index(drop=True)
            
            # Assign column names
            if len(df_data.columns) == len(expected_columns):
                df_data.columns = expected_columns
            else:
                logger.error(f"Column count mismatch for {exchange_type}. Expected {len(expected_columns)}, got {len(df_data.columns)}")
                return []
            
            # Process each row
            for idx, row in df_data.iterrows():
                country = str(row.get('Country', '')).strip()
                
                # Skip empty rows or total rows
                if not country or 'total' in country.lower():
                    continue
                
                # Extract numeric values
                ppa = self._extract_numeric_value(row.get('PPA'))
                bilateral = self._extract_numeric_value(row.get('Bilateral'))
                dam_iex = self._extract_numeric_value(row.get('DAM IEX'))
                dam_pxil = self._extract_numeric_value(row.get('DAM PXIL'))
                dam_hpx = self._extract_numeric_value(row.get('DAM HPX'))
                rtm_iex = self._extract_numeric_value(row.get('RTM IEX'))
                rtm_pxil = self._extract_numeric_value(row.get('RTM PXIL'))
                rtm_hpx = self._extract_numeric_value(row.get('RTM HPX'))
                total = self._extract_numeric_value(row.get('Total'))
                
                record = {
                    "country": country,
                    "ppa": ppa,
                    "bilateral": bilateral,
                    "dam_iex": dam_iex,
                    "dam_pxil": dam_pxil,
                    "dam_hpx": dam_hpx,
                    "rtm_iex": rtm_iex,
                    "rtm_pxil": rtm_pxil,
                    "rtm_hpx": rtm_hpx,
                    "total": total,
                    "exchange_type": exchange_type,
                    "report_date": report_date
                }
                
                records.append(record)
            
            return records
            
        except Exception as e:
            logger.error(f"Error extracting {exchange_type} exchange data: {e}")
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
        """Validate the processed transnational exchange data."""
        if not super().validate(data):
            return False
        
        # Check for expected exchange types
        expected_types = {"Export", "Import", "NET"}
        found_types = {record.get("exchange_type") for record in data}
        
        if not found_types.intersection(expected_types):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected exchange types found.")
        
        # Check for plausible numeric values
        for record in data:
            numeric_fields = ['ppa', 'bilateral', 'dam_iex', 'dam_pxil', 'dam_hpx', 'rtm_iex', 'rtm_pxil', 'rtm_hpx', 'total']
            
            for field in numeric_fields:
                value = record.get(field)
                if value is not None and not isinstance(value, (int, float)):
                    logger.error(f"Validation failed for {self.TABLE_TYPE}: '{field}' is not a number for country {record.get('country')}.")
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
            "exchange_types_found": list(set(record.get("exchange_type") for record in data)),
            "total_ppa": sum(record.get("ppa", 0) or 0 for record in data),
            "total_bilateral": sum(record.get("bilateral", 0) or 0 for record in data),
            "total_dam_iex": sum(record.get("dam_iex", 0) or 0 for record in data),
            "total_dam_pxil": sum(record.get("dam_pxil", 0) or 0 for record in data),
            "total_dam_hpx": sum(record.get("dam_hpx", 0) or 0 for record in data),
            "total_rtm_iex": sum(record.get("rtm_iex", 0) or 0 for record in data),
            "total_rtm_pxil": sum(record.get("rtm_pxil", 0) or 0 for record in data),
            "total_rtm_hpx": sum(record.get("rtm_hpx", 0) or 0 for record in data),
            "total_exchange": sum(record.get("total", 0) or 0 for record in data)
        }
        
        # Calculate averages by exchange type
        exchange_totals = {}
        for record in data:
            exchange_type = record.get("exchange_type")
            total = record.get("total", 0) or 0
            if exchange_type not in exchange_totals:
                exchange_totals[exchange_type] = []
            exchange_totals[exchange_type].append(total)
        
        for exchange_type, values in exchange_totals.items():
            if values:
                summary[f"avg_{exchange_type.lower()}_total"] = sum(values) / len(values)
            
        return summary 