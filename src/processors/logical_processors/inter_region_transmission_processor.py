#!/usr/bin/env python3
"""
Inter-Region Transmission Processor

This processor handles the 'Inter-regional Transmission Line Flow' table type.
It extracts inter-regional transmission data and structures it for database storage.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict, Optional
from loguru import logger
import pandas as pd

class InterRegionTransmissionProcessor(BaseProcessor):
    """Processor for the 'Inter-regional Transmission Line Flow' table."""
    TABLE_TYPE = "inter_regional_transmission"
    KEYWORDS = ['inter-regional', 'voltage_level', 'line_details', 'sl_no', 'no_of_circuit', 'max import', 'max export', 'import', 'net']
    REQUIRED_COLUMNS = [
        "sl_no", "voltage_level", "line_details", "no_of_circuit", "from_region", "to_region", "power_max_import_mw", "power_max_export_mw", "energy_import_mu", "energy_export_mu", "energy_net_mu", "report_date"
    ]
    PROMPT_TEMPLATE = """
Extract transmission line data from this table.

**Table:**
```
{input_table}
```

**Output JSON array with objects containing:**
- sl_no: serial number
- voltage_level: voltage level
- line_details: line name
- no_of_circuit: number of circuits (null if "-")
- from_region: source region
- to_region: destination region
- power_max_import_mw: max import MW
- power_max_export_mw: max export MW
- energy_import_mu: import energy MU
- energy_export_mu: export energy MU
- energy_net_mu: net energy MU
- report_date: "{report_date}"

**IMPORTANT: Look for section headers like "Import/Export of ER (With NR)" or "Import/Export of ER (With WR)" in the Sl No column.**
**When you see a section header, extract the regions: "Import/Export of ER (With NR)" → from_region="ER", to_region="NR"**
**Use these regions for all transmission line rows that follow until you encounter another section header.**
**When you see a new section header, update the regions accordingly.**

**Example:**
- Section header: "Import/Export of ER (With NR)" → use from_region="ER", to_region="NR" for following rows
- Section header: "Import/Export of ER (With WR)" → use from_region="ER", to_region="WR" for following rows
- Section header: "Import/Export of ER (With SR)" → use from_region="ER", to_region="SR" for following rows

**Output ONLY valid JSON array.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the power exchanges table."""
        if not super().validate(data):
            return False
        
        # Check for expected voltage levels
        expected_voltage_levels = {"765 kV", "400 kV", "220 kV", "132 kV", "HVDC"}
        found_voltage_levels = {record.get("voltage_level") for record in data}

        if not found_voltage_levels.intersection(expected_voltage_levels):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected voltage levels found.")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_import_mu") is not None and not isinstance(record["energy_import_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_import_mu' is not a number for line {record.get('line_details')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 

    def process(self, table_df: pd.DataFrame, report_date: str) -> Optional[List[Dict]]:
        """Override process method to handle large tables by processing all rows with region preservation."""
        logger.info(f"Processing {len(table_df)} rows with region preservation")
        
        # Initialize variables to track current regions
        current_from_region = None
        current_to_region = None
        all_records = []
        
        # Process each row individually
        for index, row in table_df.iterrows():
            # Check if this row is a section header
            sl_no_value = str(row.get('Sl No', '')).strip()
            
            # Detect section headers like "Import/Export of ER (With NR)"
            if 'Import/Export of' in sl_no_value and '(' in sl_no_value and ')' in sl_no_value:
                # Extract regions from section header
                try:
                    # Parse "Import/Export of ER (With NR)" → from_region="ER", to_region="NR"
                    parts = sl_no_value.split('Import/Export of ')[1]
                    from_part = parts.split(' (With ')[0]
                    to_part = parts.split(' (With ')[1].rstrip(')')
                    
                    current_from_region = from_part.strip()
                    current_to_region = to_part.strip()
                    
                    logger.info(f"Section header detected: {sl_no_value} → from_region={current_from_region}, to_region={current_to_region}")
                    continue  # Skip this header row
                except Exception as e:
                    logger.warning(f"Failed to parse section header: {sl_no_value}, error: {e}")
                    continue
            
            # Skip summary rows (like "ER-NR,9.2,18.3,-9.1")
            if '-' in sl_no_value and ',' in sl_no_value and any(char.isdigit() for char in sl_no_value):
                logger.info(f"Skipping summary row: {sl_no_value}")
                continue
            
            # Additional check for summary rows with region pairs (like "ER-NR")
            if any(region_pair in sl_no_value for region_pair in ['ER-NR', 'ER-WR', 'ER-SR', 'ER-NER', 'NER-NR', 'WR-NR', 'WR-SR']):
                logger.info(f"Skipping summary row with region pair: {sl_no_value}")
                continue
            
            # Skip empty or non-data rows
            if pd.isna(row.get('Voltage Level')) or str(row.get('Sl No', '')).strip() == '':
                continue
            
            # Check if we have valid regions
            if current_from_region is None or current_to_region is None:
                logger.warning(f"No valid regions found for row {index}, skipping")
                continue
            
            # Validate line details - check if it contains numbers instead of text
            line_details = str(row.get('Line Details', '')).strip()
            if self._is_line_details_corrupted(line_details):
                logger.warning(f"Line details appears corrupted in row {index}: '{line_details}'. Using LLM fallback.")
                # Use LLM fallback for this row
                llm_record = self._process_row_with_llm(row, current_from_region, current_to_region, report_date)
                if llm_record:
                    all_records.append(llm_record)
                continue
            
            # Create record for this transmission line
            try:
                record = {
                    "sl_no": int(row.get('Sl No', 0)),
                    "voltage_level": str(row.get('Voltage Level', '')),
                    "line_details": line_details,
                    "no_of_circuit": None if str(row.get('No. of Circuit', '')).strip() == '-' else int(row.get('No. of Circuit', 0)),
                    "from_region": current_from_region,
                    "to_region": current_to_region,
                    "power_max_import_mw": float(row.get('Max Import (MW)', 0.0)),
                    "power_max_export_mw": float(row.get('Max Export (MW)', 0.0)),
                    "energy_import_mu": float(row.get('Import (MU)', 0.0)),
                    "energy_export_mu": float(row.get('Export (MU)', 0.0)),
                    "energy_net_mu": float(row.get('NET (MU)', 0.0)),
                    "report_date": report_date
                }
                all_records.append(record)
                logger.debug(f"Added record {len(all_records)}: {record['line_details']} ({current_from_region}→{current_to_region})")
                
            except Exception as e:
                logger.error(f"Failed to process row {index}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(all_records)} transmission line records")
        return all_records
    
    def _is_line_details_corrupted(self, line_details: str) -> bool:
        """Check if line details column contains numbers instead of text."""
        if not line_details or line_details == '':
            return False
        
        # Check if the line details is mostly numeric
        numeric_chars = sum(1 for c in line_details if c.isdigit())
        total_chars = len(line_details.replace(' ', '').replace('-', '').replace('/', ''))
        
        if total_chars == 0:
            return False
        
        numeric_ratio = numeric_chars / total_chars
        
        # If more than 70% of characters are numeric, it's likely corrupted
        if numeric_ratio > 0.7:
            return True
        
        # Check if it's just a number (like "123" or "456.78")
        try:
            float(line_details)
            return True
        except ValueError:
            pass
        
        # Check if it contains typical line name patterns
        line_name_patterns = ['-', '/', 'B/B', 'I/C', 'kV', 'HVDC']
        has_line_patterns = any(pattern in line_details.upper() for pattern in line_name_patterns)
        
        # If it's mostly numeric and doesn't have line name patterns, it's corrupted
        if numeric_ratio > 0.5 and not has_line_patterns:
            return True
        
        return False
    
    def _process_row_with_llm(self, row: pd.Series, from_region: str, to_region: str, report_date: str) -> Optional[Dict]:
        """Process a single row using LLM when rule-based parsing fails."""
        try:
            # Create a simple table with just this row
            row_df = pd.DataFrame([row])
            
            # Use the LLM to process this single row
            llm_result = super().process(row_df, report_date)
            
            if llm_result and len(llm_result) > 0:
                record = llm_result[0]
                # Ensure the regions are correct
                record['from_region'] = from_region
                record['to_region'] = to_region
                return record
            else:
                logger.error(f"LLM failed to process row: {row}")
                return None
                
        except Exception as e:
            logger.error(f"LLM fallback failed for row: {e}")
            return None 