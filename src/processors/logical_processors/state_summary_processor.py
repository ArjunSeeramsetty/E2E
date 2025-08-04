#!/usr/bin/env python3
"""
State Summary Processor

This processor handles the 'State Daily Summary' table type.
It uses robust processing logic to handle structural inconsistencies in state data.
"""

from ..base_processor import BaseProcessor
from .raw_state_table_processing import (
    process_state_table_robustly,
    validate_state_table_data,
    get_state_table_summary
)
from typing import List, Dict, Optional
from loguru import logger

class StateSummaryProcessor(BaseProcessor):
    """Processor for the 'State Daily Summary' table using robust processing."""
    TABLE_TYPE = "state_daily_summary"
    KEYWORDS = ['state', 'states', 'maximum demand', 'drawal schedule', 'od/ud']
    REQUIRED_COLUMNS = [
        "region_code", "state_name", "power_max_demand_met_day_mw", "energy_met_mu", "report_date"
    ]

    def process(self, table_df, report_date: str) -> Optional[List[Dict]]:
        """
        Override the base process method to use robust processing instead of LLM.
        This handles the structural inconsistencies in state tables more reliably.
        """
        logger.info(f"Processing state table with robust processing...")
        
        try:
            # Use the robust processing logic
            structured_data = process_state_table_robustly(table_df, report_date)
            
            if not structured_data:
                logger.error(f"Robust processing returned no data for {self.TABLE_TYPE}.")
                return None
            
            # Validate the robust processing results
            validation = validate_state_table_data(structured_data)
            if not validation.get("valid", False):
                logger.error(f"Validation failed for robust processing: {validation}")
                return None
            
            # Get summary for logging
            summary = get_state_table_summary(structured_data)
            logger.info(f"Successfully processed {len(structured_data)} state records")
            logger.info(f"Regional breakdown: {summary.get('region_totals', {})}")
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Robust processing failed for {self.TABLE_TYPE}: {e}")
            return None

    def validate(self, data: List[Dict]) -> bool:
        """Validation for the state summary table using robust processing results."""
        if not isinstance(data, list) or not data:
            logger.error(f"Validation failed for {self.TABLE_TYPE}: Data is not a non-empty list.")
            return False

        # Check if all required columns are present in the first record
        first_record = data[0]
        if not all(key in first_record for key in self.REQUIRED_COLUMNS):
            missing_keys = [key for key in self.REQUIRED_COLUMNS if key not in first_record]
            logger.error(f"Validation failed for {self.TABLE_TYPE}: Missing required keys {missing_keys}.")
            return False

        # Check for expected regions
        expected_regions = {"NR", "WR", "SR", "ER", "NER"}
        found_regions = {record.get("region_code") for record in data}
        
        if not expected_regions.issubset(found_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: Not all expected regions found. Missing: {expected_regions - found_regions}")

        # Check for reasonable number of states (should be around 30-40)
        if len(data) < 20:
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: Found fewer than 20 state records ({len(data)}).")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_met_mu") is not None and not isinstance(record["energy_met_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_met_mu' is not a number for state {record.get('state_name')}.")
                return False

        logger.info(f"Validation passed for {self.TABLE_TYPE} with {len(data)} records.")
        return True 