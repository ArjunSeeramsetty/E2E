#!/usr/bin/env python3
"""
Regional Summary Processor

This processor handles the 'Regional Daily Summary' table type.
It extracts regional power supply data and structures it for database storage.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class RegionalSummaryProcessor(BaseProcessor):
    """Processor for the 'Regional Daily Summary' table."""
    TABLE_TYPE = "regional_daily_summary"
    KEYWORDS = ['demand met during evening peak', 'peak shortage', 'energy met', 'hydro gen', 'wind gen', 'solar gen', 'energy shortage', 'maximum demand met during the day', 'time of maximum demand met']
    REQUIRED_COLUMNS = [
        "region_code", "peak_demand_met_evng_mw", "energy_met_total_mu", "report_date"
    ]
    PROMPT_TEMPLATE = """
Extract regional power data from this table into JSON array with ALL regions (NR, WR, SR, ER, NER, All India).

Table: {input_table}

Output JSON array with objects containing:
- region_code (string)
- peak_demand_met_evng_mw (number)
- peak_shortage_evng_mw (number) 
- energy_met_total_mu (number)
- energy_shortage_total_mu (number)
- generation_hydro_mu (number)
- generation_wind_mu (number)
- generation_solar_mu (number)
- max_demand_met_day_mw (number)
- time_of_max_demand_day (string)
- report_date: "{report_date}"

Return ONLY valid JSON array with all 6 regions.
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the regional summary table."""
        if not super().validate(data):
            return False
        
        expected_regions = {"NR", "WR", "SR", "ER", "NER", "All India"}
        found_regions = {record.get("region_code") for record in data}

        if not expected_regions.issubset(found_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: Not all expected regions were found. Missing: {expected_regions - found_regions}")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_met_total_mu") is not None and not isinstance(record["energy_met_total_mu"], (int, float)):
                 logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_met_total_mu' is not a number for region {record.get('region_code')}.")
                 return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 