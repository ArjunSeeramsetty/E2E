#!/usr/bin/env python3
"""
Solar/Non-Solar Peak Demand Processor

This processor handles the 'Solar/Non-Solar Hour Peak Demand' table type.
It extracts peak demand data for solar and non-solar hours and structures it for database storage.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class SolarNonSolarPeakProcessor(BaseProcessor):
    """Processor for the 'Solar/Non-Solar Hour Peak Demand' table."""
    TABLE_TYPE = "solar_non_solar_peak"
    KEYWORDS = ['solar hr', 'non-solar hr', 'max demand met', 'peak demand', 'solar hour', 'non-solar hour']
    REQUIRED_COLUMNS = [
        "period_type", "power_max_demand_met_mw", "timestamp", "power_shortage_at_max_demand_mw", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of solar/non-solar peak demand data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each period (Solar hr, Non-Solar hr).
2. For each period, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `period_type`: (String) Period type ("Solar hr" or "Non-Solar hr").
- `power_max_demand_met_mw`: (Number) Maximum demand met (MW).
- `timestamp`: (String) Time of peak demand (HH:MM format).
- `power_shortage_at_max_demand_mw`: (Number) Shortage at maximum demand (MW).
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the solar/non-solar peak demand table."""
        if not super().validate(data):
            return False
        
        # Check for expected period types
        expected_periods = {"Solar hr", "Non-Solar hr"}
        found_periods = {record.get("period_type") for record in data}

        if not found_periods.intersection(expected_periods):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected period types found.")

        # Check for plausible numeric values
        for record in data:
            if record.get("power_max_demand_met_mw") is not None and not isinstance(record["power_max_demand_met_mw"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'power_max_demand_met_mw' is not a number for period {record.get('period_type')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 