#!/usr/bin/env python3
"""
Generation Outages Processor

This processor handles the 'Generation Outages' table type.
It extracts generation outage data and structures it for database storage.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class GenerationOutagesProcessor(BaseProcessor):
    """Processor for the 'Generation Outages' table."""
    TABLE_TYPE = "generation_outages"
    KEYWORDS = ['outage', 'generation outage', 'sector', 'central sector', 'state sector', 'mw']
    REQUIRED_COLUMNS = [
        "sector_name", "region_name", "power_outage_capacity_mw", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of generation outage data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each sector (Central Sector, State Sector, Private Sector, etc.).
2. For each sector, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `region_name`: (String) Name of the region (e.g., "NR", "WR", "SR", "ER", "NER", "All India")
- `sector_name`: (String) Name of the sector (e.g., "Central Sector", "State Sector", "Total").
- `power_outage_capacity_mw`: (Number) Outage Capacity (MW).
- `report_date`: (String) "{report_date}"

**Important:** For the "Total" row, extract regional data (NR, WR, SR, ER, NER, All India) just like other sectors. Do not use null for region_name in the Total row.

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the generation outages table."""
        if not super().validate(data):
            return False
        
        expected_sectors = {"Central Sector", "State Sector", "Total"}
        found_sectors = {record.get("sector_name") for record in data}

        if not expected_sectors.issubset(found_sectors):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: Not all expected sectors found. Missing: {expected_sectors - found_sectors}")

        # Check for plausible numeric values
        for record in data:
            if record.get("power_outage_capacity_mw") is not None and not isinstance(record["power_outage_capacity_mw"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'power_outage_capacity_mw' is not a number for sector {record.get('sector_name')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 