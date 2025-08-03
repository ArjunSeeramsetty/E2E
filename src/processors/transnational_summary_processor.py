#!/usr/bin/env python3
"""
Transnational Summary Processor

This processor handles the 'Transnational Summary' table type.
It extracts transnational summary data with countries and their energy/power metrics.
"""

from .base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class TransnationalSummaryProcessor(BaseProcessor):
    """Processor for the 'Transnational Summary' table."""
    TABLE_TYPE = "transnational_summary"
    KEYWORDS = ['bhutan', 'nepal', 'bangladesh', 'godda', 'actual', 'day peak', 'mu', 'mw', 'transnational']
    REQUIRED_COLUMNS = [
        "country", "energy_actual_mu", "power_day_peak_mw", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of transnational summary data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each country/entity (Bhutan, Nepal, Bangladesh, Godda -> Bangladesh).
2. For each country/entity, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.
6. **IMPORTANT: Extract both energy (MU) and power (MW) data for each country.**

**Target JSON Schema:**
- `country`: (String) Country/entity name (Bhutan, Nepal, Bangladesh, Godda -> Bangladesh).
- `energy_actual_mu`: (Number) Actual energy in MU (from "Actual (MU)" row).
- `power_day_peak_mw`: (Number) Day peak power in MW (from "Day Peak (MW)" row).
- `report_date`: (String) "{report_date}"

**EXACT COLUMN MAPPING (based on actual data structure):**
The table has countries as columns and metrics as rows:
- Row 1: Country names (Bhutan, Nepal, Bangladesh, Godda -> Bangladesh)
- Row 2: "Actual (MU)" values for each country
- Row 3: "Day Peak (MW)" values for each country

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the transnational summary table."""
        if not super().validate(data):
            return False
        
        # Check for expected countries/entities
        expected_entities = {"Bhutan", "Nepal", "Bangladesh", "Godda -> Bangladesh"}
        found_entities = {record.get("country") for record in data}

        if not found_entities.intersection(expected_entities):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected countries/entities found.")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_actual_mu") is not None and not isinstance(record["energy_actual_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_actual_mu' is not a number for country {record.get('country')}.")
                return False
            
            if record.get("power_day_peak_mw") is not None and not isinstance(record["power_day_peak_mw"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'power_day_peak_mw' is not a number for country {record.get('country')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 