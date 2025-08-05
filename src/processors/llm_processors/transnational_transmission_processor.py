#!/usr/bin/env python3
"""
Transnational Transmission Processor

This processor handles the 'Transnational Transmission Line Flow' table type.
It extracts transnational transmission data and structures it for database storage.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class TransnationalTransmissionProcessor(BaseProcessor):
    """Processor for the 'Transnational Transmission Line Flow' table."""
    TABLE_TYPE = "transnational_transmission"
    KEYWORDS = ['transnational transmission', 'state', 'region', 'line_name', 'max', 'min', 'avg', 'energy exchange', 'transnational transmission line']
    REQUIRED_COLUMNS = [
        "state", "region", "line_name", "power_max_mw", "power_min_mw", "power_avg_mw", "energy_exchange_mu", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of transnational transmission data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each transnational transmission line.
2. For each line, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.
6. **IMPORTANT: When the State column is empty (NaN or blank), use the last non-empty state value from previous rows.**
7. **State propagation: BHUTAN, NEPAL, and BANGLADESH should be propagated to all their respective transmission lines.**

**Target JSON Schema:**
- `state`: (String) State name (propagate from previous rows if empty).
- `region`: (String) Region name.
- `line_name`: (String) Name of the transmission line.
- `power_max_mw`: (Number) Maximum power flow (MW).
- `power_min_mw`: (Number) Minimum power flow (MW).
- `power_avg_mw`: (Number) Average power flow (MW).
- `energy_exchange_mu`: (Number) Energy exchange (MU).
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the transnational transmission table."""
        if not super().validate(data):
            return False
        
        # Check for expected regions
        expected_regions = {"NR", "ER", "WR", "SR", "NER"}
        found_regions = {record.get("region") for record in data}

        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_exchange_mu") is not None and not isinstance(record["energy_exchange_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_exchange_mu' is not a number for line {record.get('line_name')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 