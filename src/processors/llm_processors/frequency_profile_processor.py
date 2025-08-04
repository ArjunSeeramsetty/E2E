#!/usr/bin/env python3
"""
Frequency Profile Processor

This processor handles the 'Frequency Profile' table type.
It extracts frequency profile data and structures it for database storage.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class FrequencyProfileProcessor(BaseProcessor):
    """Processor for the 'Frequency Profile' table."""
    TABLE_TYPE = "frequency_profile"
    KEYWORDS = [
        'fvi', '49.7', '49.8', '49.9', '50.05', 'frequency variation index', 'frequency_variation_index'
    ]
    REQUIRED_COLUMNS = [
        "frequency_band_hz", "percentage_pct", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of frequency profile data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each frequency band.
2. For each frequency band, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.

**Target JSON Schema:**
- `frequency_band_hz`: (String) Frequency band range or FVI (e.g., "FVI", "<49.7 Hz", "49.7-49.8 Hz", ">50.05").
- `percentage_pct`: (Number) Percentage of time for bands and index value for FVI
- `report_date`: (String) "{report_date}"

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the frequency profile table."""
        if not super().validate(data):
            return False
        
        # Check for expected frequency bands
        expected_bands = {"<49.7 Hz", "49.7-49.8 Hz", "49.8-49.9 Hz", "<49.9 Hz", "49.9-50.05 Hz", ">50.05 Hz", "FVI"}
        found_bands = {record.get("frequency_band_hz") for record in data}

        if not found_bands.intersection(expected_bands):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected frequency bands found.")

        # Check for plausible numeric values
        for record in data:
            if record.get("percentage_pct") is not None and not isinstance(record["percentage_pct"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'percentage_pct' is not a number for band {record.get('frequency_band_hz')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 