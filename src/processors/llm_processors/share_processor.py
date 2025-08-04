#!/usr/bin/env python3
"""
Share Processor

This processor handles the 'Share' table type.
It extracts share data with measures and regions.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class ShareProcessor(BaseProcessor):
    """Processor for the 'Share' table."""
    TABLE_TYPE = "share"
    KEYWORDS = ['share', 'res', 'non-fossil', 'hydro', 'nuclear', 'generation', 'nr', 'wr', 'sr', 'er', 'ner', 'all india']
    REQUIRED_COLUMNS = [
        "measure", "region_name", "share_percentage", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of share data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each measure and region combination.
2. For each measure-region combination, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.
6. **IMPORTANT: Extract all share percentages for each measure and region.**

**Target JSON Schema:**
- `region_name`: (String) Region name (NR, WR, SR, ER, NER, ALL India).
- `measure`: (String) Measure name (e.g., "Share of RES in total generation (%)", "Share of Non-fossil fuel (Hydro,Nuclear and RES) in total generation(%)").
- `share_percentage`: (Number) Share percentage value.
- `report_date`: (String) "{report_date}"

**EXACT COLUMN MAPPING (based on actual data structure):**
The table has measures as rows and regions as columns:
- Column 1: Measure names
- Column 2: NR values
- Column 3: WR values
- Column 4: SR values
- Column 5: ER values
- Column 6: NER values
- Column 7: ALL India values

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the share table."""
        if not super().validate(data):
            return False
        
        # Check for expected regions
        expected_regions = {"NR", "WR", "SR", "ER", "NER", "ALL India"}
        found_regions = {record.get("region_name") for record in data}

        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")

        # Check for expected measures
        expected_measures = {
            "Share of RES in total generation (%)",
            "Share of Non-fossil fuel (Hydro,Nuclear and RES) in total generation(%)"
        }
        found_measures = {record.get("measure") for record in data}

        if not found_measures.intersection(expected_measures):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected measures found.")

        # Check for plausible numeric values (percentages should be between 0 and 100)
        for record in data:
            if record.get("share_percentage") is not None:
                if not isinstance(record["share_percentage"], (int, float)):
                    logger.error(f"Validation failed for {self.TABLE_TYPE}: 'share_percentage' is not a number for measure {record.get('measure')} and region {record.get('region')}.")
                    return False
                
                # Check if percentage is within reasonable range (0-100)
                if record["share_percentage"] < 0 or record["share_percentage"] > 100:
                    logger.warning(f"Validation warning for {self.TABLE_TYPE}: 'share_percentage' {record['share_percentage']} is outside expected range (0-100) for measure {record.get('measure')} and region {record.get('region')}.")

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 