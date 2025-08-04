#!/usr/bin/env python3
"""
Regional Import Export Summary Processor

This processor handles the 'Regional Import Export Summary' table type.
It extracts regional import/export summary data with regions and their energy metrics.
"""

from ..base_processor import BaseProcessor
from typing import List, Dict
from loguru import logger

class RegionalImportExportSummaryProcessor(BaseProcessor):
    """Processor for the 'Regional Import Export Summary' table."""
    TABLE_TYPE = "regional_import_export_summary"
    KEYWORDS = ['nr', 'wr', 'sr', 'er', 'ner', 'total', 'schedule', 'actual', 'o/d/u/d', 'mu', 'regional', 'import', 'export']
    REQUIRED_COLUMNS = [
        "region", "energy_schedule_mu", "energy_actual_mu", "energy_overdrawal_mu", "report_date"
    ]
    PROMPT_TEMPLATE = """
You are an expert data extraction agent. Your task is to analyze the following raw table text of regional import/export summary data and convert it into a structured JSON array. Adhere strictly to the specified format.

**Raw Table Text:**
```
{input_table}
```

**Instructions:**
1. Parse the text to identify data for each region (NR, WR, SR, ER, NER, TOTAL).
2. For each region, create a JSON object with the following keys.
3. Use the provided `report_date` for all records.
4. If a value is missing or cannot be determined, use `null`.
5. Ensure all numerical values are cleaned and converted to numbers.
6. **IMPORTANT: Extract all energy metrics for each region.**

**Target JSON Schema:**
- `region`: (String) Region name (NR, WR, SR, ER, NER, TOTAL).
- `energy_schedule_mu`: (Number) Scheduled energy in MU (from "Schedule(MU)" row).
- `energy_actual_mu`: (Number) Actual energy in MU (from "Actual(MU)" row).
- `energy_overdrawal_mu`: (Number) Overdrawal energy in MU (from "O/D/U/D(MU)" row - positive - Overdrawal/ negative - Underdrawal).
- `report_date`: (String) "{report_date}"

**EXACT COLUMN MAPPING (based on actual data structure):**
The table has regions as columns and metrics as rows:
- Row 1: Region names (NR, WR, SR, ER, NER, TOTAL)
- Row 2: "Schedule(MU)" values for each region
- Row 3: "Actual(MU)" values for each region
- Row 4: "O/D/U/D(MU)" values for each region (Overdraw/Underdraw)

**Output ONLY the final JSON array. Do not include any explanations or surrounding text.**
"""

    def validate(self, data: List[Dict]) -> bool:
        """Stricter validation for the regional import/export summary table."""
        if not super().validate(data):
            return False
        
        # Check for expected regions
        expected_regions = {"NR", "WR", "SR", "ER", "NER", "TOTAL"}
        found_regions = {record.get("region") for record in data}

        if not found_regions.intersection(expected_regions):
            logger.warning(f"Validation warning for {self.TABLE_TYPE}: No expected regions found.")

        # Check for plausible numeric values
        for record in data:
            if record.get("energy_schedule_mu") is not None and not isinstance(record["energy_schedule_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_schedule_mu' is not a number for region {record.get('region')}.")
                return False
            
            if record.get("energy_actual_mu") is not None and not isinstance(record["energy_actual_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_actual_mu' is not a number for region {record.get('region')}.")
                return False
            
            if record.get("energy_overdrawal_mu") is not None and not isinstance(record["energy_overdrawal_mu"], (int, float)):
                logger.error(f"Validation failed for {self.TABLE_TYPE}: 'energy_overdrawal_mu' is not a number for region {record.get('region')}.")
                return False

        logger.info(f"Strict validation passed for {self.TABLE_TYPE}.")
        return True 