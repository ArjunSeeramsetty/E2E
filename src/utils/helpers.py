"""
Utility functions for the power supply data warehouse
"""

import os
import re
from typing import Optional
from datetime import datetime
from loguru import logger

def extract_date_from_filename(filename: str) -> Optional[str]:
    """Extract date from filename like '19.04.25_NLDC_PSP.pdf'"""
    # Look for date patterns like DD.MM.YY or DD.MM.YYYY
    date_pattern = r'(\d{2})\.(\d{2})\.(\d{2,4})'
    match = re.search(date_pattern, filename)
    
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = f"20{year}"
        return f"{year}-{month}-{day}"
    
    return None

def extract_source_entity_from_filename(filename: str) -> Optional[str]:
    """Extract source entity from filename"""
    filename_lower = filename.lower()
    
    if "nldc" in filename_lower:
        return "NLDC"
    elif "srl" in filename_lower:
        return "SRLDC"
    elif "nrl" in filename_lower:
        return "NRLDC"
    elif "wrl" in filename_lower:
        return "WRLDC"
    elif "erl" in filename_lower:
        return "ERLDC"
    elif "nerl" in filename_lower:
        return "NERLDC"
    
    return None

def ensure_directory_exists(directory_path: str):
    """Ensure directory exists, create if it doesn't"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def clean_numeric_string(value: str) -> Optional[str]:
    """Clean numeric string for conversion"""
    if not value or value.strip() == "" or value.strip() == "-":
        return None
    
    # Remove common non-numeric characters except decimal point and minus
    cleaned = re.sub(r'[^\d.-]', '', value.strip())
    return cleaned if cleaned else None 