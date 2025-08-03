#!/usr/bin/env python3
"""
National Power Supply Data Warehouse - Main Extractor
Simplified version based on the working modular_psp_parser.py approach
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import re
import tempfile
import subprocess
import shutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import functools
import io

# Import tabula directly without any JVM configuration
import tabula

# Import other required libraries
import fitz  # PyMuPDF
import PyPDF2
from PIL import Image
import cv2
import numpy as np

# Add LLM imports and configuration
import json
import requests
from typing import Dict, List, Any, Optional, Tuple
import time

# Import the state table processing module
from src.processors.raw_state_table_processing import (
    process_state_table_robustly,
    validate_state_table_data,
    get_state_table_summary
)

# LLM Configuration
LLM_ENDPOINT = "http://localhost:11434/api/generate"  # Ollama endpoint
LLM_MODEL = "llama3.2:3b"  # Using available model
LLM_TIMEOUT = 300  # Increased timeout for complex tables
LLM_MAX_RETRIES = 3

# Table type detection patterns
TABLE_TYPE_PATTERNS = {
    'regional_daily_summary': [
        'demand met during evening peak', 'peak shortage', 'energy met', 'hydro gen', 'wind gen', 'solar gen', 'energy shortage', 'maximum demand met during the day', 'time of maximum demand met'
    ],
    'state_daily_summary': [
        'state_name', 'region_code', 'max_demand_met_day_mw', 'state_max_demand_met_mw', 'energy_met_total_mu', 'state_drawal_schedule_mu'
    ],
    'generation_by_source': [
        'source', 'sourcewise', 'gross generation', 'fuel type', 'coal', 'hydro', 'nuclear', 'thermal'
    ],
    'generation_outages': [
        'central sector', 'state sector', 'total', 'res_share_total_gen_pct'
    ],
    'inter_regional_transmission': [
        'inter-regional', 'voltage_level', 'line_details', 'sl_no', 'no_of_circuit', 'max import', 'max export', 'import', 'net'
    ],
    'transnational_transmission': [
        'transnational', 'state', 'region', 'line_name', 'max', 'min', 'avg', 'energy exchange'
    ],
    'frequency_profile': [
        'frequency_variation_index', 'frequency variation index', '49_7', '49_8', '49_9', '50_05', 'fvi'
    ],
    'solar_non_solar_peak': [
        'solar hr', 'non-solar hr', 'max demand met', 'peak demand', 'solar hour', 'non-solar hour'
    ],
    'scada_timeseries': [
        'scada', 'time', 'timestamp', 'instantaneous', '15-minute', 'all india scada', 'frequency', 'demand'
    ]
}

# One-shot prompts for each table type
TABLE_PROMPTS = {
    'regional_daily_summary': """
You are a data processing expert. Convert the following table into a structured JSON format for Regional Power Supply Summary.

Input Table:
{input_table}

IMPORTANT: This table may have a confusing structure where the first column contains labels like "Schedule(MU)", "Actual(MU)", "O/D/U/D(MU)" and the actual data is in subsequent columns. You need to:

1. Identify the data structure - look for row labels vs column headers
2. Extract the actual numerical data, not the labels
3. Map the data correctly to the expected fields

Expected Output Format (JSON):
{{
  "region_code": "Region abbreviation (NR, WR, SR, ER, NER, ALL_INDIA)",
  "peak_demand_met_evng_mw": "Peak demand met during evening peak (MW) - number",
  "peak_shortage_evng_mw": "Peak shortage during evening peak (MW) - number", 
  "energy_met_total_mu": "Total energy met (MU) - number",
  "energy_shortage_total_mu": "Total energy shortage (MU) - number",
  "generation_hydro_mu": "Hydro generation (MU) - number",
  "generation_wind_mu": "Wind generation (MU) - number", 
  "generation_solar_mu": "Solar generation (MU) - number",
  "max_demand_met_day_mw": "Maximum demand met during day (MW) - number",
  "time_of_max_demand_day": "Time of maximum demand (string like '14:30')",
  "report_date": "2025-04-19"
}}

Rules:
1. DO NOT use row labels like "Schedule(MU)", "Actual(MU)" as data values
2. Output ONLY valid JSON - no explanations or additional text
3. Use null for missing values
4. Convert all numbers to actual numeric values, not strings
2. Extract actual numerical data from the table cells
3. Convert all numeric values to appropriate units
4. Standardize region codes
5. Use "2025-04-19" as default report date if not found
6. Handle missing values as null
7. Return ONLY valid JSON format

Output:
""",

    'state_daily_summary': """
You are a data processing expert. Convert the following table into a structured JSON format for State-wise Power Supply Position.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
  "region_code": "Region abbreviation (NR, WR, SR, ER, NER)",
  "state_name": "State or Union Territory name",
  "power_max_demand_met_day_mw": "Maximum demand met during day (MW) - number",
  "power_shortage_at_max_demand_mw": "Shortage at maximum demand (MW) - number",
  "energy_met_mu": "Energy met (MU) - number",
  "energy_drawal_schedule_mu": "Scheduled drawal (MU) - number",
  "energy_over_under_drawal_mu": "Over/Under drawal (MU) - number",
  "power_max_overdrawal_mw": "Maximum overdrawal (MW) - number",
  "energy_shortage_mu": "Energy shortage (MU) - number",
  "report_date": "2025-04-19"
}}

Rules:
1. Standardize state names
2. Convert numeric values to actual numbers, not strings
3. Handle positive/negative values for over/under drawal
4. Use null for missing values
5. Output ONLY valid JSON - no explanations or additional text
6. Use "2025-04-19" as default report date

Output ONLY the JSON object. No explanations or additional text.
""",

    'generation_by_source': """
You are a data processing expert. Convert the following table into a structured format for Source-wise Generation.

Input Table:
{input_table}

Expected Output Format:
- region_code: Region code
- source_type: Fuel/source type (COAL, LIGNITE, HYDRO, NUCLEAR, GAS_NAPTHA_DIESEL, RES)
- gross_generation_mu: Gross generation (MU)
- res_share_total_gen_pct: RES share in total generation (%)
- non_fossil_share_total_gen_pct: Non-fossil share in total generation (%)
- report_date: Report date (YYYY-MM-DD format)

Rules:
1. Standardize source types
2. Convert percentages to decimal format
3. Calculate RES and non-fossil shares if not provided
4. Extract date from table or use provided date
5. Return JSON format only

Output:
""",

    'generation_outages': """
You are a data processing expert. Convert the following table into a structured format for Generation Outages.

Input Table:
{input_table}

Expected Output Format:
- region_code: Region code
- sector: Sector type (CENTRAL, STATE)
- outage_mw: Outage capacity (MW)
- report_date: Report date (YYYY-MM-DD format)

Rules:
1. Standardize sector names
2. Convert outage values to MW
3. Extract date from table or use provided date
4. Return JSON format only

Output:
""",

    'inter_regional_transmission': """
You are a data processing expert. Convert the following table into a structured format for Inter-regional Transmission Line Flow.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
  "sl_no": "Serial number (number)",
  "voltage_level": "Voltage level (e.g., '765 kV', '400 kV', 'HVDC')",
  "line_details": "Line name/details (string)",
  "no_of_circuit": "Number of circuits (number)",
  "power_max_import_mw": "Maximum Import (MW) - number",
  "power_max_export_mw": "Maximum Export (MW) - number",
  "energy_import_mu": "Import energy (MU) - number",
  "energy_net_mu": "Net energy flow (MU) - number",
  "report_date": "2025-04-19"
}}

Rules:
1. Extract serial numbers, voltage levels, and line details
2. Convert all power values to MW and energy values to MU
3. Handle missing values as null
4. Use "2025-04-19" as default report date
5. Output ONLY valid JSON - no explanations or additional text
6. Convert all numbers to actual numeric values, not strings

Output ONLY the JSON array. No explanations or additional text.
""",

    'transnational_transmission': """
You are a data processing expert. Convert the following table into a structured format for Transnational Transmission Line Flow.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
  "state": "State name (string)",
  "region": "Region name (string)",
  "line_name": "Transmission line name (string)",
  "power_max_mw": "Maximum power flow (MW) - number",
  "power_min_mw": "Minimum power flow (MW) - number",
  "power_avg_mw": "Average power flow (MW) - number",
  "energy_exchange_mu": "Energy exchange (MU) - number",
  "report_date": "2025-04-19"
}}

Rules:
1. Extract state, region, and line information
2. Convert all power values to MW and energy values to MU
3. Handle missing values as null
4. Use "2025-04-19" as default report date
5. Output ONLY valid JSON - no explanations or additional text
6. Convert all numbers to actual numeric values, not strings

Output ONLY the JSON array. No explanations or additional text.
""",

    'frequency_profile': """
You are a data processing expert. Convert the following table into a structured format for Frequency Profile.

Input Table:
{input_table}

Expected Output Format:
- frequency_band_hz: Frequency band (e.g., "LT_49.7", "49.9_TO_50.05", "GT_50.05")
- duration_pct: Duration percentage (%)
- frequency_variation_index: Frequency Variation Index (FVI)
- report_date: Report date (YYYY-MM-DD format)

Rules:
1. Standardize frequency band descriptions
2. Convert percentages to decimal format
3. Calculate FVI if not provided
4. Extract date from table or use provided date
5. Return JSON format only

Output:
""",

    'solar_non_solar_peak': """
You are a data processing expert. Convert the following table into a structured format for Solar/Non-Solar Hour Peak Demand.

Input Table:
{input_table}

Expected Output Format (JSON):
{{
  "period_type": "Period type ('Solar hr' or 'Non-Solar hr')",
  "max_demand_met_mw": "Maximum demand met (MW) - number",
  "timestamp": "Time of peak demand (HH:MM format)",
  "shortage_at_max_demand_mw": "Shortage at maximum demand (MW) - number",
  "report_date": "2025-04-19"
}}

Rules:
1. Extract period type (Solar hr, Non-Solar hr)
2. Convert all power values to MW
3. Handle missing values as null
4. Use "2025-04-19" as default report date
5. Output ONLY valid JSON - no explanations or additional text
6. Convert all numbers to actual numeric values, not strings

Output ONLY the JSON array. No explanations or additional text.
""",

    'scada_timeseries': """
You are a data processing expert. Convert the following table into a structured format for SCADA Time-series Data.

Input Table:
{input_table}

Expected Output Format:
- timestamp: Timestamp (YYYY-MM-DD HH:MM:SS format)
- frequency_hz: Grid frequency (Hz)
- demand_met_mw: Demand met (MW)
- gen_nuclear_mw: Nuclear generation (MW)
- gen_wind_mw: Wind generation (MW)
- gen_solar_mw: Solar generation (MW)
- gen_hydro_mw: Hydro generation (MW)
- gen_gas_mw: Gas generation (MW)
- gen_thermal_mw: Thermal generation (MW)
- gen_others_mw: Other generation (MW)
- gen_total_mw: Total generation (MW)
- net_transnational_exchange_mw: Net transnational exchange (MW)

Rules:
1. Parse timestamp from time column
2. Convert all generation values to MW
3. Handle missing values as null
4. Calculate total generation if not provided
5. Return JSON format only

Output:
"""
}

# Configure logging
from loguru import logger

# Remove any existing handlers and add our own
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/main_extractor.log",
    rotation="10 MB",
    retention="7 days",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG"
)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Configuration
INPUT_PDF_PATH = r"C:\Users\arjun\Desktop\E2E\data\raw\19.04.25_NLDC_PSP.pdf"
STANDARD_OUTPUT_DIR = "data/processed/standard_extraction"
ENHANCED_OUTPUT_DIR = "data/processed/enhanced_extraction"

# Create output directories
Path(STANDARD_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(ENHANCED_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# GPU Configuration
USE_GPU = True
MAX_WORKERS = 4
GPU_DEVICE = 0

# Comprehensive TARGET_SCHEMA for column standardization
TARGET_SCHEMA = {
    # Regional Power Supply Summary
    'region_code': ['region', 'region code', 'region_code', 'nr', 'wr', 'sr', 'er', 'ner', 'all india'],
    'power_peak_demand_met_evng_mw': ['peak demand met', 'demand met during evening peak hrs', 'peak demand met (mw)', 'demand met'],
    'power_peak_shortage_evng_mw': ['peak shortage', 'peak shortage (mw)', 'shortage during peak'],
    'energy_met_total_mu': ['energy met', 'energy met (mu)', 'total energy met'],
    'energy_shortage_total_mu': ['energy shortage', 'energy shortage (mu)', 'total energy shortage'],
    'generation_hydro_mu': ['hydro', 'hydro generation', 'hydro (mu)', 'generation hydro'],
    'generation_wind_mu': ['wind', 'wind generation', 'wind (mu)', 'generation wind'],
    'generation_solar_mu': ['solar', 'solar generation', 'solar (mu)', 'generation solar'],
    'max_demand_met_day_mw': ['maximum demand met during the day', 'max demand scada', 'max demand met'],
    'time_of_max_demand_day': ['time of maximum demand met', 'time of max demand met', 'max demand time'],
    'report_date': ['date', 'report date', 'date of report'],
    
    # State-wise Power Supply Position
    'state_name': ['states', 'state', 'state name', 'state/ut'],
    'state_max_demand_met_mw': ['maximum demand', 'max demand', 'max.demand', 'maximumdemand', 'maximum demand (mw)'],
    'state_shortage_at_max_demand_mw': ['shortage', 'shortage (mw)', 'shortage during', 'energy shortage', 'energy shortage (mu)'],
    'state_energy_met_mu': ['energy met', 'energy met (mu)', 'energymet'],
    'state_drawal_schedule_mu': ['drawal schedule', 'schedule (mu)', 'drawal\rSchedule', 'drawalschedule'],
    'state_over_under_drawal_mu': ['od/ud', 'over under drawal', 'od(+)/ud(-)', 'overunderdrawal', 'od(+)/ud(-) (mu)', 'o/d/u/d(mu)'],
    'state_max_overdrawal_mw': ['max od', 'max over drawal', 'max od\r(mw)', 'maxoverdrawal', 'max od (mw)'],
    'state_energy_shortage_mu': ['energy shortage', 'energy shortage (mu)', 'energyshortage'],
    
    # Source-wise Generation
    'source_type': ['source', 'source type', 'fuel type', 'generation source'],
    'gross_generation_mu': ['gross generation', 'generation', 'gross generation (mu)', 'total generation'],
    'res_share_total_gen_pct': ['share of res in total generation', 'res share', 'renewable share', 're share'],
    'non_fossil_share_total_gen_pct': ['share of non-fossil', 'non-fossil share', 'non fossil share'],
    
    # Generation Outage
    'sector': ['sector', 'central sector', 'state sector', 'sector type'],
    'outage_mw': ['outage', 'outage (mw)', 'generation outage', 'outage capacity'],
    
    # Power Exchanges
    'exchange_type': ['exchange type', 'type', 'exchange category'],
    'source_entity': ['source', 'source entity', 'from', 'exporting entity'],
    'destination_entity': ['destination', 'destination entity', 'to', 'importing entity'],
    'scheduled_exchange_mu': ['schedule', 'scheduled', 'schedule(mu)', 'scheduled exchange'],
    'actual_exchange_mu': ['actual', 'actual(mu)', 'actual exchange'],
    'deviation_mu': ['deviation', 'deviation (mu)', 'difference'],
    
    # Frequency Profile
    'frequency_band_hz': ['frequency band', 'frequency range', 'frequency (hz)', 'band'],
    'duration_pct': ['duration', 'duration (%)', 'time duration', 'percentage'],
    'frequency_variation_index': ['fvi', 'frequency violation index', 'frequency variation index'],
    
    # SCADA Time-series Data
    'timestamp': ['time', 'timestamp', 'date time', 'datetime'],
    'frequency_hz': ['frequency', 'frequency (hz)', 'grid frequency'],
    'power_demand_met_mw': ['demand met', 'demand met (mw)', 'total demand'],
    'power_nuclear_mw': ['nuclear', 'nuclear (mw)', 'nuclear generation'],
    'power_wind_mw': ['wind', 'wind (mw)', 'wind generation'],
    'power_solar_mw': ['solar', 'solar (mw)', 'solar generation'],
    'power_hydro_mw': ['hydro', 'hydro (mw)', 'hydro generation'],
    'power_gas_mw': ['gas', 'gas (mw)', 'gas generation', 'gas naptha diesel'],
    'power_thermal_mw': ['thermal', 'thermal (mw)', 'thermal generation', 'coal', 'lignite'],
    'power_others_mw': ['others', 'others (mw)', 'other generation'],
    'power_total_mw': ['total generation', 'total (mw)', 'total gen'],
    'power_net_transnational_exchange_mw': ['net transnational exchange', 'net exchange', 'transnational exchange']
}

def check_gpu_availability():
    """Check if GPU is available and configure it"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{GPU_DEVICE}')
            torch.cuda.set_device(device)
            
            # Optimize GPU memory allocation
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Set environment variables for better GPU performance
            os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
            os.environ['CUDA_CACHE_DISABLE'] = '0'
            
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(device)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
            return True
        else:
            logger.info("No GPU detected, using CPU")
            return False
    except ImportError:
        logger.warning("PyTorch not available, using CPU")
        return False
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return allocated, cached, total
        return 0, 0, 0
    except:
        return 0, 0, 0

def extract_tables_standard(pdf_path):
    """
    Extract tables from PDF using tabula-py with standard approach.
    Enhanced to capture all 16 expected tables including missing Import/Export and Transnational tables.
    Uses multiple strategies with smart deduplication.
    """
    logger.info(f"Running Standard Extraction on {pdf_path}")
    
    try:
        raw_tables = {}
        reader = PyPDF2.PdfReader(pdf_path)
        num_pages = len(reader.pages)
        
        for page_num in range(1, num_pages + 1):
            max_retries = 3
            for retry in range(max_retries):
                try:
                    # Enhanced extraction strategy for each page
                    if page_num == 1:  # First page - Regional summary tables
                        # Try multiple strategies for page 1
                        page_tables = []
                        
                        # Strategy 1: Standard extraction
                        try:
                            tables_std = tabula.read_pdf(
                                pdf_path,
                                pages=page_num,
                                multiple_tables=True,
                                guess=True,
                                lattice=True,
                                stream=True,
                                silent=True
                            )
                            page_tables.extend(tables_std)
                        except:
                            pass
                        
                        # Strategy 2: Lattice extraction
                        try:
                            tables_lattice = tabula.read_pdf(
                                pdf_path,
                                pages=page_num,
                                multiple_tables=True,
                                guess=True,
                                lattice=True,
                                stream=False,
                                silent=True
                            )
                            page_tables.extend(tables_lattice)
                        except:
                            pass
                        
                        # Strategy 3: Stream extraction
                        try:
                            tables_stream = tabula.read_pdf(
                                pdf_path,
                                pages=page_num,
                                multiple_tables=True,
                                guess=True,
                                lattice=False,
                                stream=True,
                                silent=True
                            )
                            page_tables.extend(tables_stream)
                        except:
                            pass
                        
                        # Strategy 4: Area-specific extraction for Import/Export tables
                        try:
                            # Try different areas of the page to capture Import/Export tables
                            areas = [
                                [0, 0, 50, 100],   # Top half
                                [50, 0, 100, 100],  # Bottom half
                                [0, 0, 100, 50],    # Left half
                                [0, 50, 100, 100]   # Right half
                            ]
                            
                            for area in areas:
                                try:
                                    tables_area = tabula.read_pdf(
                                        pdf_path,
                                        pages=page_num,
                                        multiple_tables=True,
                                        guess=True,
                                        lattice=True,
                                        stream=True,
                                        silent=True,
                                        area=area
                                    )
                                    page_tables.extend(tables_area)
                                except:
                                    pass
                        except:
                            pass
                        
                        tables_on_page = page_tables
                        
                    elif page_num == 2:  # Second page - State-wise data and Import/Export tables
                        # Try multiple strategies for page 2
                        page_tables = []
                        
                        # Strategy 1: Standard extraction
                        try:
                            tables_std = tabula.read_pdf(
                                pdf_path,
                                pages=page_num,
                                multiple_tables=True,
                                guess=True,
                                lattice=True,
                                stream=True,
                                silent=True
                            )
                            page_tables.extend(tables_std)
                        except:
                            pass
                        
                        # Strategy 2: Area-specific extraction for Import/Export tables
                        try:
                            # Try different areas to capture Import/Export tables
                            areas = [
                                [0, 0, 50, 100],   # Top half
                                [50, 0, 100, 100],  # Bottom half
                                [0, 0, 100, 50],    # Left half
                                [0, 50, 100, 100]   # Right half
                            ]
                            
                            for area in areas:
                                try:
                                    tables_area = tabula.read_pdf(
                                        pdf_path,
                                        pages=page_num,
                                        multiple_tables=True,
                                        guess=True,
                                        lattice=True,
                                        stream=True,
                                        silent=True,
                                        area=area
                                    )
                                    page_tables.extend(tables_area)
                                except:
                                    pass
                        except:
                            pass
                        
                        tables_on_page = page_tables
                        
                    elif page_num == 3:  # Third page - International exchanges and Transnational tables
                        # Try multiple strategies for page 3
                        page_tables = []
                        
                        # Strategy 1: Standard extraction
                        try:
                            tables_std = tabula.read_pdf(
                                pdf_path,
                                pages=page_num,
                                multiple_tables=True,
                                guess=True,
                                lattice=True,
                                stream=True,
                                silent=True
                            )
                            page_tables.extend(tables_std)
                        except:
                            pass
                        
                        # Strategy 2: Area-specific extraction for Transnational tables
                        try:
                            # Try different areas to capture Transnational tables
                            areas = [
                                [0, 0, 50, 100],   # Top half
                                [50, 0, 100, 100],  # Bottom half
                                [0, 0, 100, 50],    # Left half
                                [0, 50, 100, 100]   # Right half
                            ]
                            
                            for area in areas:
                                try:
                                    tables_area = tabula.read_pdf(
                                        pdf_path,
                                        pages=page_num,
                                        multiple_tables=True,
                                        guess=True,
                                        lattice=True,
                                        stream=True,
                                        silent=True,
                                        area=area
                                    )
                                    page_tables.extend(tables_area)
                                except:
                                    pass
                        except:
                            pass
                        
                        tables_on_page = page_tables
                        
                    elif page_num == 5:  # Fifth page - Blockwise data (special handling)
                        # Try multiple strategies for page 5
                        page_tables = []
                        
                        # Strategy 1: Standard extraction
                        try:
                            tables_std = tabula.read_pdf(
                                pdf_path,
                                pages=page_num,
                                multiple_tables=True,
                                guess=False,
                                lattice=True,
                                stream=False,
                                silent=True,
                                java_options=["-Dfile.encoding=UTF8", "-Xmx2g"]
                            )
                            page_tables.extend(tables_std)
                        except:
                            pass
                        
                        # Strategy 2: Stream extraction
                        try:
                            tables_stream = tabula.read_pdf(
                                pdf_path,
                                pages=page_num,
                                multiple_tables=True,
                                guess=True,
                                lattice=False,
                                stream=True,
                                silent=True
                            )
                            page_tables.extend(tables_stream)
                        except:
                            pass
                        
                        # Strategy 3: Lattice extraction
                        try:
                            tables_lattice = tabula.read_pdf(
                                pdf_path,
                                pages=page_num,
                                multiple_tables=True,
                                guess=True,
                                lattice=True,
                                stream=False,
                                silent=True
                            )
                            page_tables.extend(tables_lattice)
                        except:
                            pass
                        
                        tables_on_page = page_tables
                        
                    else:
                        # Default strategy for other pages
                        tables_on_page = tabula.read_pdf(
                            pdf_path,
                            pages=page_num,
                            multiple_tables=True,
                            guess=True,
                            lattice=True,
                            stream=True,
                            silent=True
                        )
                    
                    # Process extracted tables with very lenient filtering
                    for table_idx, table_df in enumerate(tables_on_page):
                        if isinstance(table_df, pd.DataFrame) and not table_df.empty:
                            # Accept any table with at least 1 row and 1 column
                            if table_df.shape[0] > 0 and table_df.shape[1] > 0:
                                # Check if table has any meaningful content (at least 1 non-empty cell)
                                non_empty_cells = table_df.notna().sum().sum()
                                if non_empty_cells > 0:
                                    key = f"page_{page_num}_table_{table_idx}"
                                    raw_tables[key] = table_df
                                    logger.debug(f"Added table from page {page_num}: {table_df.shape[0]}x{table_df.shape[1]} with {non_empty_cells} non-empty cells")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num} (attempt {retry + 1}/{max_retries}): {e}")
                    if retry < max_retries - 1:
                        import time
                        time.sleep(1)
                    else:
                        logger.error(f"Failed to extract tables from page {page_num} after {max_retries} attempts")
        
        # Smart deduplication: Keep the best version of each table
        logger.info(f"Before deduplication: {len(raw_tables)} tables")
        deduplicated_tables = smart_deduplicate_tables(raw_tables)
        logger.info(f"After smart deduplication: {len(deduplicated_tables)} tables")
        
        # Convert to list format expected by the rest of the pipeline
        tables_list = list(deduplicated_tables.values())
        logger.info(f"Standard extraction completed: {len(tables_list)} tables found")
        return tables_list
        
    except Exception as e:
        logger.error(f"Standard extraction failed: {e}")
        return []

def smart_deduplicate_tables(raw_tables):
    """
    Smart deduplication that keeps the best version of each table.
    Compares tables based on content similarity and keeps the most complete version.
    """
    if not raw_tables:
        return {}
    
    # Group tables by content similarity
    table_groups = {}
    
    for key, table in raw_tables.items():
        # Create a signature for the table based on its content and structure
        signature = create_table_signature(table)
        
        if signature not in table_groups:
            table_groups[signature] = []
        table_groups[signature].append((key, table))
    
    # For each group, keep the best table
    deduplicated = {}
    
    for signature, group in table_groups.items():
        if len(group) == 1:
            # Only one table in group, keep it
            key, table = group[0]
            deduplicated[key] = table
        else:
            # Multiple similar tables, keep the best one
            best_key, best_table = select_best_table(group)
            deduplicated[best_key] = best_table
            logger.debug(f"Kept best table from group of {len(group)} similar tables")
    
    return deduplicated

def create_table_signature(table):
    """
    Create a signature for a table based on its content and structure.
    This helps identify similar tables.
    """
    # Get table dimensions
    rows, cols = table.shape
    
    # Get column headers (first row)
    headers = []
    if not table.empty:
        headers = [str(col).strip().lower() for col in table.columns]
    
    # Get a sample of data (first few rows)
    data_sample = []
    if not table.empty and rows > 1:
        for i in range(min(3, rows)):
            row_data = []
            for j in range(min(5, cols)):
                if i < len(table) and j < len(table.columns):
                    cell_value = str(table.iloc[i, j]).strip()
                    row_data.append(cell_value)
            data_sample.append('|'.join(row_data))
    
    # Create signature
    signature = f"{rows}x{cols}_{'_'.join(headers[:5])}_{'_'.join(data_sample)}"
    return signature

def select_best_table(table_group):
    """
    Select the best table from a group of similar tables.
    Criteria: most complete data, best formatting, largest size.
    """
    best_score = -1
    best_key = None
    best_table = None
    
    for key, table in table_group:
        score = calculate_table_quality_score(table)
        if score > best_score:
            best_score = score
            best_key = key
            best_table = table
    
    return best_key, best_table

def calculate_table_quality_score(table):
    """
    Calculate a quality score for a table.
    Higher score = better table.
    """
    if table.empty:
        return 0
    
    score = 0
    
    # Size score (larger tables get higher scores)
    rows, cols = table.shape
    score += rows * cols * 0.1
    
    # Completeness score (more non-empty cells = better)
    non_empty_cells = table.notna().sum().sum()
    total_cells = rows * cols
    if total_cells > 0:
        completeness_ratio = non_empty_cells / total_cells
        score += completeness_ratio * 100
    
    # Header quality score (better headers = better table)
    if not table.empty:
        headers = [str(col).strip() for col in table.columns]
        # Count headers that look like proper column names
        good_headers = sum(1 for h in headers if len(h) > 0 and h != 'unnamed' and 'unnamed' not in h.lower())
        score += good_headers * 10
    
    # Data quality score (numeric data is often better)
    numeric_columns = 0
    for col in table.columns:
        if not table.empty:
            # Check if column contains numeric data
            numeric_count = 0
            for val in table[col].dropna():
                try:
                    float(str(val))
                    numeric_count += 1
                except:
                    pass
            if numeric_count > len(table[col]) * 0.5:  # More than 50% numeric
                numeric_columns += 1
    
    score += numeric_columns * 5
    
    return score

def extract_tables_standard_parallel(pdf_path):
    """
    Extract tables from PDF using tabula-py with parallel processing of page ranges.
    Enhanced to capture all 16 expected tables without deduplication.
    """
    logger.info(f"Running Parallel Standard Extraction on {pdf_path}")
    
    try:
        # Get total pages for parallel processing
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        logger.info(f"PDF has {total_pages} pages. Processing page ranges in parallel...")
        
        # Create page ranges for parallel processing - smaller chunks for better extraction
        chunk_size = max(1, total_pages // 5)  # Process in 5 chunks for better granularity
        page_ranges = []
        
        for i in range(0, total_pages, chunk_size):
            start_page = i + 1  # tabula-py uses 1-based page numbers
            end_page = min(i + chunk_size, total_pages)
            range_name = f"Range_{i//chunk_size + 1}"
            page_ranges.append((pdf_path, start_page, end_page, range_name))
        
        logger.info(f"Created {len(page_ranges)} page ranges for parallel processing")
        
        # Process page ranges in parallel using ThreadPoolExecutor
        all_tables = []
        with ThreadPoolExecutor(max_workers=5) as executor:  # Increased workers
            # Submit all page range processing tasks
            future_to_range = {
                executor.submit(extract_tables_from_page_range, args): args[3] 
                for args in page_ranges
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_range):
                range_name = future_to_range[future]
                try:
                    tables = future.result()
                    all_tables.extend(tables)
                    logger.info(f"Completed {range_name}: {len(tables)} tables")
                except Exception as e:
                    logger.error(f"Error in {range_name}: {e}")
        
        logger.info(f"Total tables found across all ranges: {len(all_tables)}")
        
        # No deduplication - keep all tables as they are
        if all_tables:
            logger.info(f"Keeping all {len(all_tables)} tables without deduplication")
            
            # Sort tables by size (largest first) to prioritize important tables
            all_tables.sort(key=lambda x: x.shape[0] * x.shape[1], reverse=True)
            
            return all_tables
        else:
            logger.warning("No tables found with parallel standard extraction")
            return []
            
    except Exception as e:
        logger.error(f"Parallel standard extraction failed: {e}")
        return []

def extract_tables_from_page_range(args):
    """
    Extract tables from a specific page range using tabula-py.
    This is a worker function for parallel processing.
    Simplified approach to capture all 16 expected tables without JVM conflicts.
    """
    pdf_path, start_page, end_page, page_range_name = args
    
    try:
        # Import tabula (no JVM configuration needed)
        import tabula
        
        # Extract tables from the specific page range
        page_range = f"{start_page}-{end_page}" if start_page != end_page else str(start_page)
        
        logger.info(f"Processing {page_range_name} (pages {page_range})...")
        
        tables = []
        for page_num in range(start_page, end_page + 1):
            try:
                # Use a single, reliable strategy for all pages
                tables_on_page = tabula.read_pdf(
                    pdf_path,
                    pages=page_num,
                    multiple_tables=True,
                    guess=True,
                    lattice=True,
                    stream=True,
                    silent=True
                )
                
                # Accept all tables with any content
                for table in tables_on_page:
                    if isinstance(table, pd.DataFrame) and not table.empty:
                        # Accept any table with at least 1 row and 1 column
                        if table.shape[0] > 0 and table.shape[1] > 0:
                            # Check if table has any meaningful content (at least 1 non-empty cell)
                            non_empty_cells = table.notna().sum().sum()
                            if non_empty_cells > 0:
                                tables.append(table)
                                logger.debug(f"Added table from page {page_num}: {table.shape[0]}x{table.shape[1]} with {non_empty_cells} non-empty cells")
                        
            except Exception as e:
                logger.error(f"Error processing page {page_num} in {page_range_name}: {e}")
                continue
        
        logger.info(f"{page_range_name}: Found {len(tables)} valid tables")
        return tables
        
    except Exception as e:
        logger.error(f"Error processing {page_range_name}: {e}")
        return []

def extract_tables_enhanced_ocr(pdf_path):
    """
    Enhanced OCR extraction using Tesseract + tabula-py approach.
    """
    logger.info(f"Running Enhanced OCR Extraction on {pdf_path}")
    
    try:
        # Step 1: Convert PDF to images
        logger.info("Converting PDF to images...")
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            img_data = pix.tobytes("png")
            images.append(img_data)
        
        doc.close()
        
        # Step 2: Process images with Tesseract OCR
        logger.info("Processing images with Tesseract OCR...")
        searchable_pdf_path = tempfile.mktemp(suffix='.pdf')
        
        try:
            # Create searchable PDF from images
            with open(searchable_pdf_path, 'wb') as f:
                for i, img_data in enumerate(images):
                    # Convert bytes to PIL Image
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Save as temporary image
                    temp_img_path = tempfile.mktemp(suffix='.png')
                    img.save(temp_img_path)
                    
                    # Run Tesseract OCR
                    output_path = temp_img_path.replace('.png', '')
                    subprocess.run([
                        'tesseract', temp_img_path, output_path,
                        '--oem', '3', '--psm', '6', 'pdf'
                    ], check=True, capture_output=True)
                    
                    # Clean up temporary image
                    os.remove(temp_img_path)
            
            # Step 3: Extract tables from searchable PDF using tabula-py
            logger.info("Extracting tables from searchable PDF...")
            tables = tabula.read_pdf(
                searchable_pdf_path,
                pages='all',
                multiple_tables=True,
                guess=True,
                lattice=True,
                stream=True,
                silent=True
            )
            
            # Filter out empty tables
            filtered_tables = []
            for table in tables:
                if isinstance(table, pd.DataFrame) and not table.empty and table.shape[0] > 1 and table.shape[1] > 1:
                    filtered_tables.append(table)
            
            logger.info(f"Enhanced OCR extraction completed: {len(filtered_tables)} tables found")
            return filtered_tables
            
        finally:
            # Clean up temporary files
            if os.path.exists(searchable_pdf_path):
                os.remove(searchable_pdf_path)
        
    except Exception as e:
        logger.error(f"Enhanced OCR extraction failed: {e}")
        return []

def extract_tables_enhanced_ocr_gpu(pdf_path):
    """
    GPU-accelerated enhanced OCR extraction.
    """
    logger.info(f"Running GPU-Accelerated Enhanced OCR Extraction on {pdf_path}")
    
    try:
        # Check GPU availability
        gpu_available = check_gpu_availability()
        
        # Step 1: Convert PDF to images with GPU acceleration
        logger.info("Converting PDF to images with GPU acceleration...")
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = pix.tobytes("png")
            images.append(img_data)
        
        doc.close()
        
        # Step 2: Process images with GPU-accelerated Tesseract
        logger.info("Processing images with GPU-accelerated Tesseract...")
        searchable_pdf_path = tempfile.mktemp(suffix='.pdf')
        
        try:
            # Create searchable PDF from images with GPU acceleration
            with open(searchable_pdf_path, 'wb') as f:
                for i, img_data in enumerate(images):
                    # Convert bytes to PIL Image
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Save as temporary image
                    temp_img_path = tempfile.mktemp(suffix='.png')
                    img.save(temp_img_path)
                    
                    # Run Tesseract OCR with GPU flags
                    output_path = temp_img_path.replace('.png', '')
                    tesseract_cmd = [
                        'tesseract', temp_img_path, output_path,
                        '--oem', '3', '--psm', '6', 'pdf'
                    ]
                    
                    if gpu_available:
                        tesseract_cmd.extend(['--tessedit_use_cuda', '1'])
                    
                    subprocess.run(tesseract_cmd, check=True, capture_output=True)
                    
                    # Clean up temporary image
                    os.remove(temp_img_path)
            
            # Step 3: Extract tables from searchable PDF using tabula-py
            logger.info("Extracting tables from searchable PDF...")
            tables = tabula.read_pdf(
                searchable_pdf_path,
                pages='all',
                multiple_tables=True,
                guess=True,
                lattice=True,
                stream=True,
                silent=True
            )
            
            # Filter out empty tables
            filtered_tables = []
            for table in tables:
                if isinstance(table, pd.DataFrame) and not table.empty and table.shape[0] > 1 and table.shape[1] > 1:
                    filtered_tables.append(table)
            
            logger.info(f"GPU-accelerated enhanced OCR extraction completed: {len(filtered_tables)} tables found")
            return filtered_tables
            
        finally:
            # Clean up temporary files
            if os.path.exists(searchable_pdf_path):
                os.remove(searchable_pdf_path)
        
    except Exception as e:
        logger.error(f"GPU-accelerated enhanced OCR extraction failed: {e}")
        return []

def standardize_column_names(df, target_schema=TARGET_SCHEMA):
    """
    Standardize column names using fuzzy matching and comprehensive schema.
    """
    if df is None or df.empty:
        return df
    
    standardized_columns = []
    
    for col in df.columns:
        col_str = str(col).strip()
        
        # Find best match in target schema
        best_match = None
        best_score = 0
        
        for target_col, aliases in target_schema.items():
            for alias in aliases:
                # Calculate similarity
                similarity = calculate_similarity(col_str.lower(), alias.lower())
                if similarity > best_score and similarity > 0.6:  # Lowered threshold
                    best_score = similarity
                    best_match = target_col
        
        if best_match:
            standardized_columns.append(best_match)
        else:
            # Fallback: convert to snake_case
            fallback_name = re.sub(r'[^\w\s]', ' ', col_str)
            fallback_name = re.sub(r'\s+', '_', fallback_name.lower().strip())
            standardized_columns.append(fallback_name)
    
    df.columns = standardized_columns
    return df

def calculate_similarity(str1, str2):
    """Calculate similarity between two strings"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, str1, str2).ratio()

def save_tables_to_csv(tables, output_dir, extraction_method):
    """Save extracted tables to CSV files"""
        if not tables:
        logger.warning("No tables to save")
            return

    logger.info(f"Saving {len(tables)} tables to {output_dir}")

        for i, table in enumerate(tables):
        if table is not None and not table.empty:
            # Standardize column names
            table = standardize_column_names(table)
            
            # Save to CSV
            output_file = os.path.join(output_dir, f"{extraction_method}_table_{i+1}.csv")
            table.to_csv(output_file, index=False)
            logger.info(f"Saved table {i+1} to {output_file}")

# NOTE: This function is now imported from src.processors.raw_state_table_processing
# Keeping for backward compatibility - will be removed in future versions
def detect_row_structure_robust(row: pd.Series) -> Tuple[str, str, int]:
    """
    Detect the structure of a row and determine:
    - region_code: The region this row belongs to
    - state_name: The state name
    - data_start_col: The column index where data starts (to handle shifted data)
    """
    # Clean the row data
    row_clean = [str(cell).strip() if pd.notna(cell) else "" for cell in row]
    
    # Check if this is a region header row
    if row_clean[0] in ['NR', 'WR', 'SR', 'ER', 'NER']:
        region_code = row_clean[0]
        state_name = row_clean[1] if row_clean[1] else ""
        data_start_col = 2  # Data starts from column 2
        return region_code, state_name, data_start_col
    
    # Check if this is a state row with region in first column
    elif row_clean[0] in ['NR', 'WR', 'SR', 'ER', 'NER'] and row_clean[1]:
        region_code = row_clean[0]
        state_name = row_clean[1]
        data_start_col = 2
        return region_code, state_name, data_start_col
    
    # Check if this is a state row with empty first column (under a region)
    elif not row_clean[0] and row_clean[1]:
        # This is a state row under the current region
        # We need to determine the region from context
        region_code = "UNKNOWN"  # Will be set by caller
        state_name = row_clean[1]
        data_start_col = 2
        return region_code, state_name, data_start_col
    
    # Check for shifted data (when state name is in column 0 but no region)
    elif row_clean[0] and row_clean[0] not in ['NR', 'WR', 'SR', 'ER', 'NER']:
        # This might be a state row with shifted data
        region_code = "UNKNOWN"  # Will be set by caller
        state_name = row_clean[0]
        data_start_col = 1  # Data starts from column 1 (shifted)
        return region_code, state_name, data_start_col
    
    # Default case
    else:
        region_code = "UNKNOWN"
        state_name = ""
        data_start_col = 0
        return region_code, state_name, data_start_col

def extract_numeric_value_robust(cell_value: Any) -> float:
    """Safely extract numeric value from a cell."""
    if pd.isna(cell_value):
        return None
    
    try:
        # Convert to string and clean
        cell_str = str(cell_value).strip()
        if not cell_str or cell_str == 'nan':
            return None
        
        # Try to convert to float
        return float(cell_str)
    except (ValueError, TypeError):
        return None

def process_state_table_robustly(table_df: pd.DataFrame, report_date: str) -> List[Dict]:
    """Robustly process the state table structure handling inconsistencies."""
    logger.info("Processing state table robustly...")
    
    # Clean column names
    table_df.columns = [col.replace('\r', ' ').replace('\n', ' ') for col in table_df.columns]
    
    # Initialize result list
    results = []
    current_region = None
    
    for index, row in table_df.iterrows():
        # Skip completely empty rows
        if row.isna().all():
                continue

        # Get row data as list
        row_data = list(row)
        
        # Check if this is a region header row (has region code in first column)
        if len(row_data) > 0 and pd.notna(row_data[0]) and str(row_data[0]).strip() in ['NR', 'WR', 'SR', 'ER', 'NER']:
            current_region = str(row_data[0]).strip()
            # If second column has state name, process it
            if len(row_data) > 1 and pd.notna(row_data[1]) and str(row_data[1]).strip():
                state_name = str(row_data[1]).strip()
                if state_name != 'nan' and state_name != 'States':
                    try:
                        record = {
                            "region_code": current_region,
                            "state_name": state_name,
                            "power_max_demand_met_day_mw": extract_numeric_value_robust(row_data[2]) if len(row_data) > 2 else None,
                            "power_shortage_at_max_demand_mw": extract_numeric_value_robust(row_data[3]) if len(row_data) > 3 else None,
                            "energy_met_mu": extract_numeric_value_robust(row_data[4]) if len(row_data) > 4 else None,
                            "energy_drawal_schedule_mu": extract_numeric_value_robust(row_data[5]) if len(row_data) > 5 else None,
                            "energy_over_under_drawal_mu": extract_numeric_value_robust(row_data[6]) if len(row_data) > 6 else None,
                            "power_max_overdrawal_mw": extract_numeric_value_robust(row_data[7]) if len(row_data) > 7 else None,
                            "energy_shortage_mu": extract_numeric_value_robust(row_data[8]) if len(row_data) > 8 else None,
                            "report_date": report_date
                        }
                        results.append(record)
                    except (ValueError, TypeError, IndexError) as e:
                        logger.warning(f"Failed to process region row {index}: {e}")
        
        # Check if this is a state row (empty first column, state name in second column)
        elif len(row_data) > 1 and (pd.isna(row_data[0]) or str(row_data[0]).strip() == '') and pd.notna(row_data[1]):
            state_name = str(row_data[1]).strip()
            if state_name != 'nan' and state_name != 'States' and current_region:
                try:
                    record = {
                        "region_code": current_region,
                        "state_name": state_name,
                        "power_max_demand_met_day_mw": extract_numeric_value_robust(row_data[2]) if len(row_data) > 2 else None,
                        "power_shortage_at_max_demand_mw": extract_numeric_value_robust(row_data[3]) if len(row_data) > 3 else None,
                        "energy_met_mu": extract_numeric_value_robust(row_data[4]) if len(row_data) > 4 else None,
                        "energy_drawal_schedule_mu": extract_numeric_value_robust(row_data[5]) if len(row_data) > 5 else None,
                        "energy_over_under_drawal_mu": extract_numeric_value_robust(row_data[6]) if len(row_data) > 6 else None,
                        "power_max_overdrawal_mw": extract_numeric_value_robust(row_data[7]) if len(row_data) > 7 else None,
                        "energy_shortage_mu": extract_numeric_value_robust(row_data[8]) if len(row_data) > 8 else None,
                        "report_date": report_date
                    }
                    results.append(record)
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Failed to process state row {index}: {e}")
        
        # Check for shifted data structure (state name in first column, no region code)
        elif len(row_data) > 0 and pd.notna(row_data[0]) and str(row_data[0]).strip() and str(row_data[0]).strip() not in ['NR', 'WR', 'SR', 'ER', 'NER'] and current_region:
            state_name = str(row_data[0]).strip()
            if state_name != 'nan' and state_name != 'States':
                try:
                    # Data is shifted left by one column
                    record = {
                        "region_code": current_region,
                        "state_name": state_name,
                        "power_max_demand_met_day_mw": extract_numeric_value_robust(row_data[1]) if len(row_data) > 1 else None,
                        "power_shortage_at_max_demand_mw": extract_numeric_value_robust(row_data[2]) if len(row_data) > 2 else None,
                        "energy_met_mu": extract_numeric_value_robust(row_data[3]) if len(row_data) > 3 else None,
                        "energy_drawal_schedule_mu": extract_numeric_value_robust(row_data[4]) if len(row_data) > 4 else None,
                        "energy_over_under_drawal_mu": extract_numeric_value_robust(row_data[5]) if len(row_data) > 5 else None,
                        "power_max_overdrawal_mw": extract_numeric_value_robust(row_data[6]) if len(row_data) > 6 else None,
                        "energy_shortage_mu": extract_numeric_value_robust(row_data[7]) if len(row_data) > 7 else None,
                        "report_date": report_date
                    }
                    results.append(record)
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Failed to process shifted state row {index}: {e}")
    
    logger.info(f"Successfully processed {len(results)} records robustly")
    return results

def detect_table_type(table_df: pd.DataFrame) -> str:
    """
    Detect the type of table based on its content and headers.
    """
    if table_df.empty:
        return 'unknown'
    
    # Get table content as string for pattern matching
    table_content = ' '.join([
        ' '.join([str(col) for col in table_df.columns]),
        ' '.join([str(val) for val in table_df.values.flatten() if pd.notna(val)])
    ]).lower()
    
    # Score each table type based on pattern matches
    scores = {}
    for table_type, patterns in TABLE_TYPE_PATTERNS.items():
        score = sum(1 for pattern in patterns if pattern in table_content)
        scores[table_type] = score
    
    # Return the table type with highest score
    if scores:
        best_type = max(scores, key=scores.get)
        if scores[best_type] > 0:
            return best_type
    
    return 'unknown'

def call_llm(prompt: str, model: str = LLM_MODEL) -> Optional[str]:
    """
    Call the local LLM (Ollama) to process the prompt.
    """
    try:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 2000
            }
        }
        
        response = requests.post(
            LLM_ENDPOINT,
            json=payload,
            timeout=LLM_TIMEOUT
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('response', '').strip()
                else:
            logger.error(f"LLM API error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"LLM processing error: {e}")
        return None

def process_table_with_llm(table_df: pd.DataFrame, table_index: int, report_date: str = None) -> Dict[str, Any]:
    """
    Process a table using LLM inference to return structured format.
    For state tables, use robust processing instead of LLM.
    """
    try:
        # Detect table type
        table_type = detect_table_type(table_df)
        logger.info(f"Table {table_index}: Detected type '{table_type}'")
        
        if table_type == 'unknown':
            logger.warning(f"Table {table_index}: Unknown table type, skipping processing")
            return {
                'table_index': table_index,
                'table_type': 'unknown',
                'structured_data': None,
                'error': 'Unknown table type'
            }
        
        # For state tables, use robust processing instead of LLM
        if table_type == 'state_daily_summary':
            logger.info(f"Table {table_index}: Using robust processing for state table")
            try:
                robust_results = process_state_table_robustly(table_df, report_date or "2025-04-19")
                if robust_results:
                    logger.info(f"Table {table_index}: Successfully processed {len(robust_results)} records robustly")
                    return {
                        'table_index': table_index,
                        'table_type': table_type,
                        'structured_data': robust_results,
                        'error': None,
                        'processing_method': 'robust'
                    }
                else:
                    logger.warning(f"Table {table_index}: Robust processing returned no results")
                    return {
                        'table_index': table_index,
                        'table_type': table_type,
                        'structured_data': None,
                        'error': 'Robust processing returned no results'
                    }
            except Exception as e:
                logger.error(f"Table {table_index}: Error in robust processing: {e}")
                return {
                    'table_index': table_index,
                    'table_type': table_type,
                    'structured_data': None,
                    'error': f'Robust processing failed: {e}'
                }
        
        # For other table types, use LLM processing
        # Prepare table content for LLM
        table_content = table_df.to_string(index=False, max_rows=20, max_cols=10)
        
        # Get the appropriate prompt for this table type
        if table_type in TABLE_PROMPTS:
            prompt_template = TABLE_PROMPTS[table_type]
            prompt = prompt_template.format(input_table=table_content)
        else:
            logger.warning(f"Table {table_index}: No prompt template for type '{table_type}'")
            return {
                'table_index': table_index,
                'table_type': table_type,
                'structured_data': None,
                'error': f'No prompt template for type {table_type}'
            }
        
        # Call LLM
        logger.info(f"Table {table_index}: Calling LLM for processing...")
        llm_response = call_llm(prompt)
        
        if not llm_response:
            logger.error(f"Table {table_index}: LLM returned no response")
            return {
                'table_index': table_index,
                'table_type': table_type,
                'structured_data': None,
                'error': 'LLM returned no response'
            }
        
        # Try to parse JSON response
        try:
            # Clean the response and extract JSON
            cleaned_response = llm_response.strip()
            
            # Remove any markdown formatting
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-3]
            
            # Try multiple JSON extraction strategies
            structured_data = None
            
            # Strategy 1: Look for JSON object
            json_start = cleaned_response.find('{')
            json_end = cleaned_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = cleaned_response[json_start:json_end]
                try:
                    structured_data = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Look for JSON array
            if structured_data is None:
                array_start = cleaned_response.find('[')
                array_end = cleaned_response.rfind(']') + 1
                
                if array_start != -1 and array_end > array_start:
                    json_str = cleaned_response[array_start:array_end]
                    try:
                        structured_data = json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            # Strategy 3: Try to fix common JSON issues
            if structured_data is None:
                # Try to fix common issues like missing quotes around property names
                fixed_response = re.sub(r'(\w+):', r'"\1":', cleaned_response)
                try:
                    structured_data = json.loads(fixed_response)
                except json.JSONDecodeError:
                    pass
            
            # Strategy 4: Try parsing the entire response as last resort
            if structured_data is None:
                try:
                    structured_data = json.loads(llm_response)
                except json.JSONDecodeError:
                    pass
            
            if structured_data is not None:
                logger.info(f"Table {table_index}: Successfully processed with LLM")
                return {
                    'table_index': table_index,
                    'table_type': table_type,
                    'structured_data': structured_data,
                    'error': None,
                    'processing_method': 'llm'
                }
            else:
                logger.error(f"Table {table_index}: Failed to parse LLM response as JSON")
                logger.debug(f"LLM Response: {llm_response[:500]}...")
                return {
                    'table_index': table_index,
                    'table_type': table_type,
                    'structured_data': None,
                    'error': 'JSON parsing failed after multiple strategies'
                }

    except Exception as e:
            logger.error(f"Table {table_index}: Error parsing LLM response: {e}")
            logger.debug(f"LLM Response: {llm_response[:500]}...")
            return {
                'table_index': table_index,
                'table_type': table_type,
                'structured_data': None,
                'error': f'JSON parsing error: {e}'
            }
            
    except Exception as e:
        logger.error(f"Table {table_index}: Error in processing: {e}")
        return {
            'table_index': table_index,
            'table_type': 'unknown',
            'structured_data': None,
            'error': str(e)
        }

def process_all_tables_with_llm(tables: List[pd.DataFrame], report_date: str = None) -> List[Dict[str, Any]]:
    """
    Process all tables using LLM inference and robust processing for state tables.
    """
    logger.info(f"Starting table processing for {len(tables)} tables...")
    
    results = []
    for i, table in enumerate(tables):
        logger.info(f"Processing table {i+1}/{len(tables)}...")
        
        # Add delay between LLM calls to avoid overwhelming the service
        if i > 0:
            time.sleep(1)
        
        result = process_table_with_llm(table, i+1, report_date)
        results.append(result)
        
        # Log progress
        if result['error']:
            logger.warning(f"Table {i+1}: {result['error']}")
        else:
            processing_method = result.get('processing_method', 'unknown')
            logger.info(f"Table {i+1}: Successfully processed as {result['table_type']} using {processing_method}")
    
    # Summary
    successful = sum(1 for r in results if r['structured_data'] is not None)
    failed = len(results) - successful
    
    logger.info(f"Table processing completed: {successful} successful, {failed} failed")
    
    return results

def save_llm_results(results: List[Dict[str, Any]], output_dir: str, extraction_method: str):
    """
    Save LLM and robust processing results to JSON files.
    """
    if not results:
        logger.warning("No processing results to save")
        return
    
    # Create processed results directory
    processed_output_dir = os.path.join(output_dir, "processed_tables")
    Path(processed_output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving processing results to {processed_output_dir}")
    
    # Track processing methods
    llm_count = 0
    robust_count = 0
    
    # Save individual table results
    for result in results:
        if result['structured_data'] is not None:
            table_index = result['table_index']
            table_type = result['table_type']
            processing_method = result.get('processing_method', 'unknown')
            
            # Track processing method
            if processing_method == 'robust':
                robust_count += 1
            elif processing_method == 'llm':
                llm_count += 1
            
            # Create filename
            filename = f"{extraction_method}_table_{table_index}_{table_type}_{processing_method}.json"
            filepath = os.path.join(processed_output_dir, filename)
            
            # Save structured data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {processing_method} result for table {table_index} ({table_type}) to {filepath}")
    
    # Save summary report
    summary = {
        'total_tables': len(results),
        'successful_tables': sum(1 for r in results if r['structured_data'] is not None),
        'failed_tables': sum(1 for r in results if r['structured_data'] is None),
        'llm_processed': llm_count,
        'robust_processed': robust_count,
        'table_types': {},
        'errors': []
    }
    
    # Count table types
    for result in results:
        table_type = result['table_type']
        if table_type not in summary['table_types']:
            summary['table_types'][table_type] = 0
        summary['table_types'][table_type] += 1
    
    # Collect errors
    for result in results:
        if result['error']:
            summary['errors'].append({
                'table_index': result['table_index'],
                'table_type': result['table_type'],
                'error': result['error']
            })
    
    # Save summary
    summary_file = os.path.join(processed_output_dir, f"{extraction_method}_processing_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved processing summary to {summary_file}")
    logger.info(f"Processing Summary: {summary['successful_tables']}/{summary['total_tables']} tables successfully processed")
    logger.info(f"  - LLM processed: {llm_count}")
    logger.info(f"  - Robust processed: {robust_count}")

def main():
    """Main function with simplified approach"""
    logger.info("Starting National Power Supply Data Warehouse - Main Extractor")
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    
    while True:
        logger.info("\n" + "="*50)
        logger.info("PDF Table Extraction and Standardization Pipeline")
        logger.info("="*50)
        
        if gpu_available:
            logger.info("🚀 GPU detected - GPU acceleration will be used where available")
        
        logger.info("Choose an extraction method:")
        logger.info("  1. Standard Extraction (Fast, tabula-py only)")
        logger.info("  2. Parallel Standard Extraction (Fast, Multiple tabula-py strategies)")
        logger.info("  3. Enhanced OCR Extraction (Slow, Tesseract + tabula-py)")
        logger.info("  4. GPU-Accelerated Enhanced OCR (Fast, Parallel Tesseract + tabula-py)")
        logger.info("  5. Auto Mode (Standard by default, fallback to Enhanced if needed)")
        logger.info("  6. Exit")
        
        choice = input("Enter your choice (1/2/3/4/5/6): ")

        extracted_tables = []
        output_dir = ""
        extraction_method = ""

        if choice == '1':
            extracted_tables = extract_tables_standard(INPUT_PDF_PATH)
            output_dir = STANDARD_OUTPUT_DIR
            extraction_method = "Standard"
        elif choice == '2':
            extracted_tables = extract_tables_standard_parallel(INPUT_PDF_PATH)
            output_dir = STANDARD_OUTPUT_DIR
            extraction_method = "Parallel Standard"
        elif choice == '3':
            extracted_tables = extract_tables_enhanced_ocr(INPUT_PDF_PATH)
            output_dir = ENHANCED_OUTPUT_DIR
            extraction_method = "Enhanced OCR"
        elif choice == '4':
            extracted_tables = extract_tables_enhanced_ocr_gpu(INPUT_PDF_PATH)
            output_dir = ENHANCED_OUTPUT_DIR
            extraction_method = "GPU-Accelerated Enhanced OCR"
        elif choice == '5':
            # Auto Mode: Try standard first, fallback to enhanced if needed
            logger.info("🔄 Auto Mode: Starting with Standard Extraction...")
            
            # Try standard extraction first
            if gpu_available:
                logger.info("Using GPU-accelerated standard extraction...")
                extracted_tables = extract_tables_standard_parallel(INPUT_PDF_PATH)
            else:
                logger.info("Using CPU standard extraction...")
                extracted_tables = extract_tables_standard(INPUT_PDF_PATH)
            
            output_dir = STANDARD_OUTPUT_DIR
            extraction_method = "Auto (Standard)"
            
            # Validate the extracted tables
            if not extracted_tables or len(extracted_tables) == 0:
                logger.warning("⚠️ Standard extraction failed or returned no tables")
                logger.info("Would you like to try Enhanced OCR extraction? (y/n): ")
                fallback_choice = input().lower().strip()
                
                if fallback_choice in ['y', 'yes']:
                    logger.info("🔄 Switching to Enhanced OCR extraction...")
                    if gpu_available:
                        logger.info("Using GPU-accelerated enhanced OCR...")
                        extracted_tables = extract_tables_enhanced_ocr_gpu(INPUT_PDF_PATH)
                    else:
                        logger.info("Using CPU enhanced OCR...")
                        extracted_tables = extract_tables_enhanced_ocr(INPUT_PDF_PATH)
                    
                    output_dir = ENHANCED_OUTPUT_DIR
                    extraction_method = "Auto (Enhanced OCR)"
                else:
                    logger.info("User chose not to try enhanced extraction. Exiting.")
                    break
            else:
                logger.info("✅ Standard extraction successful!")
                
        elif choice == '6':
            logger.info("Exiting.")
            break
        else:
            logger.error("Invalid choice. Please try again.")
            continue

        # Save results
        if extracted_tables:
            save_tables_to_csv(extracted_tables, output_dir, extraction_method)
            logger.info(f"✅ Extraction completed successfully! Saved {len(extracted_tables)} tables to {output_dir}")
            
            # Process tables with LLM
            logger.info("🤖 Starting LLM processing for structured data extraction...")
            try:
                # Extract report date from PDF filename
                pdf_filename = os.path.basename(INPUT_PDF_PATH)
                report_date = None
                if pdf_filename:
                    # Try to extract date from filename (e.g., "19.04.25_NLDC_PSP.pdf")
                    date_match = re.search(r'(\d{2})\.(\d{2})\.(\d{2})', pdf_filename)
                    if date_match:
                        day, month, year = date_match.groups()
                        report_date = f"20{year}-{month}-{day}"
                
                # Process all tables with LLM
                llm_results = process_all_tables_with_llm(extracted_tables, report_date)
                
                # Save LLM results
                save_llm_results(llm_results, output_dir, extraction_method)
                
                logger.info("✅ LLM processing completed successfully!")
                
            except Exception as e:
                logger.error(f"❌ LLM processing failed: {e}")
                logger.info("Continuing without LLM processing...")
        else:
            logger.warning("No tables were extracted.")

if __name__ == "__main__":
    main()

