"""
Abstract base class for all report parsers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from docling.document_converter import DocumentConverter
from src.models.data_models import ParsedReport, Report, RegionalSummary, StateSummary, GenerationBySource
from loguru import logger
import re
from decimal import Decimal

class BaseParser(ABC):
    """Abstract base class for all report parsers"""
    
    def __init__(self):
        self.table_mappings = self._get_table_mappings()
        self.converter = DocumentConverter()
    
    @abstractmethod
    def _get_table_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Define how to identify and parse specific tables"""
        pass
    
    @abstractmethod
    def parse_regional_summary(self, table_data: List[List[str]]) -> List[RegionalSummary]:
        """Parse regional summary table (Table A equivalent)"""
        pass
    
    @abstractmethod
    def parse_state_summary(self, table_data: List[List[str]]) -> List[StateSummary]:
        """Parse state summary table (Table C equivalent)"""
        pass
    
    @abstractmethod
    def parse_generation_by_source(self, table_data: List[List[str]]) -> List[GenerationBySource]:
        """Parse generation by source table (Table G equivalent)"""
        pass
    
    def extract_tables(self, pdf_path: str) -> Dict[str, List[List[str]]]:
        """Extract all tables from PDF using docling"""
        try:
            # Use docling's convert method
            result = self.converter.convert(pdf_path)
            logger.info(f"Conversion result type: {type(result)}")
            logger.info(f"Conversion result attributes: {dir(result)}")
            
            # Get the document content
            document = result.document
            logger.info(f"Document type: {type(document)}")
            logger.info(f"Document attributes: {dir(document)}")
            
            # Extract tables from the document
            tables = []
            
            # Check if document has tables directly
            if hasattr(document, 'tables'):
                logger.info(f"Document has {len(document.tables)} tables directly")
                tables = document.tables
            # Check if document has pages
            elif hasattr(document, 'pages'):
                for page in document.pages:
                    if hasattr(page, 'elements'):
                        for element in page.elements:
                            if hasattr(element, 'type') and element.type == 'table':
                                tables.append(element)
            # Check if document has elements directly
            elif hasattr(document, 'elements'):
                for element in document.elements:
                    if hasattr(element, 'type') and element.type == 'table':
                        tables.append(element)
            # Check if document has content directly
            elif hasattr(document, 'content'):
                # Try to find tables in content
                logger.info("Document has content attribute, searching for tables...")
                logger.info(f"Document structure: {type(document)}")
                logger.info(f"Document attributes: {dir(document)}")
                
                # Try to access content and look for tables
                content = document.content
                logger.info(f"Content type: {type(content)}")
                logger.info(f"Content attributes: {dir(content)}")
                
                if hasattr(content, 'elements'):
                    logger.info(f"Content has {len(content.elements)} elements")
                    for i, element in enumerate(content.elements):
                        logger.info(f"Element {i}: type={getattr(element, 'type', 'unknown')}, class={type(element)}")
                        if hasattr(element, 'type') and element.type == 'table':
                            tables.append(element)
                            logger.info(f"Found table {len(tables)} in content.elements")
                
                if not tables:
                    logger.warning("No tables found in document content")
                    # Try to explore the content structure more deeply
                    logger.info("Exploring content structure...")
                    self._explore_content_structure(content)
                    return {}
            
            # Map tables by their headers/content
            table_map = {}
            logger.info(f"Found {len(tables)} tables in document")
            
            for i, table in enumerate(tables):
                logger.info(f"Processing table {i+1}: {type(table)}")
                logger.info(f"Table {i+1} attributes: {dir(table)}")
                
                # Try to get table data
                table_data = self._convert_table_to_list(table)
                logger.info(f"Table {i+1} data: {len(table_data)} rows")
                
                if table_data and len(table_data) > 0:
                    headers = table_data[0]
                    logger.info(f"Table {i+1} headers: {headers}")
                    table_type = self._identify_table(headers)
                    if table_type != "unknown":
                        table_map[table_type] = table_data
                        logger.info(f"Identified table {i+1} as {table_type}")
                    else:
                        logger.warning(f"Could not identify table {i+1}")
                else:
                    logger.warning(f"Table {i+1} has no data")
            
            logger.info(f"Final table map: {list(table_map.keys())}")
            return table_map
            
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {str(e)}")
            raise
    
    def _explore_content_structure(self, content, depth=0, max_depth=3):
        """Recursively explore content structure to find tables"""
        if depth > max_depth:
            return
        
        indent = "  " * depth
        logger.info(f"{indent}Exploring content at depth {depth}: {type(content)}")
        
        if hasattr(content, '__dict__'):
            for attr_name, attr_value in content.__dict__.items():
                logger.info(f"{indent}  {attr_name}: {type(attr_value)}")
                if hasattr(attr_value, 'type') and attr_value.type == 'table':
                    logger.info(f"{indent}  FOUND TABLE: {attr_name}")
                elif hasattr(attr_value, '__iter__') and not isinstance(attr_value, str):
                    try:
                        for i, item in enumerate(attr_value):
                            if i < 3:  # Limit to first 3 items
                                logger.info(f"{indent}    [{i}]: {type(item)}")
                                if hasattr(item, 'type') and item.type == 'table':
                                    logger.info(f"{indent}    FOUND TABLE in [{i}]")
                                elif depth < max_depth:
                                    self._explore_content_structure(item, depth + 1, max_depth)
                    except Exception as e:
                        logger.warning(f"{indent}  Could not iterate {attr_name}: {e}")
        
        # Also check for common attributes
        for attr in ['elements', 'children', 'content', 'data']:
            if hasattr(content, attr):
                attr_value = getattr(content, attr)
                logger.info(f"{indent}  {attr}: {type(attr_value)}")
                if hasattr(attr_value, '__iter__') and not isinstance(attr_value, str):
                    try:
                        for i, item in enumerate(attr_value):
                            if i < 3:  # Limit to first 3 items
                                if hasattr(item, 'type') and item.type == 'table':
                                    logger.info(f"{indent}    FOUND TABLE in {attr}[{i}]")
                    except Exception as e:
                        logger.warning(f"{indent}  Could not iterate {attr}: {e}")
    
    def _convert_table_to_list(self, table) -> List[List[str]]:
        """Convert docling table to list of lists format"""
        try:
            # Check if table has value attribute
            if hasattr(table, 'value') and table.value:
                # Handle different table formats
                if isinstance(table.value, list):
                    return table.value
                elif hasattr(table.value, 'rows'):
                    return [[cell.text if hasattr(cell, 'text') else str(cell) for cell in row] 
                           for row in table.value.rows]
                else:
                    # Fallback: try to convert to string and parse
                    return str(table.value).split('\n')
            
            # Check if table has rows attribute directly
            elif hasattr(table, 'rows'):
                logger.info(f"Table has rows attribute with {len(table.rows)} rows")
                result = []
                for row in table.rows:
                    row_data = []
                    for cell in row:
                        if hasattr(cell, 'text'):
                            row_data.append(cell.text)
                        else:
                            row_data.append(str(cell))
                    result.append(row_data)
                return result
            
            # Check if table has cells attribute
            elif hasattr(table, 'cells'):
                logger.info(f"Table has cells attribute")
                # Try to organize cells into rows
                cells = table.cells
                if hasattr(cells, '__iter__'):
                    # Group cells by row
                    rows = {}
                    for cell in cells:
                        if hasattr(cell, 'row') and hasattr(cell, 'col'):
                            row_idx = cell.row
                            if row_idx not in rows:
                                rows[row_idx] = {}
                            rows[row_idx][cell.col] = cell.text if hasattr(cell, 'text') else str(cell)
                    
                    # Convert to list of lists
                    result = []
                    for row_idx in sorted(rows.keys()):
                        row_data = []
                        for col_idx in sorted(rows[row_idx].keys()):
                            row_data.append(rows[row_idx][col_idx])
                        result.append(row_data)
                    return result
            
            # Check if table has data attribute
            elif hasattr(table, 'data'):
                logger.info(f"Table has data attribute")
                if isinstance(table.data, list):
                    return table.data
                else:
                    return str(table.data).split('\n')
            
            # Check if table has content attribute
            elif hasattr(table, 'content'):
                logger.info(f"Table has content attribute")
                content = table.content
                if hasattr(content, 'rows'):
                    return [[cell.text if hasattr(cell, 'text') else str(cell) for cell in row] 
                           for row in content.rows]
                elif isinstance(content, list):
                    return content
                else:
                    return str(content).split('\n')
            
            logger.warning(f"Could not find table data in attributes: {dir(table)}")
            return []
            
        except Exception as e:
            logger.warning(f"Could not convert table: {str(e)}")
            return []
    
    def _identify_table(self, headers: List[str]) -> str:
        """Identify table type based on headers"""
        header_text = " ".join(headers).lower()
        
        # Log the headers for debugging
        logger.info(f"Table headers: {headers}")
        logger.info(f"Header text: {header_text}")
        
        # NLDC table identification patterns
        if "power supply position" in header_text and "regional" in header_text:
            return "regional_summary"
        elif "power supply position" in header_text and "states" in header_text:
            return "state_summary"
        elif "sourcewise generation" in header_text or "source wise" in header_text:
            return "generation_by_source"
        elif "frequency profile" in header_text:
            return "frequency_profile"
        elif "inter regional" in header_text or "inter-regional" in header_text:
            return "inter_regional_exchange"
        elif "transnational" in header_text or "international" in header_text:
            return "transnational_exchange"
        elif "generation outage" in header_text:
            return "generation_outage"
        elif "station" in header_text and "generation" in header_text:
            return "station_generation"
        elif "time" in header_text and ("frequency" in header_text or "demand" in header_text):
            return "frequency_profile"
        elif "time" in header_text and "generation" in header_text:
            return "generation_profile"
        # Add more generic patterns for time-series data
        elif "time" in header_text:
            return "time_series_data"
        elif len(headers) > 0 and any("time" in h.lower() for h in headers):
            return "time_series_data"
        elif len(headers) > 0 and any("frequency" in h.lower() for h in headers):
            return "time_series_data"
        elif len(headers) > 0 and any("demand" in h.lower() for h in headers):
            return "time_series_data"
        elif len(headers) > 0 and any("generation" in h.lower() for h in headers):
            return "time_series_data"
        # If we have numeric data in the first row, it's likely time series
        elif len(headers) > 0 and any(re.match(r'^\d+', h.strip()) for h in headers):
            return "time_series_data"
        else:
            return "unknown"
    
    def _clean_numeric_value(self, value: str) -> Optional[Decimal]:
        """Clean and convert string to Decimal"""
        if not value or value.strip() == "" or value.strip() == "-":
            return None
        
        # Remove common non-numeric characters
        cleaned = re.sub(r'[^\d.-]', '', value.strip())
        
        try:
            return Decimal(cleaned)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert '{value}' to numeric value")
            return None
    
    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from filename like '19.04.25_NLDC_PSP.pdf'"""
        # Look for date patterns like DD.MM.YY or DD.MM.YYYY
        date_pattern = r'(\d{2})\.(\d{2})\.(\d{2,4})'
        match = re.search(date_pattern, filename)
        
        if match:
            day, month, year = match.groups()
            if len(year) == 2:
                year = f"20{year}"
            return f"{year}-{month}-{day}"
        
        raise ValueError(f"Could not extract date from filename: {filename}")
    
    def parse(self, pdf_path: str, source_entity: str) -> ParsedReport:
        """Main parsing method"""
        try:
            # Extract tables
            table_map = self.extract_tables(pdf_path)
            
            # Extract date from filename
            filename = pdf_path.split('/')[-1]
            report_date = self._extract_date_from_filename(filename)
            
            # Create report metadata
            report = Report(
                report_date=report_date,
                source_entity=source_entity
            )
            
            # Parse different table types
            regional_summaries = []
            if "regional_summary" in table_map:
                regional_summaries = self.parse_regional_summary(table_map["regional_summary"])
            
            state_summaries = []
            if "state_summary" in table_map:
                state_summaries = self.parse_state_summary(table_map["state_summary"])
            
            generation_by_source = []
            if "generation_by_source" in table_map:
                generation_by_source = self.parse_generation_by_source(table_map["generation_by_source"])
            
            return ParsedReport(
                report=report,
                regional_summaries=regional_summaries,
                state_summaries=state_summaries,
                generation_by_source=generation_by_source
            )
            
        except Exception as e:
            logger.error(f"Error parsing {pdf_path}: {str(e)}")
            raise 