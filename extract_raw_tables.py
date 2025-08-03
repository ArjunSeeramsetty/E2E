#!/usr/bin/env python3
"""
Individual Table Extraction Script
Extracts individual tables from PDF using tabula-py and saves each table as a separate CSV file.
"""

import os
import sys
import pandas as pd
import tabula
from pathlib import Path
from loguru import logger

# Configuration
INPUT_PDF_PATH = r"C:\Users\arjun\Desktop\E2E\data\raw\19.04.25_NLDC_PSP.pdf"
RAW_OUTPUT_DIR = "data/processed/raw_tables"

def setup_logging():
    """Setup logging configuration."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")
    logger.add("logs/extraction.log", rotation="10 MB", level="DEBUG")

def identify_table_sections(df: pd.DataFrame) -> list:
    """
    Identify individual table sections within a DataFrame.
    Looks for section headers and splits the data accordingly.
    """
    sections = []
    current_section = []
    section_name = "Unknown"
    
    for idx, row in df.iterrows():
        row_str = ' '.join([str(cell) for cell in row if pd.notna(cell)]).strip()
        
        # Check for section headers (A., B., C., etc.)
        if any(header in row_str for header in ['A. ', 'B. ', 'C. ', 'D. ', 'E. ', 'F. ', 'G. ', 'H. ', 'I. ']):
            # Save previous section if it exists
            if current_section:
                sections.append({
                    'name': section_name,
                    'data': pd.DataFrame(current_section, columns=df.columns)
                })
            
            # Start new section
            current_section = [row.tolist()]
            section_name = row_str
        else:
            current_section.append(row.tolist())
    
    # Add the last section
    if current_section:
        sections.append({
            'name': section_name,
            'data': pd.DataFrame(current_section, columns=df.columns)
        })
    
    return sections

def extract_individual_tables(pdf_path: str) -> list:
    """
    Extract individual tables from PDF using tabula-py.
    Each identified table is saved as a separate CSV.
    """
    logger.info(f"Starting individual table extraction from: {pdf_path}")
    
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return []
    
    try:
        # Get total number of pages
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        logger.info(f"PDF has {total_pages} pages")
        
        all_tables = []
        table_counter = 0
        
        # Extract tables from each page separately
        for page_num in range(1, total_pages + 1):
            logger.info(f"Processing page {page_num}/{total_pages}")
            
            try:
                # Extract tables from current page
                page_tables = tabula.read_pdf(
                    pdf_path,
                    pages=page_num,
                    multiple_tables=True,
                    guess=True,
                    lattice=True,
                    stream=True
                )
                
                logger.info(f"Found {len(page_tables)} raw tables on page {page_num}")
                
                # Process each table from the page
                for table_idx, table in enumerate(page_tables):
                    if not table.empty and table.shape[0] > 0 and table.shape[1] > 0:
                        # Try to identify individual sections within the table
                        sections = identify_table_sections(table)
                        
                        if len(sections) > 1:
                            # Multiple sections found, save each as separate table
                            logger.info(f"Table {table_counter + 1} has {len(sections)} sections")
                            for section_idx, section in enumerate(sections):
                                table_counter += 1
                                
                                # Add metadata to section
                                section['data'].attrs['page_number'] = page_num
                                section['data'].attrs['original_table_index'] = table_idx
                                section['data'].attrs['section_index'] = section_idx
                                section['data'].attrs['global_table_id'] = table_counter
                                section['data'].attrs['section_name'] = section['name']
                                
                                all_tables.append(section['data'])
                                logger.info(f"Section {table_counter}: {section['name']} - Shape {section['data'].shape}")
                        else:
                            # Single table, save as is
                            table_counter += 1
                            
                            # Add metadata to table
                            table.attrs['page_number'] = page_num
                            table.attrs['original_table_index'] = table_idx
                            table.attrs['section_index'] = 0
                            table.attrs['global_table_id'] = table_counter
                            table.attrs['section_name'] = 'Single Table'
                            
                            all_tables.append(table)
                            logger.info(f"Table {table_counter}: Page {page_num}, Shape {table.shape}")
                    else:
                        logger.warning(f"Empty table on page {page_num}, index {table_idx}")
                        
            except Exception as e:
                logger.error(f"Error processing page {page_num}: {e}")
                continue
        
        logger.info(f"Total individual tables extracted: {len(all_tables)}")
        return all_tables
        
    except Exception as e:
        logger.error(f"Error extracting tables: {e}")
        return []

def save_individual_tables(tables: list, output_dir: str):
    """
    Save each individual table as a separate CSV file.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving {len(tables)} individual tables to {output_dir}")
    
    for table in tables:
        try:
            # Get table metadata
            page_num = table.attrs.get('page_number', 'unknown')
            original_table_idx = table.attrs.get('original_table_index', 'unknown')
            section_idx = table.attrs.get('section_index', 'unknown')
            global_id = table.attrs.get('global_table_id', 'unknown')
            section_name = table.attrs.get('section_name', 'unknown')
            
            # Clean section name for filename
            clean_name = section_name.replace(' ', '_').replace('.', '').replace(',', '').replace('(', '').replace(')', '')[:30]
            
            # Create filename
            if section_name != 'Single Table':
                filename = f"table_{global_id:02d}_page_{page_num:02d}_{clean_name}.csv"
            else:
                filename = f"table_{global_id:02d}_page_{page_num:02d}.csv"
            
            filepath = os.path.join(output_dir, filename)
            
            # Save table with headers preserved
            table.to_csv(filepath, index=False, header=True)
            
            logger.info(f"Saved {filename}: Shape {table.shape}")
            
            # Log preview for first few tables
            if global_id <= 3:
                logger.debug(f"Table {global_id} preview:")
                logger.debug(table.head(3).to_string())
            
        except Exception as e:
            logger.error(f"Error saving table {global_id}: {e}")

def main():
    """Main function to extract and save individual tables."""
    setup_logging()
    
    logger.info("="*60)
    logger.info("INDIVIDUAL TABLE EXTRACTION SCRIPT")
    logger.info("="*60)
    
    # Extract individual tables
    individual_tables = extract_individual_tables(INPUT_PDF_PATH)
    
    if not individual_tables:
        logger.error("No tables extracted. Exiting.")
        return
    
    # Save individual tables
    save_individual_tables(individual_tables, RAW_OUTPUT_DIR)
    
    logger.info("="*60)
    logger.info("EXTRACTION COMPLETED")
    logger.info(f"Individual tables saved to: {RAW_OUTPUT_DIR}")
    logger.info(f"Total individual tables extracted: {len(individual_tables)}")
    logger.info("="*60)

if __name__ == "__main__":
    main() 