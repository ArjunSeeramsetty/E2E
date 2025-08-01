"""
Enhanced main extractor for the Power Supply Data Warehouse
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any
from loguru import logger

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.parsers.parser_factory import ParserFactory
from src.database.connection import DatabaseConnection
from src.validation.data_validator import DataValidator
from src.models.data_models import ParsedReport
from config.settings import settings

class PowerSupplyDataExtractor:
    def __init__(self):
        self.parser_factory = ParserFactory()
        self.db_connection = DatabaseConnection()
        self.validator = DataValidator()
        settings.create_directories()
        logger.add("logs/extractor.log", rotation="1 day", retention="30 days", level=settings.LOG_LEVEL)

    def process_report(self, pdf_path: str, source_entity: str = None) -> bool:
        """Process a single PDF report"""
        try:
            # Auto-detect source entity if not provided
            if source_entity is None:
                filename = os.path.basename(pdf_path)
                source_entity = self._detect_source_entity(filename)
            
            # Get appropriate parser
            parser = self.parser_factory.get_parser(source_entity)
            
            # Parse the report
            parsed_report = parser.parse(pdf_path, source_entity)
            
            # Validate the parsed data
            validation_result = self.validator.validate(parsed_report)
            
            if not validation_result.is_valid:
                logger.error(f"Validation failed for {pdf_path}: {validation_result.errors}")
                return False
            
            if validation_result.warnings:
                logger.warning(f"Validation warnings for {pdf_path}: {validation_result.warnings}")
            
            # Store in database
            success = self.db_connection.store_report(parsed_report)
            
            if success:
                logger.info(f"Successfully processed {pdf_path}")
            else:
                logger.error(f"Failed to store {pdf_path} in database")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return False

    def process_directory(self, directory_path: str) -> Dict[str, int]:
        """Process all PDF files in a directory"""
        success_count = 0
        failure_count = 0
        
        try:
            # Find all PDF files
            pdf_files = []
            for file in os.listdir(directory_path):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(directory_path, file))
            
            logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
            
            # Process each file
            for pdf_file in pdf_files:
                if self.process_report(pdf_file):
                    success_count += 1
                else:
                    failure_count += 1
            
            logger.info(f"Processing complete. Success: {success_count}, Failures: {failure_count}")
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            failure_count += 1
        
        return {"success": success_count, "failure": failure_count}

    def _detect_source_entity(self, filename: str) -> str:
        """Auto-detect source entity from filename"""
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
        else:
            # Default to NLDC if can't detect
            return "NLDC"

    def test_connection(self) -> bool:
        return self.db_connection.test_connection()

def main():
    extractor = PowerSupplyDataExtractor()
    
    if not extractor.test_connection():
        logger.error("Database connection test failed")
        return
    
    if len(sys.argv) > 1:
        pdf_file_path = sys.argv[1]
        if os.path.exists(pdf_file_path):
            success = extractor.process_report(pdf_file_path)
            print(f"Processing {'successful' if success else 'failed'}")
        else:
            print(f"File not found: {pdf_file_path}")
    else:
        raw_data_dir = settings.RAW_DATA_DIR
        if os.path.exists(raw_data_dir):
            results = extractor.process_directory(raw_data_dir)
            print(f"Processed {results['success']} files successfully, {results['failure']} failures")
        else:
            print(f"Raw data directory not found: {raw_data_dir}")
            print("Please place PDF files in the data/raw directory or provide a file path as argument")

if __name__ == "__main__":
    main()

