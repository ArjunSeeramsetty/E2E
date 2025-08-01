# Power Supply Data Warehouse

A comprehensive system for extracting, processing, and analyzing daily Power Supply Position (PSP) reports from the National Load Despatch Centre (NLDC) and Regional Load Despatch Centres (RLDCs).

## Features

- **Modular PDF Parsing**: Advanced table extraction using docling
- **Schema-Aware Processing**: Region-specific parsers for different report formats
- **Data Validation**: Comprehensive validation with detailed error reporting
- **Database Integration**: PostgreSQL/SQLite support with SQLAlchemy ORM
- **Extensible Architecture**: Easy to add new report formats and data sources

## Quick Start

### 1. Setup Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Usage

```bash
# Process a single PDF file
python main_extractor.py path/to/your/report.pdf

# Process all PDFs in data/raw directory
python main_extractor.py
```

### 3. Project Structure

```
E2E/
├── src/                    # Source code
│   ├── models/            # Data models and database schema
│   ├── parsers/           # PDF parsing logic
│   ├── database/          # Database operations
│   ├── validation/        # Data validation
│   └── utils/             # Utility functions
├── config/                # Configuration settings
├── data/                  # Data directories
│   ├── raw/              # Input PDF files
│   └── processed/        # Processed data
├── logs/                  # Application logs
├── tests/                 # Unit tests
├── requirements.txt       # Python dependencies
└── main_extractor.py     # Main application
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

- **Parsers**: Handle different report formats (NLDC, SRLDC, etc.)
- **Models**: Define data structures and database schema
- **Validation**: Ensure data quality and consistency
- **Database**: Manage data persistence and retrieval

## Development

### Adding New Parsers

1. Create a new parser class inheriting from `BaseParser`
2. Implement the required abstract methods
3. Register the parser in `ParserFactory`

### Database Schema

The system uses SQLAlchemy ORM with support for:
- Core tables (Reports, RegionalSummaries, StateSummaries)
- Region-specific tables (SR_StationGeneration, etc.)
- Extensible schema for future requirements

## Configuration

Edit `config/settings.py` to customize:
- Database connection
- File paths
- Logging settings
- Validation rules

## Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## License

This project is licensed under the MIT License. 