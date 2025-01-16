# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Asynchronous HTTP requests using aiohttp
- Proxy support through SSH tunnels to EC2 instances
- Command-line arguments for SSH tunnel management
- Active tunnels tracking in .active_tunnels.json
- Raw HTML caching system with metadata tracking
- Updated training script with standardized column names
- Progress bars using tqdm for better visibility

### Changed
- Converted synchronous scrapers to async versions
- Unified base scraper class with proxy and async support
- Improved fighter data collection with parallel processing
- Better error reporting and logging
- Moved preprocessing logic to dedicated module
- Updated model training to use new data structure

### Removed
- Old synchronous HTTP request methods
- Redundant base.py in favor of base_scraper.py
- Unused preprocessing functions
- Legacy column naming conventions

### Technical Details

#### Base Scraper Changes
- Added SSH tunnel management for proxy support
- Implemented retry logic with exponential backoff
- Added async request handling with timeout support
- Added get_with_retry method for resilient HTTP requests
- Added process_urls methods for batch URL processing
- Added progress bar integration
- Added proxy rotation and load balancing

#### Fighter Scraper Changes
- Converted to async/await pattern
- Added batch processing of fighter URLs
- Improved error handling and data validation
- Added proxy support through base class
- Added automatic cleanup of resources

#### Event Scraper Changes
- Converted to async/await pattern
- Improved event data parsing
- Added standardized column naming
- Added session management for requests

#### Fight Scraper Changes
- Updated to use async methods
- Improved fight data parsing
- Added standardized column naming
- Better error handling for missing data

#### Infrastructure
- Added EC2 proxy support
- Added SSH tunnel management
- Added persistent tunnel state tracking
- Added automatic port management
- Added cleanup procedures

#### Data Processing
- Standardized column naming across all datasets
- Updated all DataFrame operations to use lowercase column names
- Improved data type handling
- Better handling of missing values
- Added data validation steps

### Migration Notes
- Requires Python 3.12+ for new syntax features 