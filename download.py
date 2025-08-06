from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError
import csv
from datetime import datetime, timedelta, timezone
import sys
import os
import time
import math

# --- Load Configuration from config.env ---
def load_config():
    """Load configuration from config.env file if it exists"""
    config_file = 'config.env'
    if os.path.exists(config_file):
        print(f"📋 Loading configuration from {config_file}")
        with open(config_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Parse key=value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        # Set environment variable
                        os.environ[key] = value
        print("✅ Configuration loaded successfully")
    else:
        print(f"⚠️  {config_file} not found, using default values")

# Load configuration at startup
load_config()

# --- Configuration ---
# Replace these with your actual values
ES_HOST = os.getenv('ES_HOST', 'http://localhost:9200')
ES_USERNAME = os.getenv('ES_USERNAME', None)
ES_PASSWORD = os.getenv('ES_PASSWORD', None)
ES_API_KEY = os.getenv('ES_API_KEY', None)
ES_VERIFY_CERTS = os.getenv('ES_VERIFY_CERTS', 'true').lower() == 'true'

INDEX_NAME = os.getenv('ES_INDEX', 'your_log_index_pattern')  # e.g., 'logstash-*'
CONTAINER_NAME = os.getenv('CONTAINER_NAME', '02ECFAE9-611062cdf1047b265b652d229858e5a18dbf487afbb62f0f6d30ec4fd81366f9-20')
DAYS_BACK = int(os.getenv('DAYS_BACK', '7'))
HOURS_BACK = os.getenv('HOURS_BACK')  # Will be None if not set
USE_UTC_TIME = os.getenv('USE_UTC_TIME', 'true').lower() == 'true'

# Auto-generate output filename based on ES_INDEX if not explicitly set
OUTPUT_FILE = os.getenv('OUTPUT_FILE')
if not OUTPUT_FILE:
    # Sanitize index name for filename (remove wildcards and invalid chars)
    def sanitize_index_name(index_name):
        """Remove wildcards and invalid filename characters from index name"""
        # Remove wildcards
        sanitized = index_name.replace('*', '').replace('?', '')
        # Remove other invalid filename characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '')
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip('. ')
        # If empty after sanitization, use a default name
        if not sanitized:
            sanitized = 'elasticsearch_data'
        return sanitized
    
    safe_index_name = sanitize_index_name(INDEX_NAME)
    OUTPUT_FILE = f"{safe_index_name}_exported_data.csv"

# Performance and memory management settings
MAX_RECORDS = int(os.getenv('MAX_RECORDS', '1000000'))  # Max records to export (1M default)
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))  # Records per scroll batch
SCROLL_TIMEOUT = os.getenv('SCROLL_TIMEOUT', '5m')  # Scroll timeout
ENABLE_PROGRESS_BAR = os.getenv('ENABLE_PROGRESS_BAR', 'true').lower() == 'true'
CHUNKED_OUTPUT = os.getenv('CHUNKED_OUTPUT', 'false').lower() == 'true'  # Split into multiple files

# Fields to export - customize these based on your actual field names
FIELDS_TO_EXPORT = [
    "timestamp", 
    "message",
    "container_name",
    "source",
    "context_blockRID",
    "context_blockchainRID", 
    "context_chainIID"
]

# --- Elasticsearch Connection ---
def create_es_client():
    """Create and return Elasticsearch client with proper authentication"""
    try:
        print(f"🔌 Attempting to connect to: {ES_HOST}")
        
        # Build connection parameters
        es_params = {
            'hosts': [ES_HOST],
            'verify_certs': ES_VERIFY_CERTS,
            'request_timeout': 30,  # Connection timeout
            'max_retries': 3,  # Retry failed requests
            'retry_on_timeout': True
        }
        
        # Add authentication if provided
        if ES_USERNAME and ES_PASSWORD:
            print(f"🔐 Using basic authentication with username: {ES_USERNAME}")
            es_params['basic_auth'] = (ES_USERNAME, ES_PASSWORD)
        elif ES_API_KEY:
            print("🔑 Using API key authentication")
            es_params['api_key'] = ES_API_KEY
        else:
            print("⚠️  No authentication provided")
            
        es = Elasticsearch(**es_params)
        
        # Test connection
        print("🔄 Testing connection...")
        if not es.ping():
            raise ConnectionError("Failed to connect to Elasticsearch - ping failed")
            
        print("✅ Successfully connected to Elasticsearch")
        return es
        
    except Exception as e:
        print(f"❌ Error creating Elasticsearch client: {e}")
        print(f"   Host: {ES_HOST}")
        print(f"   Username: {ES_USERNAME}")
        print(f"   Verify certs: {ES_VERIFY_CERTS}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if Elasticsearch is running")
        print("   2. Verify the hostname and port")
        print("   3. Check network connectivity")
        print("   4. Verify credentials")
        print("   5. Try: curl -u username:password http://host:9200")
        sys.exit(1)

# --- Time Range Setup ---
def get_server_time(es):
    """Get current time from Elasticsearch server"""
    try:
        # Get cluster info which includes timestamp
        info = es.info()
        # Elasticsearch info doesn't directly give us server time, so we'll use UTC
        # But we can verify the connection is working
        print(f"✅ Connected to Elasticsearch cluster: {info.get('cluster_name', 'Unknown')}")
        
        if USE_UTC_TIME:
            return datetime.now(timezone.utc)
        else:
            return datetime.now()  # Local time
    except Exception as e:
        print(f"⚠️  Could not get server time, using local time: {e}")
        if USE_UTC_TIME:
            return datetime.now(timezone.utc)
        else:
            return datetime.now()

def get_time_range(es, days_back=None, hours_back=None):
    """Generate ISO formatted time range for the last N days or hours using UTC"""
    # Get current time in UTC (server time)
    end_time = get_server_time(es)
    
    if hours_back is not None:
        # Use hours if specified (takes precedence over days)
        start_time = end_time - timedelta(hours=int(hours_back))
        time_unit = f"{hours_back} hours"
    elif days_back is not None:
        # Use days if hours not specified
        start_time = end_time - timedelta(days=days_back)
        time_unit = f"{days_back} days"
    else:
        # Default to 7 days if neither specified
        start_time = end_time - timedelta(days=7)
        time_unit = "7 days"
    
    # Format for Elasticsearch query (ISO 8601 format)
    if USE_UTC_TIME:
        start_time_iso = start_time.isoformat(timespec='milliseconds')
        end_time_iso = end_time.isoformat(timespec='milliseconds')
        timezone_info = "UTC"
    else:
        start_time_iso = start_time.isoformat(timespec='milliseconds')
        end_time_iso = end_time.isoformat(timespec='milliseconds')
        timezone_info = "Local"
    
    print(f"🕐 Using {timezone_info} time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 Time range: {start_time.strftime('%Y-%m-%d %H:%M:%S')} to {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    return start_time_iso, end_time_iso, time_unit

# --- Output Directory Setup ---
def setup_output_directory():
    """Create data directory with datetime timestamp and return the path"""
    # Create timestamp for directory name using UTC
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = f"data/{timestamp}"
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory created: {output_dir}")
    
    return output_dir

# --- Query Builder ---
def build_query(container_name, start_time, end_time):
    """Build the Elasticsearch query"""
    return {
        "query": {
            "bool": {
                "must": [
                    {"term": {"source": "postchain"}},
                    {"term": {"container_name.keyword": container_name}},
                    {"range": {"timestamp": {"gte": start_time, "lte": end_time}}}
                ]
            }
        },
        "size": CHUNK_SIZE,
        "sort": [{"timestamp": {"order": "asc"}}]
    }

# --- Field Flattening ---
def flatten_nested_fields(data, fields):
    """Flatten nested fields like 'context.blockRID' to 'context_blockRID', always include all context fields even if missing."""
    flattened = {}
    context_fields = [
        "context_blockRID",
        "context_blockchainRID",
        "context_chainIID"
    ]
    for field in fields:
        if field.startswith("context_"):
            # Extract context fields
            context = data.get("context", {})
            subfield = field.split("_", 1)[1]
            flattened[field] = context.get(subfield) if isinstance(context, dict) else None
        elif '.' in field:
            # Handle other nested fields like 'host.name'
            keys = field.split('.')
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    value = None
                    break
            flattened[field.replace('.', '_')] = value
        else:
            flattened[field] = data.get(field)
    # Ensure all context fields are present
    for cf in context_fields:
        if cf not in flattened:
            flattened[cf] = None
    return flattened

# --- Progress Bar ---
def print_progress(current, total, start_time):
    """Print a simple progress bar"""
    if not ENABLE_PROGRESS_BAR:
        return
    
    if total > 0:
        percentage = min(100, (current / total) * 100)
        bar_length = 50
        filled_length = int(bar_length * current // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Calculate ETA
        elapsed_time = time.time() - start_time
        if current > 0:
            eta_seconds = (elapsed_time / current) * (total - current)
            eta_str = f"ETA: {eta_seconds:.0f}s"
        else:
            eta_str = "ETA: calculating..."
        
        print(f"\rProgress: [{bar}] {percentage:.1f}% ({current:,}/{total:,}) {eta_str}", end='', flush=True)

# --- Chunked File Writer ---
class ChunkedCSVWriter:
    """Handles writing large datasets to multiple CSV files"""
    
    def __init__(self, base_filename, fields, records_per_file=100000):
        self.base_filename = base_filename
        self.fields = fields
        self.records_per_file = records_per_file
        self.current_file = None
        self.current_writer = None
        self.current_chunk = 0
        self.records_in_current_file = 0
        self.total_records = 0
        
    def __enter__(self):
        self._open_new_file()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_file:
            self.current_file.close()
            
    def _open_new_file(self):
        """Open a new CSV file for the current chunk"""
        if self.current_file:
            self.current_file.close()
            
        # Generate filename with chunk number
        if self.current_chunk == 0:
            filename = f"{self.base_filename}"
        else:
            name, ext = os.path.splitext(self.base_filename)
            filename = f"{name}_chunk_{self.current_chunk:03d}{ext}"
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
            
        self.current_file = open(filename, 'w', newline='', encoding='utf-8')
        self.current_writer = csv.DictWriter(self.current_file, fieldnames=self.fields)
        self.current_writer.writeheader()
        self.records_in_current_file = 0
        
        if self.current_chunk > 0:
            print(f"\nStarted new file: {filename}")
            
    def write_row(self, row):
        """Write a row to the current file, creating new files as needed"""
        if self.records_in_current_file >= self.records_per_file:
            self.current_chunk += 1
            self._open_new_file()
            
        self.current_writer.writerow(row)
        self.records_in_current_file += 1
        self.total_records += 1

# --- Main Export Function ---
def export_to_csv(es, query_body, output_file, fields_to_export):
    """Export data from Elasticsearch to CSV using scroll API with memory management"""
    start_time = time.time()
    
    try:
        # Initial search with scroll
        print(f"Starting search with query...")
        page = es.search(
            index=INDEX_NAME,
            body=query_body,
            scroll=SCROLL_TIMEOUT
        )
        
        sid = page['_scroll_id']
        scroll_size = len(page['hits']['hits'])
        total_hits = page['hits']['total']['value'] if isinstance(page['hits']['total'], dict) else page['hits']['total']
        
        print(f"Total hits found: {total_hits:,}")
        print(f"Max records to export: {MAX_RECORDS:,}")
        print(f"Chunk size: {CHUNK_SIZE}")
        print(f"Scroll timeout: {SCROLL_TIMEOUT}")
        
        if total_hits > MAX_RECORDS:
            print(f"⚠️  WARNING: Found {total_hits:,} records but will only export {MAX_RECORDS:,}")
            print(f"   Consider reducing the time range or increasing MAX_RECORDS")
        
        # Determine actual export limit
        export_limit = min(total_hits, MAX_RECORDS)
        
        if CHUNKED_OUTPUT and export_limit > 20000:
            # Use chunked output for large datasets
            print(f"Using chunked output mode (100k records per file)")
            with ChunkedCSVWriter(output_file, fields_to_export, 20000) as writer:
                _process_scroll_results(es, page, sid, writer, export_limit, start_time)
        else:
            # Use single file output
            print(f"Using single file output mode")
            # Ensure the directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fields_to_export)
                writer.writeheader()
                _process_scroll_results(es, page, sid, writer, export_limit, start_time)
        
        # Clear the scroll context
        es.clear_scroll(scroll_id=sid)
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Export completed!")
        print(f"   Records exported: {export_limit:,}")
        print(f"   Time elapsed: {elapsed_time:.1f} seconds")
        print(f"   Average speed: {export_limit/elapsed_time:.0f} records/second")
        
        if CHUNKED_OUTPUT and export_limit > 100000:
            print(f"   Files created: {math.ceil(export_limit/100000)}")
            
    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        # Try to clear scroll context even if there was an error
        try:
            if 'sid' in locals():
                es.clear_scroll(scroll_id=sid)
        except:
            pass
        sys.exit(1)

def _process_scroll_results(es, initial_page, initial_sid, writer, export_limit, start_time):
    """Process scroll results with progress tracking and debug output for first 3 records"""
    page = initial_page
    sid = initial_sid
    processed_count = 0
    debug_printed = 0
    
    while processed_count < export_limit:
        for hit in page['hits']['hits']:
            if processed_count >= export_limit:
                break
                
            source_data = hit['_source']
            # Debug: print first 3 records' _source
            if debug_printed < 3:
                print("\n--- DEBUG: _source for record ---")
                print(source_data)
                debug_printed += 1
            flattened_data = flatten_nested_fields(source_data, FIELDS_TO_EXPORT)
            
            if hasattr(writer, 'write_row'):
                # ChunkedCSVWriter
                writer.write_row(flattened_data)
            else:
                # Regular CSV writer
                writer.writerow(flattened_data)
                
            processed_count += 1
            
            # Print progress every 1000 records
            if processed_count % 1000 == 0:
                print_progress(processed_count, export_limit, start_time)
        
        if processed_count >= export_limit:
            break
            
        # Get next page of results
        try:
            page = es.scroll(scroll_id=sid, scroll=SCROLL_TIMEOUT)
            sid = page['_scroll_id']
            scroll_size = len(page['hits']['hits'])
            
            if scroll_size == 0:
                break
                
        except Exception as e:
            print(f"\n❌ Error during scroll: {e}")
            break
    
    # Final progress update
    print_progress(processed_count, export_limit, start_time)
    print()  # New line after progress bar

# --- Main Execution ---
def main():
    print("🚀 Starting Elasticsearch data export...")
    print(f"Configuration:")
    print(f"  ES Host: {ES_HOST}")
    print(f"  Index: {INDEX_NAME}")
    print(f"  Container Name: {CONTAINER_NAME}")
    print(f"  Days back: {DAYS_BACK}")
    print(f"  Hours back: {HOURS_BACK if HOURS_BACK else 'Not set'}")
    print(f"  Use UTC time: {USE_UTC_TIME}")
    print(f"  Max records: {MAX_RECORDS:,}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Chunked output: {CHUNKED_OUTPUT}")
    print(f"  Fields to export: {len(FIELDS_TO_EXPORT)} fields")
    
    # Show filename sanitization info
    if '*' in INDEX_NAME or '?' in INDEX_NAME:
        print(f"📝 Note: Index pattern '{INDEX_NAME}' contains wildcards - filename will be sanitized")
    
    # Setup output directory
    output_dir = setup_output_directory()
    output_file = os.path.join(output_dir, OUTPUT_FILE)
    print(f"  Output file: {output_file}")
    
    # Create ES client
    es = create_es_client()
    print("✅ Connected to Elasticsearch")
    
    # Get time range
    start_time, end_time, time_unit = get_time_range(es, DAYS_BACK, HOURS_BACK)
    print(f"Time range: {start_time} to {end_time} ({time_unit} back)")
    
    # Debug: Show what the time range means in different timezones
    if USE_UTC_TIME:
        print(f"🔍 Debug: This query will search for logs from the last {time_unit} in UTC time")
        print(f"   If your logs are stored in a different timezone, you may need to adjust HOURS_BACK")
    else:
        print(f"🔍 Debug: This query will search for logs from the last {time_unit} in your local timezone")
        print(f"   If your logs are stored in UTC, consider setting USE_UTC_TIME=true")
    
    # Build query
    query_body = build_query(CONTAINER_NAME, start_time, end_time)
    
    # Export data
    export_to_csv(es, query_body, output_file, FIELDS_TO_EXPORT)

if __name__ == "__main__":
    main()