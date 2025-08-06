import re
import csv
import sys
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import statistics
import os
import glob

# Optional: AWS Data Wrangler for S3 export
try:
    import awswrangler as wr
    import pandas as pd
    AWS_WRANGLER_AVAILABLE = True
except ImportError:
    AWS_WRANGLER_AVAILABLE = False

# --- Load Configuration from config.env ---
def load_config():
    """Load configuration from config.env file if it exists"""
    config_file = os.path.join(os.path.dirname(__file__), 'config.env')
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

# Optional: Elasticsearch support
try:
    from elasticsearch import Elasticsearch
except ImportError:
    Elasticsearch = None

# Regex patterns
START_END_PATTERN = re.compile(r'\[(START|END|END2)\]')
ACCOUNT_PATTERN = re.compile(r'account\[(\d+)\]')

# Improved timestamp parser for nanoseconds
def parse_timestamp(ts):
    # Truncate nanoseconds to microseconds for datetime
    if '.' in ts:
        base, rest = ts.split('.', 1)
        # Remove trailing Z if present
        rest = rest.rstrip('Z')
        # Pad/truncate to 6 digits (microseconds)
        micro = (rest + '000000')[:6]
        ts_fixed = f"{base}.{micro}Z"
    else:
        ts_fixed = ts
    try:
        return datetime.strptime(ts_fixed, "%Y-%m-%dT%H:%M:%S.%fZ")
    except Exception:
        return None

# Extract operation name (first bracketed part)
def extract_operation(line):
    match = re.search(r'\[([^\]]+)\]', line)
    if match:
        op = match.group(1)
        op = re.sub(r':\d+\)', ')', op)  # Remove line number
        return op
    return None

# Extract unique_id (account) from any part of the line
def extract_unique_id(line):
    match = ACCOUNT_PATTERN.search(line)
    return match.group(1) if match else None

# Extract marker ([START], [END], [END2])
def extract_marker(line):
    match = START_END_PATTERN.search(line)
    return match.group(1) if match else None

# Extract parameters (everything after marker)
def extract_parameters(line):
    match = re.search(r'\[(START|END|END2)\](.*)', line)
    return match.group(2).strip() if match else ''

def process_log_lines(lines):
    start_records = {}
    completed = []
    for line in lines:
        # Split into timestamp and message
        if '\t' in line:
            ts_str, msg = line.split('\t', 1)
        else:
            continue
        timestamp = parse_timestamp(ts_str)
        marker = extract_marker(msg)
        if not marker:
            continue
        operation = extract_operation(msg)
        unique_id = extract_unique_id(msg)
        parameters = extract_parameters(msg)
        if not (operation and unique_id and timestamp):
            continue
        key = f"{operation}:{unique_id}"
        if marker == "START":
            start_records[key] = (timestamp, parameters)
        elif marker in ("END", "END2"):
            if key in start_records:
                start_time, params = start_records.pop(key)
                duration_ms = (timestamp - start_time).total_seconds() * 1000
                completed.append({
                    "operation": operation,
                    "unique_id": unique_id,
                    "start_time": start_time,
                    "end_time": timestamp,
                    "duration_ms": duration_ms,
                    "parameters": params
                })
    return completed

def print_stats(completed):
    stats = defaultdict(list)
    for rec in completed:
        stats[rec["operation"]].append(rec["duration_ms"])
    print(f"\n{'Operation':40} {'Count':>5} {'Min':>8} {'Max':>8} {'Mean':>10} {'Median':>10} {'Stddev':>10}")
    print("-" * 90)
    for op, durations in stats.items():
        print(f"{op:40} {len(durations):5} {min(durations):8.2f} {max(durations):8.2f} "
              f"{statistics.mean(durations):10.2f} {statistics.median(durations):10.2f} "
              f"{statistics.stdev(durations) if len(durations) > 1 else 0:10.2f}")

def export_to_s3(local_folder, s3_bucket, s3_prefix, aws_region):
    """Export parsed folder to S3 using data wrangler"""
    if not AWS_WRANGLER_AVAILABLE:
        print("❌ AWS Data Wrangler not available. Install with: pip install awswrangler")
        return False
    
    try:
        print(f"📤 Exporting {local_folder} to s3://{s3_bucket}/{s3_prefix}/")
        print("ℹ️  Note: To make files public, configure bucket policy or use 'aws s3api put-bucket-policy'")
        
        # Upload all CSV files recursively
        uploaded_count = 0
        for root, dirs, files in os.walk(local_folder):
            for file in files:
                if file.endswith('.csv'):
                    local_path = os.path.join(root, file)
                    # Extract relative path for S3 key
                    rel_path = os.path.relpath(local_path, local_folder)
                    s3_key = f"{s3_prefix}/{rel_path}"
                    
                    print(f"  Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
                    
                    # Read CSV and upload to S3
                    df = pd.read_csv(local_path)
                    wr.s3.to_csv(
                        df=df,
                        path=f"s3://{s3_bucket}/{s3_key}",
                        index=False,
                        dataset=True  # Enable dataset features for partitioning
                    )
                    uploaded_count += 1
        
        print(f"✅ Successfully uploaded {uploaded_count} files to S3")
        print("💡 To make files public, run: aws s3api put-bucket-policy --bucket chromaway-tmp --policy '{\"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"PublicReadGetObject\",\"Effect\":\"Allow\",\"Principal\":\"*\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::chromaway-tmp/*\"}]}'")
        return True
        
    except Exception as e:
        print(f"❌ Error exporting to S3: {e}")
        return False

def fetch_logs_from_es(es_host, es_index, container, days_back, hours_back, es_user=None, es_pass=None, es_api_key=None, verify_certs=True, output_file=None, time_range=None):
    if Elasticsearch is None:
        print("elasticsearch package not installed. Please install with 'pip install elasticsearch'.")
        sys.exit(1)
    # Connect
    es_params = {'hosts': [es_host], 'verify_certs': verify_certs}
    if es_user and es_pass:
        es_params['basic_auth'] = (es_user, es_pass)
    elif es_api_key:
        es_params['api_key'] = es_api_key
    es = Elasticsearch(**es_params)
    # Time range
    if time_range:
        start, end = time_range
    else:
        now = datetime.now(timezone.utc)
        if hours_back:
            start = now - timedelta(hours=int(hours_back))
        else:
            start = now - timedelta(days=int(days_back) if days_back else 7)
        end = now
    start_iso = start.isoformat(timespec='milliseconds')
    end_iso = end.isoformat(timespec='milliseconds')
    # Query
    query = {
        "query": {
            "bool": {
                "must": [
                    {"term": {"source": "postchain"}},
                    {"term": {"container_name.keyword": container}},
                    {"range": {"timestamp": {"gte": start_iso, "lt": end_iso}}}
                ]
            }
        },
        "size": 10000,
        "sort": [{"timestamp": {"order": "asc"}}]
    }
    print(f"Querying ES: {es_host}, index: {es_index}, container: {container}, time: {start_iso} to {end_iso}")
    resp = es.search(index=es_index, body=query, scroll='2m')
    sid = resp['_scroll_id']
    scroll_size = len(resp['hits']['hits'])
    all_lines = []
    # Write to CSV in _temp
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['timestamp', 'message'])
        while scroll_size > 0:
            for hit in resp['hits']['hits']:
                src = hit['_source']
                ts = src.get('timestamp') or src.get('_source.timestamp') or src.get('utc_time') or src.get('_source.utc_time')
                msg = src.get('message') or src.get('_source.message')
                if ts and msg:
                    writer.writerow([ts, msg])
                    all_lines.append(f"{ts}\t{msg}")
            resp = es.scroll(scroll_id=sid, scroll='2m')
            sid = resp['_scroll_id']
            scroll_size = len(resp['hits']['hits'])
        # Fix scroll error by wrapping in try-catch
        try:
            es.clear_scroll(scroll_id=sid)
        except Exception:
            pass  # Ignore scroll clear errors
    print(f"Fetched {len(all_lines)} log lines from Elasticsearch. Saved to {output_file}")
    return output_file, all_lines

def find_csv_files_in_folder(folder):
    """Recursively find all CSV files in a folder (date/hour partitioned)"""
    pattern = os.path.join(folder, '**', '*.csv')
    return sorted(glob.glob(pattern, recursive=True))

def main():
    load_config()
    import argparse
    parser = argparse.ArgumentParser(description="Blockchain Log Analyzer (CSV or Elasticsearch)")
    parser.add_argument('input', nargs='?', help='CSV file or folder to analyze (if not using --es)')
    parser.add_argument('--es', action='store_true', help='Query Elasticsearch and analyze (default: download to _temp and analyze)')
    parser.add_argument('--analyze', metavar='FILE_OR_FOLDER', help='Analyze a previously downloaded file or folder in _temp (skip ES download)')
    parser.add_argument('--export-s3', action='store_true', help='Export parsed results to S3')
    parser.add_argument('--es-host', default=os.getenv('ES_HOST', 'http://localhost:9200'))
    parser.add_argument('--es-index', default=os.getenv('ES_INDEX', 'logstash-*'))
    parser.add_argument('--es-user', default=os.getenv('ES_USERNAME'))
    parser.add_argument('--es-pass', default=os.getenv('ES_PASSWORD'))
    parser.add_argument('--es-api-key', default=os.getenv('ES_API_KEY'))
    parser.add_argument('--es-container', default=os.getenv('CONTAINER_NAME', ''))
    parser.add_argument('--days-back', type=int, default=int(os.getenv('DAYS_BACK', '7')))
    parser.add_argument('--hours-back', type=int, default=None if os.getenv('HOURS_BACK') is None else int(os.getenv('HOURS_BACK')))
    parser.add_argument('--no-verify-certs', action='store_true', help='Disable SSL cert verification')
    parser.add_argument('--s3-bucket', default=os.getenv('S3_BUCKET'))
    parser.add_argument('--s3-prefix', default=os.getenv('S3_PREFIX', 'parsed-logs'))
    parser.add_argument('--aws-region', default=os.getenv('AWS_REGION', 'us-east-1'))
    args = parser.parse_args()

    def parse_and_write(lines, output_file, date=None, hour=None):
        completed = process_log_lines(lines)
        if completed:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', newline='') as f:
                fieldnames = ['date', 'hour', 'operation', 'unique_id', 'start_time', 'end_time', 'duration_ms', 'parameters']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for rec in completed:
                    row = rec.copy()
                    row['start_time'] = row['start_time'].isoformat()
                    row['end_time'] = row['end_time'].isoformat()
                    row['date'] = date
                    row['hour'] = hour
                    writer.writerow(row)
            print(f"\nParsed results written to: {output_file}")
        return completed

    all_completed = []
    if args.analyze:
        input_path = args.analyze
        if os.path.isdir(input_path):
            csv_files = find_csv_files_in_folder(input_path)
            for csv_file in csv_files:
                # Extract date/hour from path if possible
                date, hour = None, None
                m = re.search(r'date=(\d{4}-\d{2}-\d{2})', csv_file)
                if m:
                    date = m.group(1)
                m = re.search(r'hour=(\d{2})', csv_file)
                if m:
                    hour = m.group(1)
                with open(csv_file, newline='') as f:
                    sample = f.read(2048)
                    f.seek(0)
                    import csv as _csv
                    sniffer = _csv.Sniffer()
                    try:
                        dialect = sniffer.sniff(sample, delimiters='\t,;')
                    except _csv.Error:
                        dialect = _csv.excel
                    reader = _csv.DictReader(f, dialect=dialect)
                    ts_candidates = ['_source.timestamp', '_source.utc_time', 'timestamp', '@timestamp', 'time', 'date']
                    msg_candidates = ['_source.message', 'message', 'log', 'entry']
                    ts_col = next((fld for fld in reader.fieldnames if fld and fld.lower() in [c.lower() for c in ts_candidates]), None)
                    msg_col = next((fld for fld in reader.fieldnames if fld and fld.lower() in [c.lower() for c in msg_candidates]), None)
                    if not ts_col:
                        ts_col = reader.fieldnames[0] if reader.fieldnames else None
                    if not msg_col:
                        msg_col = reader.fieldnames[1] if reader.fieldnames and len(reader.fieldnames) > 1 else None
                    if not ts_col or not msg_col:
                        print(f"Error: Could not find suitable timestamp and message columns in {csv_file}.")
                        print(f"Columns found: {reader.fieldnames}")
                        continue
                    lines = [f"{row[ts_col]}\t{row[msg_col]}" for row in reader if row.get(ts_col) and row.get(msg_col)]
                # Write parsed output to partitioned folder
                parsed_dir = os.path.join('_temp', 'parsed', f'date={date}' if date else '', f'hour={hour}' if hour else '')
                parsed_file = os.path.join(parsed_dir, f'parsed_{date.replace("-", "") if date else ""}_{hour if hour else ""}.csv')
                all_completed.extend(parse_and_write(lines, parsed_file, date, hour))
        else:
            # Single file
            with open(input_path, newline='') as csvfile:
                sample = csvfile.read(2048)
                csvfile.seek(0)
                import csv as _csv
                sniffer = _csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample, delimiters='\t,;')
                except _csv.Error:
                    dialect = _csv.excel
                reader = _csv.DictReader(csvfile, dialect=dialect)
                ts_candidates = ['_source.timestamp', '_source.utc_time', 'timestamp', '@timestamp', 'time', 'date']
                msg_candidates = ['_source.message', 'message', 'log', 'entry']
                ts_col = next((f for f in reader.fieldnames if f and f.lower() in [c.lower() for c in ts_candidates]), None)
                msg_col = next((f for f in reader.fieldnames if f and f.lower() in [c.lower() for c in msg_candidates]), None)
                if not ts_col:
                    ts_col = reader.fieldnames[0] if reader.fieldnames else None
                if not msg_col:
                    msg_col = reader.fieldnames[1] if reader.fieldnames and len(reader.fieldnames) > 1 else None
                if not ts_col or not msg_col:
                    print(f"Error: Could not find suitable timestamp and message columns in {input_path}.")
                    print(f"Columns found: {reader.fieldnames}")
                    sys.exit(1)
                lines = [f"{row[ts_col]}\t{row[msg_col]}" for row in reader if row.get(ts_col) and row.get(msg_col)]
            # Try to extract date/hour from filename
            date, hour = None, None
            m = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})', input_path)
            if m:
                date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                hour = m.group(4)
            parsed_dir = os.path.join('_temp', 'parsed', f'date={date}' if date else '', f'hour={hour}' if hour else '')
            parsed_file = os.path.join(parsed_dir, f'parsed_{date.replace("-", "") if date else ""}_{hour if hour else ""}.csv')
            all_completed.extend(parse_and_write(lines, parsed_file, date, hour))
    elif args.es:
        # Download from ES to _temp/es_export/date=YYYY-MM-DD/hour=HH/, then analyze
        # (Implementation for partitioned download by hour)
        if not args.es_container:
            print("--es-container is required for ES download.")
            sys.exit(1)
        now = datetime.now(timezone.utc)
        if args.hours_back:
            start = now - timedelta(hours=int(args.hours_back))
        else:
            start = now - timedelta(days=int(args.days_back) if args.days_back else 7)
        end = now
        # Build list of (date, hour) tuples to fetch
        hours = []
        cur = start.replace(minute=0, second=0, microsecond=0)
        while cur <= end:
            hours.append((cur.strftime('%Y-%m-%d'), cur.strftime('%H')))
            cur += timedelta(hours=1)
        for date, hour in hours:
            out_dir = os.path.join('_temp', 'es_export', f'date={date}', f'hour={hour}')
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f'es_export_{date.replace("-", "")}_{hour}.csv')
            if os.path.exists(out_file):
                print(f"Skipping download for {date} {hour}: already exists.")
                continue
            # Query ES for this hour
            hour_start = datetime.strptime(f'{date} {hour}', '%Y-%m-%d %H').replace(tzinfo=timezone.utc)
            hour_end = hour_start + timedelta(hours=1)
            temp_file, lines = fetch_logs_from_es(
                es_host=args.es_host,
                es_index=args.es_index,
                container=args.es_container,
                days_back=None,
                hours_back=None,
                es_user=args.es_user,
                es_pass=args.es_pass,
                es_api_key=args.es_api_key,
                verify_certs=not args.no_verify_certs,
                output_file=out_file,
                time_range=(hour_start, hour_end)
            )
            # Parse and write
            parsed_dir = os.path.join('_temp', 'parsed', f'date={date}', f'hour={hour}')
            parsed_file = os.path.join(parsed_dir, f'parsed_{date.replace("-", "")}_{hour}.csv')
            all_completed.extend(parse_and_write(lines, parsed_file, date, hour))
    else:
        if not args.input:
            print("Usage: python log_analyzer.py <logfile.csv|folder> or --es ...")
            sys.exit(1)
        input_path = args.input
        if os.path.isdir(input_path):
            csv_files = find_csv_files_in_folder(input_path)
            for csv_file in csv_files:
                date, hour = None, None
                m = re.search(r'date=(\d{4}-\d{2}-\d{2})', csv_file)
                if m:
                    date = m.group(1)
                m = re.search(r'hour=(\d{2})', csv_file)
                if m:
                    hour = m.group(1)
                with open(csv_file, newline='') as f:
                    sample = f.read(2048)
                    f.seek(0)
                    import csv as _csv
                    sniffer = _csv.Sniffer()
                    try:
                        dialect = sniffer.sniff(sample, delimiters='\t,;')
                    except _csv.Error:
                        dialect = _csv.excel
                    reader = _csv.DictReader(f, dialect=dialect)
                    ts_candidates = ['_source.timestamp', '_source.utc_time', 'timestamp', '@timestamp', 'time', 'date']
                    msg_candidates = ['_source.message', 'message', 'log', 'entry']
                    ts_col = next((fld for fld in reader.fieldnames if fld and fld.lower() in [c.lower() for c in ts_candidates]), None)
                    msg_col = next((fld for fld in reader.fieldnames if fld and fld.lower() in [c.lower() for c in msg_candidates]), None)
                    if not ts_col:
                        ts_col = reader.fieldnames[0] if reader.fieldnames else None
                    if not msg_col:
                        msg_col = reader.fieldnames[1] if reader.fieldnames and len(reader.fieldnames) > 1 else None
                    if not ts_col or not msg_col:
                        print(f"Error: Could not find suitable timestamp and message columns in {csv_file}.")
                        print(f"Columns found: {reader.fieldnames}")
                        continue
                    lines = [f"{row[ts_col]}\t{row[msg_col]}" for row in reader if row.get(ts_col) and row.get(msg_col)]
                parsed_dir = os.path.join('_temp', 'parsed', f'date={date}' if date else '', f'hour={hour}' if hour else '')
                parsed_file = os.path.join(parsed_dir, f'parsed_{date.replace("-", "") if date else ""}_{hour if hour else ""}.csv')
                all_completed.extend(parse_and_write(lines, parsed_file, date, hour))
        else:
            with open(input_path, newline='') as csvfile:
                sample = csvfile.read(2048)
                csvfile.seek(0)
                import csv as _csv
                sniffer = _csv.Sniffer()
                try:
                    dialect = sniffer.sniff(sample, delimiters='\t,;')
                except _csv.Error:
                    dialect = _csv.excel
                reader = _csv.DictReader(csvfile, dialect=dialect)
                ts_candidates = ['_source.timestamp', '_source.utc_time', 'timestamp', '@timestamp', 'time', 'date']
                msg_candidates = ['_source.message', 'message', 'log', 'entry']
                ts_col = next((f for f in reader.fieldnames if f and f.lower() in [c.lower() for c in ts_candidates]), None)
                msg_col = next((f for f in reader.fieldnames if f and f.lower() in [c.lower() for c in msg_candidates]), None)
                if not ts_col:
                    ts_col = reader.fieldnames[0] if reader.fieldnames else None
                if not msg_col:
                    msg_col = reader.fieldnames[1] if reader.fieldnames and len(reader.fieldnames) > 1 else None
                if not ts_col or not msg_col:
                    print(f"Error: Could not find suitable timestamp and message columns in {input_path}.")
                    print(f"Columns found: {reader.fieldnames}")
                    sys.exit(1)
                lines = [f"{row[ts_col]}\t{row[msg_col]}" for row in reader if row.get(ts_col) and row.get(msg_col)]
            date, hour = None, None
            m = re.search(r'(\d{4})(\d{2})(\d{2})_(\d{2})', input_path)
            if m:
                date = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                hour = m.group(4)
            parsed_dir = os.path.join('_temp', 'parsed', f'date={date}' if date else '', f'hour={hour}' if hour else '')
            parsed_file = os.path.join(parsed_dir, f'parsed_{date.replace("-", "") if date else ""}_{hour if hour else ""}.csv')
            all_completed.extend(parse_and_write(lines, parsed_file, date, hour))
    
    print_stats(all_completed)
    
    # Export to S3 if requested
    if args.export_s3:
        if not args.s3_bucket:
            print("❌ S3_BUCKET environment variable or --s3-bucket required for S3 export")
            sys.exit(1)
        
        print(f"\n📤 Exporting parsed results to s3://{args.s3_bucket}/{args.s3_prefix}/")
        success = export_to_s3(
            local_folder='_temp/parsed',
            s3_bucket=args.s3_bucket,
            s3_prefix=args.s3_prefix,
            aws_region=args.aws_region
        )
        if success:
            print(f"✅ S3 export completed: s3://{args.s3_bucket}/{args.s3_prefix}/")
        else:
            print("❌ S3 export failed")

if __name__ == "__main__":
    main()