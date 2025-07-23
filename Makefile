# Makefile for mna-logs-analyzer

# Default log file to parse
default_log=test_data.csv
parsed_csv=parsed_testdata.csv

.PHONY: all parse streamlit

all: parse streamlit

parse:
	python3 log_analyzer.py $(default_log)
	mv -f test_data_parsed.csv $(parsed_csv)

streamlit:
	streamlit run streamlit_app.py 