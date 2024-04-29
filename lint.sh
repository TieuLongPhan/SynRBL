#!/bin/bash

flake8 . --count --max-complexity=13 --max-line-length=90 \
	--per-file-ignores="__init__.py:F401 \
		mcs_graph_detector.py:C901 \
		eda_analysis.py:C901 \
		find_missing_graphs.py:C901" \
	--exclude venv \
	--statistics
