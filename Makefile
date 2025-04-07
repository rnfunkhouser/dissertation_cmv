# Makefile

.PHONY: all stage1 stage2 stage3 stage4 stage5 stage6 clean

# Run all stages in order
all: stage1 stage2 stage3 stage4 stage5 stage6

# Each stage runs one Python script from the 'code/' folder
stage1:
	@echo "Running Stage 1: Extract comment IDs from deltalog (JSON)..."
	@mkdir -p data
	python3 code/1.extract_comment_ids_from_deltalog_JSON_version.py

stage2:
	@echo "Running Stage 2: Extract comment IDs from deltalog (after JSON)..."
	@mkdir -p data
	python3 code/2.extract_comment_ids_from_deltalog_after_json.py

stage3:
	@echo "Running Stage 3: Combine delta IDs CSVs..."
	@mkdir -p data
	python3 code/3.combine_delta_ids_csvs.py

stage4:
	@echo "Running Stage 4: Add delta info to CMV CSV..."
	@mkdir -p data
	python3 code/4.add_delta_info_to_cmv_csv.py

stage5:
	@echo "Running Stage 5: Trim CMV CSV..."
	@mkdir -p data
	python3 code/5.trimming_csv.py

stage6:
	@echo "Running Stage 6: Execute core conversation filtering..."
	@mkdir -p data
	python3 code/6.execute_core_convo_filtering.py

# Optional cleanup
clean:
	@echo "Cleaning up generated files..."
	rm -f data/*.tmp