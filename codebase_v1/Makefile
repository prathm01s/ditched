# --- Compiler and Flags ---
CXX = g++
# Use -O2 for release speed, -Wall for warnings
LDFLAGS = -std=c++17 -O2 -Wall

# --- Python ---
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python3
PIP = $(VENV_DIR)/bin/pip

# --- Executables ---
# List all executables that are compiled WITHOUT special flags
GENERIC_EXECUTABLES = monotone_chain optimized_monotone_chain graham_scan

# List all executables that need special flags
SPECIAL_EXECUTABLES = quickhull

# All executables to be built by 'make all'
ALL_EXECUTABLES = $(GENERIC_EXECUTABLES) $(SPECIAL_EXECUTABLES)

# --- Default list of algos to benchmark ---
# Users can override this from the command line, e.g., make benchmark ALGO_LIST="..."
ALGO_LIST ?= ./monotone_chain ./optimized_monotone_chain ./graham_scan ./quickhull

# --- Main Targets ---

.PHONY: all setup generate benchmark clean

# Compile all executables
all: $(ALL_EXECUTABLES)

# Set up the Python virtual environment and install packages
setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate:
	@echo "--- 🐍 Setting up Python Virtual Environment ---"
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(PIP) install -q --upgrade pip
	$(PIP) install -q numpy matplotlib scipy
	@echo "--- Setup complete. Activate with: source venv/bin/activate ---"

# Generate all datasets
generate: setup
	@echo "--- Generating datasets ---"
	$(PYTHON) generate_cases.py
	$(PYTHON) dataset_gen.py

# Run the benchmark on the algorithms specified in ALGO_LIST
benchmark: all setup
	@echo "--- 📊 Running Benchmark on: $(ALGO_LIST) ---"
	$(PYTHON) benchmark.py $(ALGO_LIST)
	@echo "--- Benchmark complete. Plots saved to PNG. ---"

# --- Compilation Rules ---

# Rule for QuickHull (needs -fopenmp and assumes quickhull.cpp)
quickhull: quickhull.cpp
	$(CXX) -o $@ $< $(LDFLAGS) -fopenmp

# Generic rule for all other C++ files
# This matches: monotone_chain, optimized_monotone_chain, graham_scan
# It works by matching the target name to the .cpp file name
$(GENERIC_EXECUTABLES): %: %.cpp
	$(CXX) -o $@ $< $(LDFLAGS)

# --- Cleanup ---

clean:
	@echo "--- 🧹 Cleaning up ---"
	rm -f $(ALL_EXECUTABLES)
	rm -f *.png
	rm -rf MONOTONE_OUTPUT OPTIMIZED_MONOTONE_OUTPUT QUICKHULL_OUTPUT GRAHAM_SCAN_OUTPUT SCIPY_OUTPUT
	rm -rf __pycache__