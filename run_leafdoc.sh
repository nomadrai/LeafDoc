#!/usr/bin/env bash
# LeafDoc MATLAB Launcher Script
# Usage:
#   ./run_leafdoc.sh            -> Launches the modern LeafDoc_App GUI
#   ./run_leafdoc.sh classic    -> Launches the classic DetectDisease_GUI
#   ./run_leafdoc.sh script     -> Runs the standalone Detect.m script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if command -v matlab >/dev/null 2>&1; then
    MATLAB_BIN="$(command -v matlab)"
elif [ -x "$HOME/MATLAB/R2024b/bin/matlab" ]; then
    MATLAB_BIN="$HOME/MATLAB/R2024b/bin/matlab"
else
    MATLAB_BIN="$(find "$HOME/MATLAB" /usr/local/MATLAB -name "matlab" -type f -executable 2>/dev/null | head -n 1)"
fi

if [ -z "$MATLAB_BIN" ] || [ ! -x "$MATLAB_BIN" ]; then
    echo "Error: MATLAB executable not found. Please ensure MATLAB is installed and in your PATH."
    exit 1
fi

MODE="${1:-app}"
cd "$SCRIPT_DIR" || exit 1

if [ "$MODE" = "classic" ]; then
    echo "Launching Classic DetectDisease_GUI in MATLAB..."
    "$MATLAB_BIN" -sd "$SCRIPT_DIR" -r "DetectDisease_GUI"
elif [ "$MODE" = "script" ]; then
    echo "Running standalone Detect.m script in MATLAB..."
    "$MATLAB_BIN" -sd "$SCRIPT_DIR" -r "Detect"
else
    echo "Launching LeafDoc_App Dashboard in MATLAB..."
    "$MATLAB_BIN" -sd "$SCRIPT_DIR" -r "LeafDoc_App"
fi
