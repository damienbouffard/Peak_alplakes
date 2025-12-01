#!/usr/bin/env bash
# Execute a notebook and write outputs to an executed notebook file
# Usage: ./run_notebook.sh path/to/notebook.ipynb
set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 notebook.ipynb"
  exit 1
fi
NOTEBOOK="$1"
OUTNAME="executed-$(basename "$NOTEBOOK")"
python -m nbconvert --to notebook --execute "$NOTEBOOK" --output "$OUTNAME" --ExecutePreprocessor.timeout=600
echo "Executed notebook saved as $OUTNAME"
