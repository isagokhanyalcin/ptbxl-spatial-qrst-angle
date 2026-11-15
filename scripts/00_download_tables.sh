#!/usr/bin/env bash
# Downloads the PTB-XL v1.0.3 and PTB-XL+ v1.0.1 tables used by the analysis (PhysioNet open S3 mirror, CC BY 4.0).
set -euo pipefail
cd "$(dirname "$0")/../data"
B=https://physionet-open.s3.amazonaws.com
for f in ptb-xl-plus/1.0.1/features/unig_features.csv ptb-xl-plus/1.0.1/features/12sl_features.csv \
         ptb-xl/1.0.3/ptbxl_database.csv ptb-xl/1.0.3/scp_statements.csv; do
  echo "downloading $f"; curl -sSf -o "$(basename "$f")" "$B/$f"
done
