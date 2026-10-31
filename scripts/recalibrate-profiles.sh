#!/usr/bin/env bash
set -euo pipefail

# Source JSONL must contain independently audited SourceItem records. Use a fresh output.
source_file=${1:?usage: recalibrate-profiles.sh SOURCE.jsonl OUTPUT_DIR [HORIZONS_PER_PROFILE]}
output_dir=${2:?usage: recalibrate-profiles.sh SOURCE.jsonl OUTPUT_DIR [HORIZONS_PER_PROFILE]}
horizons=${3:-300}

distill gateway profile-build --source "$source_file" --output "$output_dir" --workload-version gateway-clients-v5 --horizons-per-profile "$horizons"
distill gateway profile-validate --input-dir "$output_dir" --output "$output_dir/validation.json" --min-horizons "$horizons"
harness calibrate --fitting "$output_dir/reference_fit.detector.jsonl" --score-calibration "$output_dir/score_calibration.detector.jsonl" --policy-calibration "$output_dir/policy_development.detector.jsonl" --task gateway --validation-report "$output_dir/validation.json" --mixture-control-required --output-dir "$output_dir/calibration"
harness adequacy --calibration "$output_dir/calibration" --output "$output_dir/adequacy.json"
