#!/usr/bin/env bash
set -euo pipefail

USER_AGENT_DEFAULT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_INPUT_DIR="${PROJECT_ROOT}/scripts/data"
DEFAULT_OUTPUT_DIR="${PROJECT_ROOT}/data/pdf"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/download_leaflets.sh [INPUT_JSON_OR_DIR] [OUTPUT_DIR]

Environment overrides:
  MAX_RETRIES       default: 2
  CONNECT_TIMEOUT   default: 15
  MAX_TIME          default: 120
  USER_AGENT        default: Chrome-like UA

This downloader reads the userscript export JSON and uses curl to download
leaflet PDFs via the captured attachment URL / download endpoint.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required but not installed." >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required but not installed." >&2
  exit 1
fi

INPUT_PATH="${1:-$DEFAULT_INPUT_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"
MAX_RETRIES="${MAX_RETRIES:-2}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-15}"
MAX_TIME="${MAX_TIME:-120}"
USER_AGENT="${USER_AGENT:-$USER_AGENT_DEFAULT}"

mkdir -p "$OUTPUT_DIR"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${OUTPUT_DIR%/}/${RUN_TAG}"
mkdir -p "$RUN_DIR"

TASKS_TSV="${RUN_DIR}/tasks.tsv"
RESULTS_JSONL="${RUN_DIR}/download_results.jsonl"
SUMMARY_JSON="${RUN_DIR}/download_summary.json"
COOKIE_JAR="${RUN_DIR}/cde_cookie.txt"

sanitize_filename() {
  local value="$1"
  value="${value//$'\n'/_}"
  value="${value//$'\r'/_}"
  value="${value//$'\t'/_}"
  value="${value//\//_}"
  value="${value//\\/_}"
  value="${value## }"
  value="${value%% }"
  if [[ -z "$value" ]]; then
    value="leaflet.pdf"
  fi
  if [[ "${value,,}" != *.pdf ]]; then
    value="${value}.pdf"
  fi
  printf '%s' "$value"
}

extract_tasks_from_file() {
  local input_json="$1"
  jq -r '
    .detail_pages[]
    | .url as $detail_url
    | .attachment_links[]
    | {
        detail_url: $detail_url,
        attachment_text: (.file_name // .text // "leaflet.pdf"),
        attachment_url: (
          if (.download_endpoint // "") != "" then .download_endpoint
          elif (.href // "") != "" and (.href != $detail_url) then .href
          else ""
          end
        )
      }
    | select(.attachment_url != "")
    | [.detail_url, .attachment_url, .attachment_text]
    | @tsv
  ' "$input_json"
}

append_result() {
  local index="$1"
  local detail_url="$2"
  local attachment_url="$3"
  local attachment_text="$4"
  local status="$5"
  local file_path="$6"
  local status_code="$7"
  local size_bytes="$8"
  local content_type="$9"
  local error="${10}"

  jq -nc \
    --argjson index "$index" \
    --arg detail_url "$detail_url" \
    --arg attachment_url "$attachment_url" \
    --arg attachment_text "$attachment_text" \
    --arg status "$status" \
    --arg file_path "$file_path" \
    --arg status_code "$status_code" \
    --arg size_bytes "$size_bytes" \
    --arg content_type "$content_type" \
    --arg error "$error" \
    '{
      index: $index,
      detail_url: $detail_url,
      attachment_url: $attachment_url,
      attachment_text: $attachment_text,
      status: $status,
      file_path: (if $file_path == "" then null else $file_path end),
      status_code: (if $status_code == "" then null else ($status_code | tonumber) end),
      size_bytes: (if $size_bytes == "" then null else ($size_bytes | tonumber) end),
      content_type: (if $content_type == "" then null else $content_type end),
      error: (if $error == "" then null else $error end)
    }' >>"$RESULTS_JSONL"
}

declare -a input_files=()
if [[ -d "$INPUT_PATH" ]]; then
  while IFS= read -r file; do
    input_files+=("$file")
  done < <(find "$INPUT_PATH" -maxdepth 1 -type f -name '*.json' | sort)
elif [[ -f "$INPUT_PATH" ]]; then
  input_files+=("$INPUT_PATH")
else
  echo "Input path not found: $INPUT_PATH" >&2
  exit 1
fi

if [[ "${#input_files[@]}" -eq 0 ]]; then
  echo "No JSON manifest files found under: $INPUT_PATH" >&2
  exit 1
fi

for input_json in "${input_files[@]}"; do
  extract_tasks_from_file "$input_json"
done | awk -F '\t' '!seen[$2]++' >"$TASKS_TSV"
TOTAL_TASKS="$(wc -l <"$TASKS_TSV" | tr -d ' ')"

if [[ "$TOTAL_TASKS" -eq 0 ]]; then
  echo "No attachment links found in input JSON." >&2
  exit 1
fi

echo "[1/3] Loaded ${TOTAL_TASKS} unique attachment links"

downloaded_count=0
skipped_count=0
failed_count=0
index=0

while IFS=$'\t' read -r detail_url attachment_url attachment_text; do
  index=$((index + 1))
  safe_name="$(sanitize_filename "$attachment_text")"
  url_hash="$(printf '%s' "$attachment_url" | sha1sum | awk '{print substr($1,1,10)}')"
  target_name="$(printf '%05d_%s_%s' "$index" "$url_hash" "$safe_name")"
  target_path="${RUN_DIR}/${target_name}"

  if [[ -f "$target_path" ]]; then
    skipped_count=$((skipped_count + 1))
    append_result "$index" "$detail_url" "$attachment_url" "$attachment_text" "skipped_exists" "$target_path" "" "" "" ""
    continue
  fi

  headers_file="$(mktemp)"
  body_file="$(mktemp)"
  curl_status=0
  : >"$COOKIE_JAR"

  if ! curl -sS \
    -c "$COOKIE_JAR" \
    -A "$USER_AGENT" \
    "$attachment_url" \
    -o /dev/null; then
    curl_status=$?
  fi

  if [[ "$curl_status" -eq 0 ]] && ! curl -fLsS \
    --retry "$MAX_RETRIES" \
    --retry-all-errors \
    --connect-timeout "$CONNECT_TIMEOUT" \
    --max-time "$MAX_TIME" \
    -b "$COOKIE_JAR" \
    -A "$USER_AGENT" \
    -H "Referer: ${detail_url}" \
    -D "$headers_file" \
    -o "$body_file" \
    "$attachment_url"; then
    curl_status=$?
  fi

  status_code="$(awk 'BEGIN{code=""} /^HTTP\// {code=$2} END{print code}' "$headers_file" | tr -d '\r')"
  content_type="$(awk 'BEGIN{IGNORECASE=1} /^content-type:/ {sub(/\r$/,""); print substr($0,15)}' "$headers_file" | tail -n 1 | sed 's/^ *//')"
  size_bytes="$(wc -c <"$body_file" | tr -d ' ')"

  if [[ "$curl_status" -ne 0 ]]; then
    failed_count=$((failed_count + 1))
    append_result "$index" "$detail_url" "$attachment_url" "$attachment_text" "failed_request" "" "$status_code" "$size_bytes" "$content_type" "curl exited with status ${curl_status}"
    rm -f "$headers_file" "$body_file"
    continue
  fi

  if [[ "$(head -c 4 "$body_file" 2>/dev/null || true)" == "%PDF" ]]; then
    mv "$body_file" "$target_path"
    downloaded_count=$((downloaded_count + 1))
    append_result "$index" "$detail_url" "$attachment_url" "$attachment_text" "downloaded" "$target_path" "$status_code" "$size_bytes" "$content_type" ""
  else
    failed_count=$((failed_count + 1))
    append_result "$index" "$detail_url" "$attachment_url" "$attachment_text" "failed_non_pdf" "" "$status_code" "$size_bytes" "$content_type" "Response does not look like a PDF"
    rm -f "$body_file"
  fi

  rm -f "$headers_file"
done <"$TASKS_TSV"

echo "[2/3] Downloaded ${downloaded_count}, skipped ${skipped_count}, failed ${failed_count}"

jq -n \
  --arg created_at_utc "$(date -u --iso-8601=seconds)" \
  --arg input_path "$INPUT_PATH" \
  --arg output_run_dir "$RUN_DIR" \
  --argjson input_files_count "${#input_files[@]}" \
  --argjson total_tasks "$TOTAL_TASKS" \
  --argjson downloaded_count "$downloaded_count" \
  --argjson skipped_count "$skipped_count" \
  --argjson failed_count "$failed_count" \
  --slurpfile results "$RESULTS_JSONL" \
  '{
    created_at_utc: $created_at_utc,
    input_path: $input_path,
    input_files_count: $input_files_count,
    output_run_dir: $output_run_dir,
    total_tasks: $total_tasks,
    downloaded_count: $downloaded_count,
    skipped_count: $skipped_count,
    failed_count: $failed_count,
    downloader: "curl-shell",
    results: $results
  }' >"$SUMMARY_JSON"

echo "[3/3] Wrote summary to ${SUMMARY_JSON}"
