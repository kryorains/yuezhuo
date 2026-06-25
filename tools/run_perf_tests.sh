#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_ROOT=${1:-"$ROOT_DIR/examples/perf-riscv64"}
TARGET=${TARGET:-riscv64}
COMPILER=${COMPILER:-"$ROOT_DIR/target/debug/compiler"}
RUNTIME=${RUNTIME:-"$ROOT_DIR/tools/sylib.c"}
WORK_DIR=${WORK_DIR:-"$ROOT_DIR/target/test-work/perf-tests"}
RUN_TIMEOUT=${RUN_TIMEOUT:-30s}
MAX_INPUT_BYTES=${MAX_INPUT_BYTES:-1048576}
OPT_LEVELS=${OPT_LEVELS:-"O0 O1"}
TEST_FILTER=${TEST_FILTER:-}
BASELINE=${BASELINE:-0}
BASELINE_OPT=${BASELINE_OPT:-"-O2"}
BASELINE_CFLAGS=${BASELINE_CFLAGS:-"-x c -std=gnu99 -Wno-implicit-function-declaration"}

case "$TARGET" in
  x86_64|x86-64|amd64)
    CC=${CC:-gcc}
    RUNNER=${RUNNER:-}
    ;;
  aarch64|arm64)
    CC=${CC:-aarch64-linux-gnu-gcc}
    RUNNER=${RUNNER:-"qemu-aarch64 -L /usr/aarch64-linux-gnu"}
    ;;
  riscv64|riscv64gc)
    CC=${CC:-riscv64-linux-gnu-gcc}
    RUNNER=${RUNNER:-"qemu-riscv64 -L /usr/riscv64-linux-gnu"}
    ;;
  *)
    printf 'Unknown TARGET=%s\n' "$TARGET" >&2
    exit 2
    ;;
esac

read -r -a runner_args <<< "$RUNNER"
read -r -a opt_levels <<< "$OPT_LEVELS"
read -r -a baseline_opt_args <<< "$BASELINE_OPT"
read -r -a baseline_cflag_args <<< "$BASELINE_CFLAGS"

if [[ ! -x "$COMPILER" ]]; then
  cargo build --manifest-path "$ROOT_DIR/Cargo.toml" >/dev/null || exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
report="$WORK_DIR/report.tsv"
printf 'test\topt\tstatus\tcompile_ms\tlink_ms\trun_ms\tasm_lines\tinput_bytes\tbaseline_ms\tscore\n' >"$report"
{
  printf 'target=%s\n' "$TARGET"
  printf 'cc=%s\n' "$CC"
  "$CC" --version | head -1
  printf 'baseline=%s\n' "$BASELINE"
  printf 'baseline_opt=%s\n' "$BASELINE_OPT"
  printf 'run_timeout=%s\n' "$RUN_TIMEOUT"
  printf 'max_input_bytes=%s\n' "$MAX_INPUT_BYTES"
} >"$WORK_DIR/config.txt"

total=0
passed=0
failed=0
skipped=0

now_ms() {
  date +%s%3N
}

elapsed_ms() {
  local start=$1
  local end=$2
  printf '%s' "$((end - start))"
}

input_size() {
  local input=$1
  if [[ -f "$input" ]]; then
    wc -c <"$input"
  else
    printf '0'
  fi
}

run_exe() {
  local exe=$1
  local input=$2
  local actual=$3
  local log=$4

  if [[ -f "$input" ]]; then
    timeout "$RUN_TIMEOUT" "${runner_args[@]}" "$exe" <"$input" >"$actual" 2>>"$log"
  else
    timeout "$RUN_TIMEOUT" "${runner_args[@]}" "$exe" >"$actual" 2>>"$log"
  fi
}

append_status_and_compare() {
  local actual=$1
  local expected=$2
  local out=$3
  local status=$4
  local diff_path=$5

  if [[ -s "$actual" ]] && [[ $(tail -c 1 "$actual" | wc -l) -eq 0 ]]; then
    printf '\n' >>"$actual"
  fi
  printf '%s\n' "$status" >>"$actual"
  cp "$out" "$expected"
  diff -u "$expected" "$actual" >"$diff_path"
}

score_ratio() {
  local baseline_ms=$1
  local run_ms=$2
  if [[ "$baseline_ms" == "NA" || "$run_ms" == "0" ]]; then
    printf 'NA'
  else
    awk -v b="$baseline_ms" -v r="$run_ms" 'BEGIN { printf "%.2f", (b / r) * 100.0 }'
  fi
}

while IFS= read -r sy; do
  out=${sy%.sy}.out
  input=${sy%.sy}.in
  rel=${sy#"$ROOT_DIR/"}

  if [[ ! -f "$out" ]]; then
    continue
  fi

  if [[ -n "$TEST_FILTER" && "$rel" != *"$TEST_FILTER"* ]]; then
    continue
  fi

  bytes=$(input_size "$input")
  if (( bytes > MAX_INPUT_BYTES )); then
    skipped=$((skipped + 1))
    continue
  fi

  stem=${rel//\//__}
  stem=${stem%.sy}
  baseline_ms=NA

  if [[ "$BASELINE" == "1" ]]; then
    baseline_exe="$WORK_DIR/$stem.gcc.exe"
    baseline_actual="$WORK_DIR/$stem.gcc.actual"
    baseline_expected="$WORK_DIR/$stem.expected"
    baseline_log="$WORK_DIR/$stem.gcc.log"

    if "$CC" "${baseline_cflag_args[@]}" "${baseline_opt_args[@]}" "$sy" "$RUNTIME" -o "$baseline_exe" -lm >"$baseline_log" 2>&1; then
      start=$(now_ms)
      run_exe "$baseline_exe" "$input" "$baseline_actual" "$baseline_log"
      baseline_status=$?
      baseline_ms=$(elapsed_ms "$start" "$(now_ms)")

      if (( baseline_status == 124 )); then
        baseline_ms=NA
        printf 'BASELINE_TIMEOUT %s\n' "$rel"
      elif ! append_status_and_compare "$baseline_actual" "$baseline_expected" "$out" "$baseline_status" "$WORK_DIR/$stem.gcc.diff"; then
        baseline_ms=NA
        printf 'BASELINE_DIFF_FAIL %s\n' "$rel"
      fi
    else
      printf 'BASELINE_COMPILE_FAIL %s\n' "$rel"
    fi
  fi

  for opt in "${opt_levels[@]}"; do
    total=$((total + 1))
    asm="$WORK_DIR/$stem.$opt.s"
    exe="$WORK_DIR/$stem.$opt.exe"
    actual="$WORK_DIR/$stem.$opt.actual"
    expected="$WORK_DIR/$stem.expected"
    log="$WORK_DIR/$stem.$opt.log"

    compiler_flags=()
    if [[ "$opt" != "O0" ]]; then
      compiler_flags=("-${opt}")
    fi

    start=$(now_ms)
    if ! "$COMPILER" "$sy" -S -o "$asm" --target "$TARGET" "${compiler_flags[@]}" >"$log" 2>&1; then
      end=$(now_ms)
      printf '%s\t%s\tCOMPILE_FAIL\t%s\t0\t0\t0\t%s\t%s\tNA\n' "$rel" "$opt" "$(elapsed_ms "$start" "$end")" "$bytes" "$baseline_ms" >>"$report"
      printf 'COMPILE_FAIL %s %s\n' "$opt" "$rel"
      failed=$((failed + 1))
      continue
    fi
    compile_ms=$(elapsed_ms "$start" "$(now_ms)")
    asm_lines=$(wc -l <"$asm")

    start=$(now_ms)
    if ! "$CC" "$asm" "$RUNTIME" -o "$exe" -lm >>"$log" 2>&1; then
      end=$(now_ms)
      printf '%s\t%s\tLINK_FAIL\t%s\t%s\t0\t%s\t%s\t%s\tNA\n' "$rel" "$opt" "$compile_ms" "$(elapsed_ms "$start" "$end")" "$asm_lines" "$bytes" "$baseline_ms" >>"$report"
      printf 'LINK_FAIL    %s %s\n' "$opt" "$rel"
      failed=$((failed + 1))
      continue
    fi
    link_ms=$(elapsed_ms "$start" "$(now_ms)")

    start=$(now_ms)
    run_exe "$exe" "$input" "$actual" "$log"
    run_status=$?
    run_ms=$(elapsed_ms "$start" "$(now_ms)")

    if (( run_status == 124 )); then
      printf '%s\t%s\tTIMEOUT\t%s\t%s\t%s\t%s\t%s\t%s\tNA\n' "$rel" "$opt" "$compile_ms" "$link_ms" "$run_ms" "$asm_lines" "$bytes" "$baseline_ms" >>"$report"
      printf 'TIMEOUT      %s %s\n' "$opt" "$rel"
      failed=$((failed + 1))
      continue
    fi

    score=$(score_ratio "$baseline_ms" "$run_ms")
    if append_status_and_compare "$actual" "$expected" "$out" "$run_status" "$WORK_DIR/$stem.$opt.diff"; then
      printf '%s\t%s\tPASS\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "$opt" "$compile_ms" "$link_ms" "$run_ms" "$asm_lines" "$bytes" "$baseline_ms" "$score" >>"$report"
      passed=$((passed + 1))
    else
      printf '%s\t%s\tDIFF_FAIL\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "$opt" "$compile_ms" "$link_ms" "$run_ms" "$asm_lines" "$bytes" "$baseline_ms" "$score" >>"$report"
      printf 'DIFF_FAIL    %s %s\n' "$opt" "$rel"
      failed=$((failed + 1))
    fi
  done
done < <(find "$TEST_ROOT" -name '*.sy' | sort)

printf '\nTOTAL=%s PASS=%s FAIL=%s SKIPPED_BY_SIZE=%s\n' "$total" "$passed" "$failed" "$skipped"
printf 'REPORT=%s\n' "$report"
printf 'ARTIFACTS=%s\n' "$WORK_DIR"

if (( failed != 0 )); then
  exit 1
fi
