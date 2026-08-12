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
TEST_LIMIT=${TEST_LIMIT:-0}
BASELINE=${BASELINE:-0}
BASELINE_OPT=${BASELINE_OPT:-"-O2"}
BASELINE_CFLAGS=${BASELINE_CFLAGS:-"-x c -std=gnu99 -Wno-implicit-function-declaration"}
BASELINE_NORMALIZE=${BASELINE_NORMALIZE:-1}
LINK_CFLAGS=${LINK_CFLAGS:-"-O2"}
FAIL_ON_CASE_ERROR=${FAIL_ON_CASE_ERROR:-1}

case "$TARGET" in
  x86_64|x86-64|amd64)
    CC=${CC:-gcc}
    RUNNER=${RUNNER:-}
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
read -r -a link_cflag_args <<< "$LINK_CFLAGS"

if [[ ! -x "$COMPILER" ]]; then
  cargo build --manifest-path "$ROOT_DIR/Cargo.toml" >/dev/null || exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"
printf 'TARGET=%s\n' "$TARGET"
printf 'CC=%s\n' "$CC"
"$CC" --version | head -1
printf 'BASELINE=%s\n' "$BASELINE"
printf 'OPT_LEVELS=%s\n' "$OPT_LEVELS"
printf 'TEST_LIMIT=%s\n' "$TEST_LIMIT"
printf 'RUN_TIMEOUT=%s\n' "$RUN_TIMEOUT"
printf 'MAX_INPUT_BYTES=%s\n' "$MAX_INPUT_BYTES"
printf 'FAIL_ON_CASE_ERROR=%s\n' "$FAIL_ON_CASE_ERROR"

total=0
passed=0
failed=0
skipped=0
processed=0

now_ms() {
  date +%s%3N
}

elapsed_ms() {
  local start=$1
  local end=$2
  printf '%s' "$((end - start))"
}

print_result() {
  local status=$1
  local opt=$2
  local test=$3
  local compile_ms=$4
  local link_ms=$5
  local run_ms=$6
  local asm_lines=$7
  local input_bytes=$8
  local baseline_ms=$9
  local score=${10}
  local total_ms=$((compile_ms + link_ms + run_ms))

  printf '%-18s %-3s %10d %10d %10d %10d %9d %11d %11s %7s  %s\n' \
    "$status" "$opt" "$compile_ms" "$link_ms" "$run_ms" "$total_ms" \
    "$asm_lines" "$input_bytes" "$baseline_ms" "$score" "$test"
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

  normalize_text_file "$actual"
  if [[ -s "$actual" ]] && [[ $(tail -c 1 "$actual" | wc -l) -eq 0 ]]; then
    printf '\n' >>"$actual"
  fi
  printf '%s\n' "$status" >>"$actual"

  tr -d '\r' <"$out" >"$expected"
  if [[ -s "$expected" ]] && [[ $(tail -c 1 "$expected" | wc -l) -eq 0 ]]; then
    printf '\n' >>"$expected"
  fi
  diff -u "$expected" "$actual" >"$diff_path"
}

normalize_text_file() {
  local path=$1
  local tmp=$path.norm
  tr -d '\r' <"$path" >"$tmp"
  mv "$tmp" "$path"
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

normalize_sysy_for_gcc() {
  local src=$1
  local dst=$2
  perl -pe 's{^(\s*)const\s+int\s+([A-Za-z_]\w*)\s*=\s*([-+]?(?:0[xX][0-9A-Fa-f]+|\d+))\s*;\s*(//.*)?$}{$1 . "enum { $2 = $3 };" . (defined($4) ? " $4" : "")}e' "$src" >"$dst"
}

printf '\n%-18s %-3s %10s %10s %10s %10s %9s %11s %11s %7s  %s\n' \
  STATUS OPT COMPILE_MS LINK_MS RUN_MS TOTAL_MS ASM_LINES INPUT_BYTES BASELINE_MS SCORE TEST
suite_start_ms=$(now_ms)

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

  if (( TEST_LIMIT > 0 && processed >= TEST_LIMIT )); then
    break
  fi
  processed=$((processed + 1))

  stem=${rel//\//__}
  stem=${stem%.sy}
  baseline_ms=NA
  baseline_ok=0

  if [[ "$BASELINE" == "1" ]]; then
    baseline_exe="$WORK_DIR/$stem.gcc.exe"
    baseline_actual="$WORK_DIR/$stem.gcc.actual"
    baseline_expected="$WORK_DIR/$stem.expected"
    baseline_log="$WORK_DIR/$stem.gcc.log"

    baseline_src=$sy
    if [[ "$BASELINE_NORMALIZE" == "1" ]]; then
      baseline_src="$WORK_DIR/$stem.gcc.c"
      normalize_sysy_for_gcc "$sy" "$baseline_src"
    fi

    if "$CC" "${baseline_cflag_args[@]}" "${baseline_opt_args[@]}" "$baseline_src" "$RUNTIME" -o "$baseline_exe" -lm >"$baseline_log" 2>&1; then
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
      else
        baseline_ok=1
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
      compile_ms=$(elapsed_ms "$start" "$(now_ms)")
      print_result COMPILE_FAIL "$opt" "$rel" "$compile_ms" 0 0 0 "$bytes" "$baseline_ms" NA
      failed=$((failed + 1))
      continue
    fi
    compile_ms=$(elapsed_ms "$start" "$(now_ms)")
    asm_lines=$(wc -l <"$asm")

    start=$(now_ms)
    if ! "$CC" "${link_cflag_args[@]}" "$asm" "$RUNTIME" -o "$exe" -lm >>"$log" 2>&1; then
      link_ms=$(elapsed_ms "$start" "$(now_ms)")
      print_result LINK_FAIL "$opt" "$rel" "$compile_ms" "$link_ms" 0 \
        "$asm_lines" "$bytes" "$baseline_ms" NA
      failed=$((failed + 1))
      continue
    fi
    link_ms=$(elapsed_ms "$start" "$(now_ms)")

    start=$(now_ms)
    run_exe "$exe" "$input" "$actual" "$log"
    run_status=$?
    run_ms=$(elapsed_ms "$start" "$(now_ms)")

    if (( run_status == 124 )); then
      print_result TIMEOUT "$opt" "$rel" "$compile_ms" "$link_ms" "$run_ms" \
        "$asm_lines" "$bytes" "$baseline_ms" NA
      failed=$((failed + 1))
      continue
    fi

    score=$(score_ratio "$baseline_ms" "$run_ms")
    if append_status_and_compare "$actual" "$expected" "$out" "$run_status" "$WORK_DIR/$stem.$opt.diff"; then
      status=PASS
      passed=$((passed + 1))
    else
      status=DIFF_FAIL
      if [[ "$BASELINE" == "1" && "$baseline_ok" == "0" ]]; then
        status=EXPECTED_MISMATCH
      fi
      if [[ "$status" == "EXPECTED_MISMATCH" ]]; then
        skipped=$((skipped + 1))
      else
        failed=$((failed + 1))
      fi
    fi
    print_result "$status" "$opt" "$rel" "$compile_ms" "$link_ms" "$run_ms" \
      "$asm_lines" "$bytes" "$baseline_ms" "$score"
  done
done < <(find "$TEST_ROOT" -name '*.sy' | sort)

suite_ms=$(elapsed_ms "$suite_start_ms" "$(now_ms)")
printf '\nFILES=%s CASES=%s PASS=%s NON_PASS=%s SKIPPED=%s TOTAL_MS=%s\n' \
  "$processed" "$total" "$passed" "$failed" "$skipped" "$suite_ms"

if (( failed != 0 )) && [[ "$FAIL_ON_CASE_ERROR" == "1" ]]; then
  exit 1
fi
