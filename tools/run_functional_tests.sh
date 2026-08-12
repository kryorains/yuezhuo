#!/usr/bin/env bash
set -u

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TEST_ROOT=${1:-"$ROOT_DIR/examples/functional"}
TARGET=${TARGET:-riscv64}
COMPILER=${COMPILER:-"$ROOT_DIR/target/debug/compiler"}
RUNTIME=${RUNTIME:-"$ROOT_DIR/tools/sylib.c"}
WORK_DIR=${WORK_DIR:-"/tmp/yuezhuo-functional-tests"}
RUN_TIMEOUT=${RUN_TIMEOUT:-5s}
COMPILER_FLAGS=${COMPILER_FLAGS:-}
read -r -a compiler_flags <<< "$COMPILER_FLAGS"
HOST_SYSTEM=$(uname -s)
HOST_ARCH=$(uname -m)

case "$TARGET" in
  riscv64|riscv64gc)
    if [[ "$HOST_SYSTEM" == Linux && "$HOST_ARCH" =~ ^(riscv64|riscv64gc)$ ]]; then
      CC=${CC:-gcc}
      RUNNER=${RUNNER:-}
    else
      CC=${CC:-riscv64-linux-gnu-gcc}
      RUNNER=${RUNNER:-"qemu-riscv64 -L /usr/riscv64-linux-gnu"}
    fi
    ;;
  *)
    printf 'Unknown TARGET=%s\n' "$TARGET" >&2
    exit 2
    ;;
esac
read -r -a runner_args <<< "$RUNNER"

if [[ ! -x "$COMPILER" ]]; then
  cargo build --manifest-path "$ROOT_DIR/Cargo.toml" >/dev/null || exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

total=0
passed=0
failed=0
compile_fail=0
link_fail=0
run_fail=0
timeout_fail=0
diff_fail=0

while IFS= read -r sy; do
  out=${sy%.sy}.out
  input=${sy%.sy}.in
  rel=${sy#"$ROOT_DIR/"}
  stem=${rel//\//__}
  stem=${stem%.sy}
  asm="$WORK_DIR/$stem.s"
  exe="$WORK_DIR/$stem.exe"
  actual="$WORK_DIR/$stem.actual"
  expected="$WORK_DIR/$stem.expected"
  log="$WORK_DIR/$stem.log"

  if [[ ! -f "$out" ]]; then
    continue
  fi

  total=$((total + 1))

  if ! "$COMPILER" "$sy" -S -o "$asm" --target "$TARGET" "${compiler_flags[@]}" >"$log" 2>&1; then
    printf 'COMPILE_FAIL %s\n' "$rel"
    failed=$((failed + 1))
    compile_fail=$((compile_fail + 1))
    continue
  fi

  if ! "$CC" "$asm" "$RUNTIME" -o "$exe" -lm >>"$log" 2>&1; then
    printf 'LINK_FAIL    %s\n' "$rel"
    failed=$((failed + 1))
    link_fail=$((link_fail + 1))
    continue
  fi

  if [[ -f "$input" ]]; then
    timeout "$RUN_TIMEOUT" "${runner_args[@]}" "$exe" <"$input" >"$actual" 2>>"$log"
  else
    timeout "$RUN_TIMEOUT" "${runner_args[@]}" "$exe" </dev/null >"$actual" 2>>"$log"
  fi
  status=$?

  if (( status == 124 )); then
    printf 'TIMEOUT      %s\n' "$rel"
    failed=$((failed + 1))
    timeout_fail=$((timeout_fail + 1))
    continue
  fi

  if [[ -s "$actual" ]] && [[ $(tail -c 1 "$actual" | wc -l) -eq 0 ]]; then
    printf '\n' >>"$actual"
  fi
  printf '%s\n' "$status" >>"$actual"
  cp "$out" "$expected"

  if diff -u "$expected" "$actual" >"$WORK_DIR/$stem.diff"; then
    passed=$((passed + 1))
  elif (( status >= 128 )); then
    printf 'RUN_FAIL     %s status=%s\n' "$rel" "$status"
    failed=$((failed + 1))
    run_fail=$((run_fail + 1))
  else
    printf 'DIFF_FAIL    %s\n' "$rel"
    failed=$((failed + 1))
    diff_fail=$((diff_fail + 1))
  fi
done < <(find "$TEST_ROOT" -name '*.sy' | sort)

printf '\nTOTAL=%s PASS=%s FAIL=%s COMPILE_FAIL=%s LINK_FAIL=%s RUN_FAIL=%s TIMEOUT=%s DIFF_FAIL=%s\n' \
  "$total" "$passed" "$failed" "$compile_fail" "$link_fail" "$run_fail" "$timeout_fail" "$diff_fail"
printf 'ARTIFACTS=%s\n' "$WORK_DIR"

if (( failed != 0 )); then
  exit 1
fi
