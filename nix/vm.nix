{
  pkgs,
  linuxPkgs,
}:

let
  alpineImage = pkgs.fetchurl {
    name = "generic-alpine-3.24.1-aarch64-uefi-cloudinit-r0.qcow2";
    url = "https://dl-cdn.alpinelinux.org/alpine/v3.24/releases/cloud/generic_alpine-3.24.1-aarch64-uefi-cloudinit-r0.qcow2";
    hash = "sha256-MFmmKAl3whIpgmMuAxfF3b05Bp1Gyh5gSA3igwkfcg8=";
  };

  linuxQemuUser = linuxPkgs.qemu-user;
in
pkgs.writeShellApplication {
  name = "yuezhuo-vm";

  runtimeInputs = [
    pkgs.coreutils
    pkgs.git
    pkgs.openssh
    pkgs.qemu
    pkgs.xorriso
  ];

  text = ''
    project_root="''${YUEZHUO_ROOT:-}"
    if [[ -z "$project_root" ]]; then
      project_root=$(git rev-parse --show-toplevel 2>/dev/null) || {
        printf 'Run this command inside the Yuezhuo repository or set YUEZHUO_ROOT.\n' >&2
        exit 2
      }
    fi

    state_dir="''${YUEZHUO_VM_STATE:-$project_root/target/qemu-vm}"
    ssh_port="''${YUEZHUO_VM_SSH_PORT:-2222}"
    memory="''${YUEZHUO_VM_MEMORY:-2048}"
    cpus="''${YUEZHUO_VM_CPUS:-4}"
    disk="$state_dir/alpine.qcow2"
    seed="$state_dir/seed.iso"
    vars="$state_dir/edk2-vars.fd"
    key="$state_dir/ssh-key"
    pid_file="$state_dir/qemu.pid"
    log_file="$state_dir/console.log"
    monitor="$state_dir/monitor.sock"
    guest="alpine@127.0.0.1"
    self=$0

    ssh_options=(
      -i "$key"
      -p "$ssh_port"
      -o BatchMode=yes
      -o ConnectTimeout=2
      -o LogLevel=ERROR
      -o StrictHostKeyChecking=no
      -o UserKnownHostsFile=/dev/null
    )

    scp_options=(
      -i "$key"
      -P "$ssh_port"
      -q
      -o BatchMode=yes
      -o ConnectTimeout=2
      -o LogLevel=ERROR
      -o StrictHostKeyChecking=no
      -o UserKnownHostsFile=/dev/null
    )

    usage() {
      printf '%s\n' \
        'Usage: yuezhuo-vm <command>' \
        '  init                              Create the writable disk, SSH key and cloud-init seed' \
        '  up                                Start the VM in the background and wait for SSH' \
        '  start                             Start the VM in the foreground' \
        '  down                              Stop the background VM' \
        '  status                            Show whether the VM is running' \
        '  ssh [command ...]                 Open a shell or run a guest command' \
        '  run --target <target> <executable> Run a Linux ELF through the guest' \
        '  test-functional [target] [path]   Run the functional suite through the guest' \
        '  test-perf [target] [path]         Run the performance suite through the guest'
    }

    is_running() {
      [[ -f "$pid_file" ]] || return 1
      local pid
      pid=$(<"$pid_file")
      kill -0 "$pid" 2>/dev/null
    }

    make_seed() {
      local public_key
      public_key=$(<"$key.pub")

      printf '%s\n' \
        'instance-id: yuezhuo-linux-v2' \
        'local-hostname: yuezhuo-linux' \
        >"$state_dir/meta-data"

      {
        printf '%s\n' \
          '#cloud-config' \
          'hostname: yuezhuo-linux' \
          'manage_etc_hosts: true' \
          'ssh_pwauth: false' \
          'disable_root: true' \
          'growpart:' \
          '  mode: auto' \
          '  devices: [/]' \
          'resize_rootfs: true' \
          'users:' \
          '  - default' \
          'ssh_authorized_keys:'
        printf '  - %s\n' "$public_key"
        printf '%s\n' \
          'mounts:' \
          '  - [nix-store, /nix/store, 9p, "trans=virtio,version=9p2000.L,ro", "0", "0"]' \
          'runcmd:' \
          '  - [sh, -c, "test -x ${linuxQemuUser}/bin/qemu-riscv64"]'
      } >"$state_dir/user-data"

      xorriso -as mkisofs \
        -quiet \
        -output "$seed" \
        -volid cidata \
        -joliet \
        -rock \
        "$state_dir/user-data" \
        "$state_dir/meta-data"
    }

    init_vm() {
      mkdir -p "$state_dir"

      if [[ ! -f "$key" ]]; then
        ssh-keygen -q -t ed25519 -N "" -C yuezhuo-vm -f "$key"
      fi

      make_seed

      if [[ ! -f "$vars" ]]; then
        cp ${pkgs.qemu}/share/qemu/edk2-arm-vars.fd "$vars"
        chmod u+w "$vars"
      fi

      if [[ ! -f "$disk" ]]; then
        qemu-img create \
          -q \
          -f qcow2 \
          -F qcow2 \
          -b ${alpineImage} \
          "$disk"
        qemu-img resize -q "$disk" 4G
      fi
    }

    qemu_arguments() {
      local accelerator cpu
      case "$(uname -s)-$(uname -m)" in
        Darwin-arm64|Darwin-aarch64)
          accelerator="''${YUEZHUO_VM_ACCELERATOR:-hvf}"
          cpu="''${YUEZHUO_VM_CPU:-host}"
          ;;
        Linux-aarch64|Linux-arm64)
          if [[ -c /dev/kvm ]]; then
            accelerator="''${YUEZHUO_VM_ACCELERATOR:-kvm}"
            cpu="''${YUEZHUO_VM_CPU:-host}"
          else
            accelerator="''${YUEZHUO_VM_ACCELERATOR:-tcg}"
            cpu="''${YUEZHUO_VM_CPU:-max}"
          fi
          ;;
        *)
          accelerator="''${YUEZHUO_VM_ACCELERATOR:-tcg}"
          cpu="''${YUEZHUO_VM_CPU:-max}"
          ;;
      esac

      qemu_args=(
        -name yuezhuo-linux
        -machine virt
        -accel "$accelerator"
        -cpu "$cpu"
        -smp "$cpus"
        -m "$memory"
        -drive "if=pflash,format=raw,readonly=on,file=${pkgs.qemu}/share/qemu/edk2-aarch64-code.fd"
        -drive "if=pflash,format=raw,file=$vars"
        -drive "if=virtio,format=qcow2,file=$disk"
        -drive "if=virtio,format=raw,readonly=on,file=$seed"
        -virtfs "local,path=/nix/store,mount_tag=nix-store,security_model=none,readonly=on"
        -device "virtio-net-pci,netdev=net0"
        -netdev "user,id=net0,hostfwd=tcp:127.0.0.1:$ssh_port-:22"
      )
    }

    wait_vm() {
      local _attempt
      for _attempt in $(seq 1 90); do
        if ssh "''${ssh_options[@]}" "$guest" \
          'test -d /nix/store && test -x ${linuxQemuUser}/bin/qemu-riscv64' \
          </dev/null 2>/dev/null; then
          printf 'Yuezhuo Linux VM is ready on SSH port %s.\n' "$ssh_port"
          return 0
        fi

        if [[ -f "$pid_file" ]] && ! is_running; then
          printf 'The VM exited before SSH became ready. Console log:\n' >&2
          tail -n 80 "$log_file" >&2 || true
          return 1
        fi
        sleep 2
      done

      printf 'Timed out waiting for the VM. Console log: %s\n' "$log_file" >&2
      return 1
    }

    up_vm() {
      if is_running; then
        wait_vm
        return
      fi

      init_vm
      qemu_arguments
      qemu-system-aarch64 \
        "''${qemu_args[@]}" \
        -display none \
        -serial "file:$log_file" \
        -monitor "unix:$monitor,server=on,wait=off" \
        -daemonize \
        -pidfile "$pid_file"
      wait_vm
    }

    run_elf() {
      local target executable remote runner status remote_command
      [[ "''${1:-}" == --target && $# -eq 3 ]] || {
        printf 'Usage: yuezhuo-vm run --target <x86_64|aarch64|riscv64> <executable>\n' >&2
        return 2
      }
      target=$2
      executable=$3
      [[ -f "$executable" ]] || {
        printf 'Executable not found: %s\n' "$executable" >&2
        return 2
      }

      if ! is_running; then
        up_vm
      fi

      remote="/tmp/yuezhuo-$target-$$"
      case "$target" in
        aarch64|arm64)
          runner=""
          ;;
        x86_64|x86-64|amd64)
          runner=${linuxQemuUser}/bin/qemu-x86_64
          ;;
        riscv64|riscv64gc)
          runner=${linuxQemuUser}/bin/qemu-riscv64
          ;;
        *)
          printf 'Unknown target: %s\n' "$target" >&2
          return 2
          ;;
      esac

      scp "''${scp_options[@]}" "$executable" "$guest:$remote" </dev/null
      remote_command="chmod 700 $remote"
      # The generated path is intentionally expanded by the trusted local client.
      # shellcheck disable=SC2029
      ssh "''${ssh_options[@]}" "$guest" "$remote_command" </dev/null

      set +e
      remote_command="$runner $remote"
      # shellcheck disable=SC2029
      ssh "''${ssh_options[@]}" "$guest" "$remote_command"
      status=$?
      set -e

      remote_command="rm -f $remote"
      # shellcheck disable=SC2029
      ssh "''${ssh_options[@]}" "$guest" "$remote_command" </dev/null || true
      return "$status"
    }

    command="''${1:-}"
    case "$command" in
      init)
        init_vm
        printf 'VM state initialized in %s\n' "$state_dir"
        ;;
      up)
        up_vm
        ;;
      start)
        init_vm
        if is_running; then
          printf 'The background VM is already running.\n' >&2
          exit 1
        fi
        qemu_arguments
        exec qemu-system-aarch64 "''${qemu_args[@]}" -nographic
        ;;
      down)
        if is_running; then
          pid=$(<"$pid_file")
          kill "$pid"
          for _ in $(seq 1 30); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
          done
          printf 'Yuezhuo Linux VM stopped.\n'
        else
          printf 'Yuezhuo Linux VM is not running.\n'
        fi
        ;;
      status)
        if is_running; then
          printf 'running (pid=%s, ssh=127.0.0.1:%s)\n' "$(<"$pid_file")" "$ssh_port"
        else
          printf 'stopped\n'
          exit 1
        fi
        ;;
      ssh)
        shift
        if ! is_running; then
          up_vm
        fi
        exec ssh "''${ssh_options[@]}" "$guest" "$@"
        ;;
      run)
        shift
        run_elf "$@"
        ;;
      test-functional)
        target="''${2:-aarch64}"
        test_root="''${3:-$project_root/examples/functional}"
        up_vm
        TARGET="$target" \
          RUNNER="$self run --target $target" \
          WORK_DIR="$project_root/target/test-work/functional-$target" \
          "$project_root/tools/run_functional_tests.sh" "$test_root"
        ;;
      test-perf)
        target="''${2:-riscv64}"
        test_root="''${3:-$project_root/examples/perf-riscv64}"
        up_vm
        TARGET="$target" \
          RUNNER="$self run --target $target" \
          "$project_root/tools/run_perf_tests.sh" "$test_root"
        ;;
      help|-h|--help|"")
        usage
        ;;
      *)
        printf 'Unknown command: %s\n' "$command" >&2
        usage >&2
        exit 2
        ;;
    esac
  '';
}
