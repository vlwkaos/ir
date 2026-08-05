#!/usr/bin/env bash
# ^ Shared local state setup for benchmark scripts.

bench_env_init() {
    local repo_root="$1"
    local scope="$2"
    local source_xdg="${XDG_CONFIG_HOME:-$HOME/.config}"
    local state_root="$repo_root/.bench-state/$scope"
    export XDG_CONFIG_HOME="$state_root/xdg"
    export IR_CONFIG_DIR="$XDG_CONFIG_HOME/ir"
    export TMPDIR="$state_root/tmp"
    export IR_BENCH_STATE_DIR="$state_root"
    mkdir -p "$IR_CONFIG_DIR" "$TMPDIR"

    local source_config="$source_xdg/ir/config.yml"
    local local_config="$IR_CONFIG_DIR/config.yml"
    if [[ -f "$source_config" && ! -f "$local_config" ]]; then
        cp "$source_config" "$local_config"
    fi

    # Replace a real directory (stale copied state) with a symlink so $IR_DIR/preprocessors/...
    # resolves when IR_CONFIG_DIR points to the isolated state dir instead of ~/.config/ir.
    local source_preprocessors="$source_xdg/ir/preprocessors"
    local local_preprocessors="$IR_CONFIG_DIR/preprocessors"
    if [[ -d "$source_preprocessors" ]] && \
       ! [[ -L "$local_preprocessors" && "$(readlink "$local_preprocessors")" == "$source_preprocessors" ]]; then
        rm -rf "$local_preprocessors"
        ln -s "$source_preprocessors" "$local_preprocessors"
    fi
}

bench_guard_enabled() {
    [[ "${IR_BENCH_GUARD:-1}" != "0" ]] || return 1
    [[ "$(uname -s)" == "Darwin" ]] || return 1
    command -v memory_pressure >/dev/null 2>&1 || return 1
    command -v vm_stat >/dev/null 2>&1 || return 1
    command -v pgrep >/dev/null 2>&1 || return 1
    command -v ps >/dev/null 2>&1 || return 1
}

bench_memory_free_pct() {
    memory_pressure -Q 2>/dev/null | awk '/System-wide memory free percentage:/ {gsub(/%/, "", $5); print $5}'
}

bench_swapouts() {
    vm_stat 2>/dev/null | awk '/Swapouts:/ {gsub(/\./, "", $2); print $2}'
}

bench_ir_cpu_pct() {
    ps -axo pcpu=,comm= 2>/dev/null | awk '$2 ~ /(^|\/)ir$/ {sum += $1} END {printf "%.0f\n", sum + 0}'
}

bench_kill_tree_term() {
    local pid="$1"
    local child=""
    while IFS= read -r child; do
        [[ -n "$child" ]] || continue
        bench_kill_tree_term "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill -TERM "$pid" 2>/dev/null || true
}

bench_kill_tree_kill() {
    local pid="$1"
    local child=""
    while IFS= read -r child; do
        [[ -n "$child" ]] || continue
        bench_kill_tree_kill "$child"
    done < <(pgrep -P "$pid" 2>/dev/null || true)
    kill -KILL "$pid" 2>/dev/null || true
}

# Teardown for THIS bench context: reap any in-flight guarded child, then stop
# the isolated daemon. bench_env_init points IR_CONFIG_DIR at the isolated state
# dir, so `daemon stop` only ever targets the bench daemon on its own socket —
# never a real work daemon used by other sessions. Register from the entrypoint:
#   trap bench_cleanup EXIT INT TERM
bench_cleanup() {
    if [[ -n "${BENCH_GUARDED_CHILD:-}" ]]; then
        bench_kill_tree_term "$BENCH_GUARDED_CHILD"
        bench_kill_tree_kill "$BENCH_GUARDED_CHILD"
        BENCH_GUARDED_CHILD=""
    fi
    if [[ -n "${BENCH_DAEMON_BIN:-}" && -x "${BENCH_DAEMON_BIN}" ]]; then
        "${BENCH_DAEMON_BIN}" daemon stop >/dev/null 2>&1 || true
    fi
}

bench_run_guarded() {
    local label="$1"
    local daemon_bin="${2:-}"
    shift 2 || true
    BENCH_DAEMON_BIN="$daemon_bin"   # recorded for bench_cleanup trap teardown

    if ! bench_guard_enabled; then
        "$@"
        return $?
    fi

    local interval_s="${IR_BENCH_GUARD_INTERVAL_S:-5}"
    local min_free_pct="${IR_BENCH_MIN_FREE_PCT:-8}"
    local max_ir_cpu_pct="${IR_BENCH_MAX_IR_CPU_PCT:-800}"
    local cpu_strikes_limit="${IR_BENCH_CPU_STRIKES:-3}"
    # Pages of cumulative system swapout drift tolerated before aborting.
    # Default 65536 (~1 GiB) tolerates benign background paging (~50k pages/15min
    # observed on a busy machine) so large-corpus embeds don't false-abort; the
    # free-pct floor (IR_BENCH_MIN_FREE_PCT) remains the real guard against
    # genuine memory exhaustion. Set to 0 for the strictest pristine-latency runs.
    local max_swapout_delta="${IR_BENCH_MAX_SWAPOUT_DELTA:-65536}"
    local swapouts_start
    swapouts_start="$(bench_swapouts)"
    [[ -n "$swapouts_start" ]] || swapouts_start=0

    local reason_file="$TMPDIR/bench-guard-$$-$RANDOM.reason"
    local log_file="${IR_BENCH_STATE_DIR:-$TMPDIR}/watchdog.log"

    "$@" &
    local child_pid=$!
    BENCH_GUARDED_CHILD="$child_pid"   # recorded so bench_cleanup can reap an orphaned child

    (
        local cpu_strikes=0
        while kill -0 "$child_pid" 2>/dev/null; do
            local free_pct
            local swapouts_now
            local ir_cpu_pct
            local reason=""

            free_pct="$(bench_memory_free_pct)"
            swapouts_now="$(bench_swapouts)"
            ir_cpu_pct="$(bench_ir_cpu_pct)"

            if [[ -n "$free_pct" && "$free_pct" =~ ^[0-9]+$ ]] && (( free_pct <= min_free_pct )); then
                reason="memory free ${free_pct}% <= ${min_free_pct}%"
            elif [[ -n "$swapouts_now" && "$swapouts_now" =~ ^[0-9]+$ ]] && (( swapouts_now > swapouts_start + max_swapout_delta )); then
                reason="swapouts increased (${swapouts_start} -> ${swapouts_now}, delta > ${max_swapout_delta})"
            elif [[ -n "$ir_cpu_pct" && "$ir_cpu_pct" =~ ^[0-9]+$ ]] && (( ir_cpu_pct >= max_ir_cpu_pct )); then
                cpu_strikes=$((cpu_strikes + 1))
                if (( cpu_strikes >= cpu_strikes_limit )); then
                    reason="ir CPU ${ir_cpu_pct}% >= ${max_ir_cpu_pct}% for ${cpu_strikes_limit} checks"
                fi
            else
                cpu_strikes=0
            fi

            if [[ -n "$reason" ]]; then
                printf '%s\n' "$reason" > "$reason_file"
                printf '[%s] bench watchdog: aborting %s (%s)\n' "$(date +%H:%M:%S)" "$label" "$reason" | tee -a "$log_file" >&2
                bench_kill_tree_term "$child_pid"
                sleep 2
                bench_kill_tree_kill "$child_pid"
                if [[ -n "$daemon_bin" && -x "$daemon_bin" ]]; then
                    "$daemon_bin" daemon stop >/dev/null 2>&1 || true
                fi
                exit 0
            fi

            sleep "$interval_s"
        done
    ) &
    local watchdog_pid=$!

    local status=0
    wait "$child_pid" || status=$?
    BENCH_GUARDED_CHILD=""            # child finished; nothing to reap
    kill "$watchdog_pid" 2>/dev/null || true
    wait "$watchdog_pid" 2>/dev/null || true

    if [[ -f "$reason_file" ]]; then
        cat "$reason_file" >&2
        rm -f "$reason_file"
        return 125
    fi

    rm -f "$reason_file"
    return "$status"
}
