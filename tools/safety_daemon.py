import psutil
import time
import os
import sys
import argparse
import signal
from datetime import datetime

# 尝试导入 pynvml
try:
    import pynvml

    HAS_GPU_MONITOR = True
except ImportError:
    HAS_GPU_MONITOR = False

# === 全局配置 ===
DAEMON_LOG_FILE = None
_PROCS_CACHE = {}


def init_daemon_logging(log_dir):
    """初始化守护进程主日志"""
    global DAEMON_LOG_FILE
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    DAEMON_LOG_FILE = os.path.join(log_dir, "daemon_master.log")
    log_to_master("--- Safety Daemon Started (Waiting for tasks) ---")


def log_to_master(msg):
    """写入守护进程主日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    try:
        with open(DAEMON_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


def get_session_log_path(log_dir, pid):
    """生成单次训练的日志路径"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"session_{timestamp}_pid{pid}.log")


def write_session_log(filepath, msg):
    """写入单次训练的详细日志"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not msg.startswith("["):
        line = f"[{timestamp}] {msg}"
    else:
        line = f"[{datetime.now().strftime('%Y-%m-%d')} {msg[1:]}"
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except:
        pass


# === 资源获取逻辑 (复用之前的优化版本) ===
def get_gpu_stats(target_pids):
    if not HAS_GPU_MONITOR:
        return 0, 0, ""
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        target_pids_set = set(target_pids)
        max_gpu_util = 0
        total_gpu_mem_used = 0
        active_gpus = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                mem_on_this_gpu = 0
                is_my_gpu = False
                for p in compute_procs:
                    if p.pid in target_pids_set:
                        mem_on_this_gpu += p.usedGpuMemory
                        is_my_gpu = True
                if is_my_gpu:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    max_gpu_util = max(max_gpu_util, util)
                    total_gpu_mem_used += mem_on_this_gpu
                    active_gpus.append(f"#{i}:{util}%")
            except pynvml.NVMLError:
                continue
        return max_gpu_util, total_gpu_mem_used / (1024**2), ",".join(active_gpus)
    except Exception:
        return 0, 0, "Err"


def get_process_tree_stats(root_pid):
    global _PROCS_CACHE
    # 1. 根进程检查
    if root_pid not in _PROCS_CACHE:
        try:
            p = psutil.Process(root_pid)
            p.cpu_percent()
            _PROCS_CACHE[root_pid] = p
        except psutil.NoSuchProcess:
            return None

    try:
        root_proc = _PROCS_CACHE[root_pid]
    except KeyError:
        return None

    # 2. 子进程更新
    try:
        current_children = root_proc.children(recursive=True)
    except psutil.NoSuchProcess:
        return None

    current_pids = {root_pid}
    for child in current_children:
        pid = child.pid
        current_pids.add(pid)
        if pid not in _PROCS_CACHE:
            try:
                child.cpu_percent()
                _PROCS_CACHE[pid] = child
            except psutil.NoSuchProcess:
                pass

    # 3. 清理无效缓存
    for pid in list(_PROCS_CACHE.keys()):
        if pid not in current_pids:
            del _PROCS_CACHE[pid]

    # 4. 统计
    total_cpu = 0.0
    total_mem = 0
    valid_procs = []

    for pid, proc in _PROCS_CACHE.items():
        try:
            c = proc.cpu_percent(interval=None)
            m = proc.memory_info().rss
            total_cpu += c
            total_mem += m
            valid_procs.append(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        "cpu": total_cpu,
        "mem_gb": total_mem / (1024**3),
        "count": len(valid_procs),
        "pids": valid_procs,
    }


def kill_process_tree(root_pid, log_file):
    msg = f"!!! KILLING process tree (PID={root_pid}) !!!"
    log_to_master(msg)
    write_session_log(log_file, msg)

    try:
        parent = psutil.Process(root_pid)
        children = parent.children(recursive=True)
        for p in children:
            try:
                p.terminate()
            except:
                pass
        try:
            parent.terminate()
        except:
            pass
        _, alive = psutil.wait_procs(children + [parent], timeout=5)
        if alive:
            for p in alive:
                try:
                    p.kill()
                except:
                    pass
        write_session_log(log_file, "Target Killed Successfully.")
    except psutil.NoSuchProcess:
        write_session_log(log_file, "Process already gone during kill.")


def find_target_pid(keyword, ignore_pids=None):
    if ignore_pids is None:
        ignore_pids = set()
    current_pid = os.getpid()

    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] == current_pid:
                continue
            if proc.info["pid"] in ignore_pids:
                continue

            cmdline = proc.info["cmdline"]
            if cmdline and keyword in " ".join(cmdline):
                return proc.info["pid"]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


def monitor_session(pid, args):
    """
    监控单个训练会话，直到结束
    """
    session_log_file = get_session_log_path(args.log_dir, pid)
    log_to_master(
        f"--> Session Found: PID {pid}. Logging to {os.path.basename(session_log_file)}"
    )

    # 写入Session头信息
    write_session_log(
        session_log_file, f"=== Monitoring Session Started for PID {pid} ==="
    )
    write_session_log(
        session_log_file,
        f"Thresholds: Mem>{args.my_mem_limit}G, CPU>{args.my_cpu_limit}%",
    )
    write_session_log(
        session_log_file,
        f"Time                 | System State                  | Target Process State          | GPU State",
    )
    write_session_log(session_log_file, "-" * 120)

    # 初始化缓存
    global _PROCS_CACHE
    _PROCS_CACHE = {}
    # 初始化 CPU 计数
    try:
        psutil.Process(pid).cpu_percent(interval=None)
    except:
        pass

    violation_streak = 0
    step_count = 0

    while True:
        time.sleep(args.interval)
        step_count += 1

        # 1. 获取目标状态
        my_stats = get_process_tree_stats(pid)
        if my_stats is None:
            write_session_log(
                session_log_file, "Target process finished/vanished. Session ended."
            )
            log_to_master(f"<-- Session Ended: PID {pid} finished naturally.")
            return  # 退出本次会话监控

        # 2. 获取其他资源
        max_gpu_util, total_gpu_mem, gpu_details = get_gpu_stats(my_stats["pids"])
        sys_mem = psutil.virtual_memory()
        sys_cpu = psutil.cpu_percent(interval=None)
        sys_free_gb = sys_mem.available / (1024**3)

        gpu_str = f"GPU: {max_gpu_util}% ({total_gpu_mem/1024:.1f}G)"
        if gpu_details:
            gpu_str += f" [{gpu_details}]"

        status_str = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"SYS: C{sys_cpu:2.0f}% M{sys_free_gb:4.1f}G | "
            f"MY: C{my_stats['cpu']:4.0f}% M{my_stats['mem_gb']:4.1f}G | "
            f"{gpu_str}"
        )

        # 3. 判定危险
        is_dangerous = False
        reasons = []
        if (sys_free_gb < args.sys_min_free_mem) and (
            my_stats["mem_gb"] > args.my_mem_limit
        ):
            is_dangerous = True
            reasons.append("OOM")
        if (sys_cpu > args.sys_max_cpu) and (my_stats["cpu"] > args.my_cpu_limit):
            is_dangerous = True
            reasons.append("CPU-HOG")

        status_suffix = ""
        if is_dangerous:
            violation_streak += 1
            status_suffix = (
                f" | ⚠️ WARN {violation_streak}/{args.patience} [{','.join(reasons)}]"
            )
        else:
            if violation_streak > 0:
                status_suffix = " | ✅ RECOVERED"
                write_session_log(session_log_file, status_str + status_suffix)
            violation_streak = 0

        # 4. 写入日志 (根据频率或危险状态)
        full_log_str = status_str + status_suffix
        if is_dangerous or (step_count % args.log_interval_steps == 0):
            write_session_log(session_log_file, full_log_str)

        # 5. 熔断
        if violation_streak >= args.patience:
            kill_msg = f"!!! EMERGENCY KILL TRIGGERED !!! Reasons: {reasons}"
            write_session_log(session_log_file, kill_msg)
            kill_process_tree(pid, session_log_file)
            log_to_master(f"<-- Session Killed: PID {pid} due to {reasons}")
            return  # 退出本次会话监控


def main():
    parser = argparse.ArgumentParser(description="Silent Safety Daemon")
    parser.add_argument("--log-dir", type=str, default="./monitor_logs")
    parser.add_argument("--keyword", type=str, default="tools/train_net.py")

    # 阈值
    parser.add_argument("--my-cpu-limit", type=float, default=2500.0)
    parser.add_argument("--my-mem-limit", type=float, default=55.0)
    parser.add_argument("--sys-min-free-mem", type=float, default=20.0)
    parser.add_argument("--sys-max-cpu", type=float, default=96.0)

    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument("--log-interval-steps", type=int, default=1)

    args = parser.parse_args()

    # 初始化主日志
    init_daemon_logging(args.log_dir)

    # 已处理过的 PID 集合 (防止重复 attach 同一个没死的进程)
    # 不过现在的逻辑是 monitor_session 是阻塞的，直到进程结束才会回来，
    # 所以其实不需要复杂的去重，只需要循环即可。

    print("Daemon started. Running in background mode.")
    print(f"Check {os.path.join(args.log_dir, 'daemon_master.log')} for status.")

    try:
        while True:
            # 1. 扫描目标
            target_pid = find_target_pid(args.keyword)

            if target_pid:
                # 2. 找到目标，开始监控 (阻塞直到训练结束)
                monitor_session(target_pid, args)
                # 3. 训练结束，休息一下防止 CPU 空转，然后继续扫描下一个
                time.sleep(5)
            else:
                # 没找到，休眠等待
                time.sleep(3)

    except KeyboardInterrupt:
        log_to_master("Daemon stopped by user.")


if __name__ == "__main__":
    # 简单的双重 Fork 实现 Daemon 化 (可选，这里用 nohup 更简单)
    main()
