"""Simple system resource monitor for CPU, RAM, GPU, and VRAM usage."""

import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass
class ResourceStats:
    """Current resource usage statistics."""
    cpu_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    gpu_percent: float = 0.0
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    vram_percent: float = 0.0
    gpu_temp_c: float = 0.0
    gpu_name: str = ""

    def __str__(self) -> str:
        lines = [
            f"CPU: {self.cpu_percent:5.1f}%",
            f"RAM: {self.ram_used_gb:.1f}/{self.ram_total_gb:.1f}GB ({self.ram_percent:.0f}%)",
        ]
        if self.gpu_name:
            lines.extend([
                f"GPU: {self.gpu_percent:5.1f}% ({self.gpu_name})",
                f"VRAM: {self.vram_used_gb:.1f}/{self.vram_total_gb:.1f}GB ({self.vram_percent:.0f}%)",
                f"Temp: {self.gpu_temp_c:.0f}°C",
            ])
        return " | ".join(lines)

    def short_str(self) -> str:
        """Compact single-line format."""
        if self.gpu_name:
            return (f"CPU:{self.cpu_percent:4.0f}% "
                    f"RAM:{self.ram_percent:3.0f}% "
                    f"GPU:{self.gpu_percent:3.0f}% "
                    f"VRAM:{self.vram_percent:3.0f}% "
                    f"{self.gpu_temp_c:.0f}°C")
        return f"CPU:{self.cpu_percent:4.0f}% RAM:{self.ram_percent:3.0f}%"


class ResourceMonitor:
    """Monitor system resources (CPU, RAM, GPU, VRAM).

    Uses psutil for CPU/RAM and nvidia-smi for GPU stats.
    Falls back gracefully if dependencies are missing.

    Example:
        >>> monitor = ResourceMonitor()
        >>> stats = monitor.get_stats()
        >>> print(stats)
        CPU: 25.0% | RAM: 8.2/32.0GB (26%) | GPU: 45.0% | VRAM: 4.1/8.0GB (51%)

        # Or run continuous monitoring in background
        >>> monitor.start_background(interval=2.0)
        >>> # ... do work ...
        >>> monitor.stop_background()
    """

    def __init__(self):
        self._psutil = None
        self._has_nvidia = False
        self._gpu_name = ""
        self._background_thread = None
        self._running = False
        self._last_stats = ResourceStats()
        self._setup()

    def _setup(self):
        """Initialize monitoring capabilities."""
        # Try to import psutil
        try:
            import psutil
            self._psutil = psutil
        except ImportError:
            print("Note: Install 'psutil' for CPU/RAM monitoring")

        # Check for nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self._has_nvidia = True
                self._gpu_name = result.stdout.strip().split('\n')[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def get_stats(self) -> ResourceStats:
        """Get current resource usage statistics."""
        stats = ResourceStats()

        # CPU and RAM via psutil
        if self._psutil:
            try:
                stats.cpu_percent = self._psutil.cpu_percent(interval=0.1)
                mem = self._psutil.virtual_memory()
                stats.ram_total_gb = mem.total / (1024 ** 3)
                stats.ram_used_gb = mem.used / (1024 ** 3)
                stats.ram_percent = mem.percent
            except Exception:
                pass

        # GPU stats via nvidia-smi
        if self._has_nvidia:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(',')
                    if len(parts) >= 4:
                        stats.gpu_percent = float(parts[0].strip())
                        stats.vram_used_gb = float(parts[1].strip()) / 1024
                        stats.vram_total_gb = float(parts[2].strip()) / 1024
                        stats.vram_percent = (stats.vram_used_gb / stats.vram_total_gb) * 100
                        stats.gpu_temp_c = float(parts[3].strip())
                        stats.gpu_name = self._gpu_name
            except (subprocess.TimeoutExpired, ValueError):
                pass

        self._last_stats = stats
        return stats

    @property
    def last_stats(self) -> ResourceStats:
        """Get the most recently collected stats."""
        return self._last_stats

    def print_stats(self):
        """Print current stats to console."""
        stats = self.get_stats()
        print(f"\r{stats.short_str()}", end="", flush=True)

    def start_background(self, interval: float = 2.0, callback=None):
        """Start background monitoring thread.

        Args:
            interval: Seconds between updates
            callback: Optional function to call with each ResourceStats
        """
        if self._running:
            return

        self._running = True

        def monitor_loop():
            while self._running:
                stats = self.get_stats()
                if callback:
                    callback(stats)
                time.sleep(interval)

        self._background_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._background_thread.start()

    def stop_background(self):
        """Stop background monitoring."""
        self._running = False
        if self._background_thread:
            self._background_thread.join(timeout=1.0)
            self._background_thread = None


# Convenience function for quick stats
def get_system_stats() -> ResourceStats:
    """Get current system resource stats (one-shot)."""
    monitor = ResourceMonitor()
    return monitor.get_stats()


def print_system_stats():
    """Print current system stats to console."""
    stats = get_system_stats()
    print(stats)


if __name__ == "__main__":
    # Demo: continuous monitoring
    print("System Resource Monitor")
    print("=" * 50)

    monitor = ResourceMonitor()

    # Print initial stats
    stats = monitor.get_stats()
    print(f"GPU: {stats.gpu_name or 'None detected'}")
    print()

    # Continuous update
    print("Monitoring (Ctrl+C to stop)...")
    try:
        while True:
            monitor.print_stats()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
