"""Application exporter"""

import os
import time

import prometheus_client
import pynvml


class AppMetrics:
    """
    Representation of Prometheus metrics and loop to fetch and transform
    application metrics into Prometheus metrics.
    """

    def __init__(self, polling_interval_seconds=5):
        self.polling_interval_seconds = polling_interval_seconds
        self.labels = ["minor_number", "uuid", "name"]
        # Prometheus metrics to collect
        self.num_devices = prometheus_client.Gauge("nvidia_num_devices", "Number of GPU devices")
        self.used_memory = prometheus_client.Gauge(
            "nvidia_memory_used_bytes", "Memory used by the GPU device in bytes", self.labels
        )
        self.total_memory = prometheus_client.Gauge(
            "nvidia_memory_total_bytes", "Total memory of the GPU device in bytes", self.labels
        )
        self.duty_cycle = prometheus_client.Gauge(
            "nvidia_duty_cycle",
            "Percent of time over the past sample period during which one or more kernels were executing on the GPU device",
            self.labels,
        )
        self.power_usage = prometheus_client.Gauge(
            "nvidia_power_usage_milliwatts", "Power usage of the GPU device in milliwatts", self.labels
        )
        self.temperature = prometheus_client.Gauge(
            "nvidia_temperature_celsius", "Temperature of the GPU device in celsius", self.labels
        )
        self.fan_speed = prometheus_client.Gauge(
            "nvidia_fanspeed_percent", "Fanspeed of the GPU device as a percent of its maximum", self.labels
        )

        self.compute_process_memory = prometheus_client.Gauge(
            "nvidia_compute_process_memory",
            "Memory used by compute process",
            self.labels + ["process_id", "process_name"],
        )
        self.graphics_process_memory = prometheus_client.Gauge(
            "nvidia_graphics_process_memory",
            "Memory used by graphics process",
            self.labels + ["process_id", "process_name"],
        )
        self.MPScompute_process_memory = prometheus_client.Gauge(
            "nvidia_MPScompute_process_memory",
            "Memory used by MPS compute process",
            self.labels + ["process_id", "process_name"],
        )

    def run_metrics_loop(self):
        """Metrics fetching loop"""

        while True:
            self.fetch()
            time.sleep(self.polling_interval_seconds)

    def fetch(self):
        """
        Get metrics from application and refresh Prometheus metrics with
        new values.
        """
        pynvml.nvmlInit()
        self.num_devices = pynvml.nvmlDeviceGetCount()
        self.compute_process_memory.clear()
        self.graphics_process_memory.clear()
        self.MPScompute_process_memory.clear()
        for i in range(self.num_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            labels = [
                pynvml.nvmlDeviceGetMinorNumber(handle),
                pynvml.nvmlDeviceGetUUID(handle),
                pynvml.nvmlDeviceGetName(handle),
            ]
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

            self.used_memory.labels(*labels).set(memory.used)
            self.total_memory.labels(*labels).set(memory.total)
            self.duty_cycle.labels(*labels).set(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            self.power_usage.labels(*labels).set(pynvml.nvmlDeviceGetPowerUsage(handle))
            self.temperature.labels(*labels).set(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            self.fan_speed.labels(*labels).set(pynvml.nvmlDeviceGetFanSpeed(handle))

            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                p_labels = [p.pid, pynvml.nvmlSystemGetProcessName(p.pid).decode()]
                self.compute_process_memory.labels(*labels, *p_labels).set(p.usedGpuMemory)

            for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle):
                p_labels = [p.pid, pynvml.nvmlSystemGetProcessName(p.pid).decode()]
                self.graphics_process_memory.labels(*labels, *p_labels).set(p.usedGpuMemory)

            for p in pynvml.nvmlDeviceGetMPSComputeRunningProcesses(handle):
                p_labels = [p.pid, pynvml.nvmlSystemGetProcessName(p.pid).decode()]
                self.MPScompute_process_memory.labels(*labels, *p_labels).set(p.usedGpuMemory)


def main():
    """Main entry point"""

    polling_interval_seconds = int(os.getenv("POLLING_INTERVAL_SECONDS", "5"))
    exporter_port = int(os.getenv("EXPORTER_PORT", "9102"))

    app_metrics = AppMetrics(polling_interval_seconds=polling_interval_seconds)
    prometheus_client.start_http_server(exporter_port)
    app_metrics.run_metrics_loop()
