"""Application exporter"""

import os
import time

from prometheus_client import start_http_server, Gauge
from pynvml import (
    nvmlInit, nvmlUnitGetDeviceCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetName,
    nvmlDeviceGetMinorNumber, nvmlDeviceGetUUID, nvmlDeviceGetMemoryInfo,
    nvmlDeviceGetUtilizationRates, nvmlDeviceGetPowerUsage, nvmlDeviceGetTemperature,
    nvmlDeviceGetFanSpeed,
)

class AppMetrics:
    """
    Representation of Prometheus metrics and loop to fetch and transform
    application metrics into Prometheus metrics.
    """

    def __init__(self, polling_interval_seconds=5):
        self.polling_interval_seconds = polling_interval_seconds
        self.labels = ["minor_number", "uuid", "name"]
        # Prometheus metrics to collect
        self.num_devices = Gauge("num_devices", "Number of GPU devices")
        self.used_memory = Gauge("memory_used_bytes", "Memory used by the GPU device in bytes", self.labels)
        self.total_memory = Gauge("memory_total_bytes", "Total memory of the GPU device in bytes", self.labels)
        self.duty_cycle = Gauge("duty_cycle", "Percent of time over the past sample period during which one or more kernels were executing on the GPU device", self.labels)
        self.power_usage = Gauge("power_usage_milliwatts", "Power usage of the GPU device in milliwatts", self.labels)
        self.temperature = Gauge("temperature_celsius", "Temperature of the GPU device in celsius", self.labels)
        self.fan_speed = Gauge("fanspeed_percent", "Fanspeed of the GPU device as a percent of its maximum", self.labels)

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
        nvmlInit()
        self.num_devices.set(nvmlUnitGetDeviceCount())
        for i in range(self.num_devices):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            minor = nvmlDeviceGetMinorNumber(handle)
            uuid = nvmlDeviceGetUUID(handle)
            labels = [minor, uuid, name]
            memory = nvmlDeviceGetMemoryInfo(handle)
            self.used_memory.labels(*labels).set(memory[1])
            self.total_memory.labels(*labels).set(memory[0])
            self.duty_cycle.labels(*labels).set(nvmlDeviceGetUtilizationRates(handle)[0])
            self.power_usage.labels(*labels).set(nvmlDeviceGetPowerUsage(handle))
            self.temperature.labels(*labels).set(nvmlDeviceGetTemperature(handle))
            self.fan_speed.labels(*labels).set(nvmlDeviceGetFanSpeed(handle))

def main():
    """Main entry point"""

    polling_interval_seconds = int(os.getenv("POLLING_INTERVAL_SECONDS", "5"))
    exporter_port = int(os.getenv("EXPORTER_PORT", "9877"))

    app_metrics = AppMetrics(
        polling_interval_seconds=polling_interval_seconds
    )
    start_http_server(exporter_port)
    app_metrics.run_metrics_loop()