
import psutil
import pynvml
import os, sys
import time

pynvml.nvmlInit()
# 这里的0是GPU id
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
memory_max = 0
temprature_max = 0
GPU_load_max = 0
while(1):
    info = psutil.virtual_memory()
    memory_max = max(memory_max, info.percent)
    print(u'内存占比：%4.1f/%4.1f' % (info.percent, memory_max), end="    ")

    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print("GPU memory: %dMB/%dMB" % (meminfo.used / pow(1024, 2), meminfo.total / pow(1024, 2)), end="    ")

    temprature = pynvml.nvmlDeviceGetTemperature(handle, 0)
    temprature_max = max(temprature_max, temprature)
    print("temprature: %d℃/%d℃" % (temprature, temprature_max), end="    ")

    PowerState = pynvml.nvmlDeviceGetPowerState(handle)
    print("PowerState: %d" % (PowerState), end="    ")

    PowerUsage = pynvml.nvmlDeviceGetPowerUsage(handle)
    PowerManagementDefaultLimit = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
    print("PowerUsage: %d/%d" % (PowerUsage, PowerManagementDefaultLimit), end="    ")

    GPU_load = PowerUsage / PowerManagementDefaultLimit * 100
    GPU_load_max = max(GPU_load_max, GPU_load)
    print("Power load: %4.1f/%4.1f%%" % (GPU_load, GPU_load_max))

    time.sleep(1)

if 0:
    print(u'cpu个数：',psutil.cpu_count())
    print(u'内存使用：', psutil.Process(os.getpid()).memory_info().rss)
    print(u'总内存：', info.total)




