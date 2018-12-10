
import datetime

def mkdir(path):
    import os

    path = path.strip()

    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def str2red(arg):
    return "\033[0;31m" + arg + "\033[0m"


def timeManager(begin_time, last_time, id, remainings):
    now = datetime.datetime.now()
    time_run_delta = now - last_time
    print('finish: %d, 总运行时间：%s， 时间间隔：%s，预计完成时间：%s' %
          (id, str(now - begin_time), str(time_run_delta), str(time_run_delta * remainings)))
    return now