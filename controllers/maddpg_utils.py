import numpy as np
from config import Config
from models.node.base import findExecTimeInEachKindOfNode, findDataRate, find_closest_fn
from models.node.fog import FixedFogNode, MobileFogNode
from task_and_user_generator import Config as Cnf


def is_deadline_miss_happening(task, executor, fn_nodes):
    """
    بررسی می‌کند که آیا با اجرای وظیفه روی یک اجراکننده مشخص، مهلت زمانی از دست می‌رود یا خیر.
    خروجی: (آیا خطا رخ می‌دهد, میزان تاخیر یا زمان باقی‌مانده)
    """
    task.real_exec_time_base = findExecTimeInEachKindOfNode(task, executor)
    real_exec_time = task.real_exec_time_base

    if executor == task.creator:
        lateness = task.deadline - (task.release_time + real_exec_time)
        return lateness < 0, lateness

    elif isinstance(executor, (FixedFogNode, MobileFogNode)):
        data_rate = findDataRate(task, executor, 0)
        real_exec_time += task.dataSize / data_rate
        lateness = task.deadline - (task.release_time + real_exec_time)
        return lateness < 0, lateness

    else:  # CloudNode
        closest_fn = find_closest_fn(task.creator.x, task.creator.y, fn_nodes, task.power)
        data_rate = findDataRate(task, executor, closest_fn)
        # فرض ساده‌شده برای محاسبه زمان انتقال به ابر
        real_exec_time += (task.dataSize / data_rate) + (task.dataSize / Config.CloudConfig.CLOUD_BANDWIDTH)
        lateness = task.deadline - (task.release_time + real_exec_time)
        return lateness < 0, lateness


def compute_agent_reward(task, executor, fn_nodes):
    """
    پاداش را بر اساس موفقیت در اجرای وظیفه و رعایت مهلت زمانی محاسبه می‌کند.
    """
    if executor is None or not executor.can_offload_task(task):
        return Config.NEGATIVE_REWARD  # پاداش منفی بزرگ برای انتخاب یک نود نامعتبر

    is_missed, lateness = is_deadline_miss_happening(task, executor, fn_nodes)

    if is_missed:
        # اگر مهلت از دست برود، پاداش منفی است و با میزان تاخیر بدتر می‌شود
        return -5.0 + (lateness / task.deadline)  # نرمال‌سازی پاداش منفی
    else:
        # اگر به موقع انجام شود، پاداش مثبت است و با زودتر تمام شدن بیشتر می‌شود
        return 5.0 * (lateness / task.deadline)  # نرمال‌سازی پاداش مثبت


def get_agent_state(task, simulator):
    """
    وضعیت سیستم را از دید یک عامل برای یک وظیفه مشخص استخراج می‌کند.
    این تابع به ارجاع به شبیه‌ساز برای دسترسی به اطلاعات کلی نیاز دارد.
    """
    creator = task.creator
    maxSpeedOfaVehicle = 13.89

    # نرمال‌سازی مقادیر برای پایداری یادگیری
    vehicle_power_ratio = creator.remaining_power / creator.power
    task_power_ratio = task.power / Cnf.TaskConfig.MAX_POWER_CONSUMPTION
    exec_time_ratio = task.exec_time / Cnf.TaskConfig.MAX_EXEC_TIME
    vehicle_speed_ratio = creator.speed / maxSpeedOfaVehicle

    max_fog_power = 19.79

    # محاسبه میانگین توان در دسترس گره‌های مه در محدوده
    nearby_fogs = [
        node for node in simulator.fixed_fog_nodes.values()
        if np.sqrt((node.x - creator.x) ** 2 + (node.y - creator.y) ** 2) <= 300
    ]
    if not nearby_fogs:
        avg_fog_power_ratio = 0.0
    else:
        avg_power = sum(node.remaining_power for node in nearby_fogs) / len(nearby_fogs)
        avg_fog_power_ratio = avg_power / max_fog_power

    # توان در دسترس ابر
    cloud_power_ratio = simulator.cloud_node.remaining_power / simulator.cloud_node.power

    return np.array([
        vehicle_power_ratio,
        task_power_ratio,
        exec_time_ratio,
        avg_fog_power_ratio,
        vehicle_speed_ratio,
        cloud_power_ratio
    ], dtype=np.float32)