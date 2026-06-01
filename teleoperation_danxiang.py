import time
import math
import numpy as np

from zlgcan import ZCAN, ZCAN_USBCANFD_MINI
from GUIyemian import ArmController, MODE_MIT, MODE_PV

# =========================
# 1. CANFD 序列号配置
# =========================

MASTER_SERIAL = "9820ECA9B0E80D6418B0"
SLAVE_SERIAL = "EBD6C68A50FF0DD4F0B0"

DEVICE_SCAN_MAX = 3
CHANNEL_INDEX = 0

# =========================
# 2. 遥操作参数
# =========================

CONTROL_HZ = 100.0
PV_VEL_LIM = 0.5

# 第一次实验建议先小一点，例如 0.005~0.02
MAX_DELTA_PER_CYCLE = 0.02

# 低通滤波系数
ALPHA = 0.20

# 如果某个关节方向相反，把对应项改成 -1
SCALE = np.array([1, 1, 1, 1, 1, 1], dtype=float)

# 电机目标夹紧时离边界留一点余量，单位 rad
MOTOR_LIMIT_MARGIN = 0.003

# 每隔多少次循环打印一次夹紧信息，避免刷屏
CLIP_PRINT_INTERVAL = 100

# =========================
# 3. CANFD 设备扫描函数
# =========================

def normalize_serial(serial):
    if serial is None:
        return ""
    return str(serial).strip().upper()

def scan_canfd_devices(max_index=8):
    zcan = ZCAN()
    devices = {}

    print("=" * 70)
    print("开始扫描 CANFD 设备...")
    print("=" * 70)

    for idx in range(max_index):
        handle = zcan.OpenDevice(ZCAN_USBCANFD_MINI, idx, 0)

        if int(handle) == 0:
            print(f"device_index={idx}: 打开失败")
            continue

        info = None

        try:
            info = zcan.GetDeviceInf(handle)
        except Exception as e:
            print(f"device_index={idx}: 读取设备信息异常: {e}")
        finally:
            try:
                zcan.CloseDevice(handle)
            except Exception:
                pass

        if info is None:
            print(f"device_index={idx}: 打开成功，但读取设备信息失败")
            continue

        serial = normalize_serial(info.serial)

        print(f"device_index={idx}: 打开成功")
        print(f"  serial  = {serial}")
        print(f"  hw_type = {info.hw_type}")
        print(f"  can_num = {info.can_num}")

        if serial:
            devices[serial] = {
                "device_index": idx,
                "hw_type": info.hw_type,
                "can_num": info.can_num,
            }

    print("=" * 70)
    print(f"扫描完成，共发现 {len(devices)} 个可用 CANFD 设备")

    for serial, item in devices.items():
        print(f"serial={serial}, device_index={item['device_index']}, hw_type={item['hw_type']}")

    print("=" * 70)

    return devices

def get_device_index_by_serial(devices, target_serial, role_name):
    target_serial = normalize_serial(target_serial)

    if target_serial not in devices:
        print(f"[ERR] 没有找到 {role_name} CANFD 设备")
        print(f"[ERR] 目标 serial = {target_serial}")
        print("[ERR] 当前扫描到的 serial 有：")

        for serial in devices.keys():
            print(f"  {serial}")

        raise RuntimeError(f"未找到 {role_name} CANFD 设备: serial={target_serial}")

    device_index = devices[target_serial]["device_index"]

    print(f"[OK] {role_name} CANFD 匹配成功")
    print(f"     serial       = {target_serial}")
    print(f"     device_index = {device_index}")

    return device_index

# =========================
# 4. 角度处理函数
# =========================

def wrap_to_pi(q):
    q = np.asarray(q, dtype=float)
    return (q + math.pi) % (2 * math.pi) - math.pi

def angle_diff(q_target, q_now):
    """
    返回 q_target - q_now 的最短角度差，范围 [-pi, pi]。
    注意：这个函数只用于计算差值，不再用于强行包裹绝对目标角。
    """
    return wrap_to_pi(np.asarray(q_target, dtype=float) - np.asarray(q_now, dtype=float))

def limit_delta_continuous(q_target, q_last, max_delta):
    q_target = np.asarray(q_target, dtype=float)
    q_last = np.asarray(q_last, dtype=float)

    delta = angle_diff(q_target, q_last)
    delta = np.clip(delta, -max_delta, max_delta)

    return q_last + delta

def lowpass_continuous(q_target, q_last, alpha):
    q_target = np.asarray(q_target, dtype=float)
    q_last = np.asarray(q_last, dtype=float)

    delta = angle_diff(q_target, q_last)
    return q_last + alpha * delta

def get_dh(controller):
    snapshot = controller.get_status_snapshot()

    if snapshot is None:
        return None

    return np.array(snapshot["dh_rad"], dtype=float)

# =========================
# 5. 从端安全目标写入
# =========================

def dh_to_motor_near_current_clamped(controller, target_dh, margin=0.003):
    """
    将目标 DH 角转换为电机角。

    与 robot.dh2motor() 的区别：
    1. 会优先选择最接近当前电机位置的 2pi 等效电机角；
    2. 如果目标略微超出电机限位，会夹紧到限位内部；
    3. 适合遥操作连续跟随，避免实际反馈在边界附近造成反复超限。
    """
    if controller.can is None or controller.robot is None:
        raise RuntimeError("controller 尚未初始化")

    target_dh = np.asarray(target_dh, dtype=float).reshape(6)

    ratio = np.asarray(controller.robot.ratio, dtype=float)
    zero_offset = np.asarray(controller.robot.zero_offset, dtype=float)

    motor_target = np.zeros(6, dtype=float)
    clip_infos = []

    for i, motor in enumerate(controller.can.motors):
        lo = float(motor.angle_lim[0])
        hi = float(motor.angle_lim[1])

        # 原始 DH -> motor
        raw = (target_dh[i] - zero_offset[i]) * ratio[i]

        # 选择与当前电机反馈最接近的等效角
        current_motor = float(motor.Position)
        candidates = np.array([raw, raw + 2.0 * math.pi, raw - 2.0 * math.pi], dtype=float)

        valid_candidates = [c for c in candidates if lo + margin <= c <= hi - margin]

        if valid_candidates:
            chosen = min(valid_candidates, key=lambda x: abs(x - current_motor))
            clipped = chosen
            was_clipped = False
        else:
            chosen = min(candidates, key=lambda x: abs(x - current_motor))
            clipped = float(np.clip(chosen, lo + margin, hi - margin))
            was_clipped = abs(clipped - chosen) > 1e-9

        motor_target[i] = clipped

        if was_clipped:
            clip_infos.append({
                "joint": i + 1,
                "dh": float(target_dh[i]),
                "raw_motor": float(chosen),
                "clipped_motor": float(clipped),
                "limit": [lo, hi],
            })

    return motor_target, clip_infos

def safe_set_slave_target_dh(slave, target_dh, velocity_lim, loop_count=0):
    """
    从端安全写入目标：
    DH 角 -> 电机角，必要时夹紧，然后写入 PV 目标。
    """
    try:
        motor_target, clip_infos = dh_to_motor_near_current_clamped(slave, target_dh, MOTOR_LIMIT_MARGIN)

        if clip_infos and loop_count % CLIP_PRINT_INTERVAL == 0:
            print("[INFO] 从端目标接近或超过电机限位，已进行安全夹紧：")
            for item in clip_infos:
                print(
                    f"  关节{item['joint']}: "
                    f"raw_motor={item['raw_motor']:.4f}, "
                    f"clipped={item['clipped_motor']:.4f}, "
                    f"limit=[{item['limit'][0]:.4f}, {item['limit'][1]:.4f}]"
                )

        ok = slave.set_pv_target_motor_position(motor_target.tolist(), velocity_lim)

        return ok

    except Exception as e:
        print(f"[WARN] safe_set_slave_target_dh 异常: {e}")
        return False

# =========================
# 6. 主程序
# =========================

def main():
    devices = scan_canfd_devices(DEVICE_SCAN_MAX)

    master_index = get_device_index_by_serial(devices, MASTER_SERIAL, "主端")
    slave_index = get_device_index_by_serial(devices, SLAVE_SERIAL, "从端")

    if master_index == slave_index:
        raise RuntimeError("主端和从端匹配到了同一个 device_index，请检查 serial 配置")

    print()
    print("最终设备匹配结果：")
    print(f"  主端 CANFD: serial={MASTER_SERIAL}, device_index={master_index}")
    print(f"  从端 CANFD: serial={SLAVE_SERIAL}, device_index={slave_index}")
    print()

    master = ArmController(device_index=master_index, channel_index=CHANNEL_INDEX, name="master")
    slave = ArmController(device_index=slave_index, channel_index=CHANNEL_INDEX, name="slave")

    try:
        print("初始化主端机械臂...")
        if not master.initialize_system():
            print("[ERR] 主端初始化失败")
            return

        print("初始化从端机械臂...")
        if not slave.initialize_system():
            print("[ERR] 从端初始化失败")
            return

        time.sleep(1.0)

        q_master_home = get_dh(master)
        q_slave_home = get_dh(slave)

        if q_master_home is None or q_slave_home is None:
            print("[ERR] 无法读取初始关节角")
            return

        print()
        print("主端初始 DH 角 rad:")
        print(["{:.4f}".format(x) for x in q_master_home])

        print("从端初始 DH 角 rad:")
        print(["{:.4f}".format(x) for x in q_slave_home])
        print()

        print("主端切换到 MIT 模式...")
        if not master.switch_mode(MODE_MIT, None, PV_VEL_LIM):
            print("[ERR] 主端切换 MIT 模式失败")
            return

        print("从端切换到 PV 模式...")
        # 关键修改：
        # 不再用 q_slave_home 作为 PV 目标，因为 q_slave_home 反算成电机角时可能略微越过限位。
        # 这里让从端直接保持当前位置。
        if not slave.switch_mode(MODE_PV, None, PV_VEL_LIM):
            print("[ERR] 从端切换 PV 模式失败")
            return

        q_cmd_last = q_slave_home.copy()

        dt = 1.0 / CONTROL_HZ
        loop_count = 0

        print()
        print("开始单机同构遥操作")
        print("主端：MIT 重力补偿，可拖动")
        print("从端：PV 模式跟随")
        print("按 Ctrl+C 停止")
        print()

        while True:
            loop_start = time.time()
            loop_count += 1

            q_master = get_dh(master)

            if q_master is None:
                time.sleep(dt)
                continue

            # 只对“主端相对变化量”做 wrap，不再对从端绝对目标角整体 wrap
            delta_master = angle_diff(q_master, q_master_home)

            # 从端目标 DH 角，保持连续形式
            q_target_raw = q_slave_home + SCALE * delta_master

            # 单周期限幅，防止目标突变
            q_target = limit_delta_continuous(q_target_raw, q_cmd_last, MAX_DELTA_PER_CYCLE)

            # 低通滤波，使从端运动更平滑
            q_target = lowpass_continuous(q_target, q_cmd_last, ALPHA)

            # 安全写入从端 PV 目标
            ok = safe_set_slave_target_dh(slave, q_target, PV_VEL_LIM, loop_count)

            if ok:
                q_cmd_last = q_target.copy()
            else:
                print("[WARN] 从端目标写入失败，保持当前位置")
                slave.set_pv_hold_current_position(PV_VEL_LIM)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print()
        print("收到 Ctrl+C，停止遥操作")

    finally:
        print("清理从端机械臂...")
        try:
            slave.cleanup()
        except Exception as e:
            print(f"[WARN] 从端清理异常: {e}")

        print("清理主端机械臂...")
        try:
            master.cleanup()
        except Exception as e:
            print(f"[WARN] 主端清理异常: {e}")

        print("程序结束")

if __name__ == "__main__":
    main()