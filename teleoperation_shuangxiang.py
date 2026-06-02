import time
import math
import threading
from types import MethodType

import numpy as np

from zlgcan import ZCAN, ZCAN_USBCANFD_MINI
from GUIyemian import ArmController, MODE_MIT, MODE_PV

try:
    from GUIyemian import GRAVITY_COMP_PERIOD_S, GRAVITY_TORQUE_SCALE
except Exception:
    GRAVITY_COMP_PERIOD_S = 0.001
    GRAVITY_TORQUE_SCALE = [0.0, 1.15, 1.1, 1.0, 1.0, 0.0]


# =========================
# 1. CANFD 序列号配置
# =========================

MASTER_SERIAL = "9820ECA9B0E80D6418B0"
SLAVE_SERIAL = "EBD6C68A50FF0DD4F0B0"

DEVICE_SCAN_MAX = 3
CHANNEL_INDEX = 0


# =========================
# 2. 单向遥操作参数
# =========================

CONTROL_HZ = 500.0
PV_VEL_LIM = 0.8

# 每个控制周期允许从端目标 DH 角变化的最大值，单位 rad/cycle
MAX_DELTA_PER_CYCLE = 0.02

# 从端目标角低通滤波系数
ALPHA = 0.20

# 如果某个关节方向相反，把对应项改成 -1
SCALE = np.array([1, 1, 1, 1, 1, 1], dtype=float)

# 电机目标夹紧时离边界留一点余量，单位 rad
MOTOR_LIMIT_MARGIN = 0.003

# 每隔多少次循环打印一次夹紧信息，避免刷屏
CLIP_PRINT_INTERVAL = 100

# 是否打印安全夹紧信息
PRINT_CLIP_INFO = True

# 夹紧量超过该阈值时才打印，单位 rad；小于该值的轻微夹紧不刷屏
CLIP_PRINT_MIN_DELTA = 0.03


# =========================
# 3. 弱双向反馈参数
# =========================
# 弱双向含义：
# 主端 -> 从端：主端关节角驱动从端 PV 跟随
# 从端 -> 主端：从端跟踪误差和电机反馈力矩，转换成主端 MIT 反馈力矩

ENABLE_WEAK_BILATERAL = True

# 是否启用从端“位置跟踪误差”反馈
ENABLE_ERROR_FEEDBACK = True

# 是否启用从端“电机反馈力矩”反馈
ENABLE_TORQUE_FEEDBACK = True

# 启动遥操作前，自动采集一段从端空载/静止时的电机力矩作为零偏
AUTO_ZERO_SLAVE_TORQUE = True
SLAVE_TORQUE_ZERO_TIME_S = 0.5
SLAVE_TORQUE_ZERO_SAMPLE_HZ = 100.0

# 误差反馈：q_error = q_target - q_slave
# 若从端因为碰到物体、限位或阻力而跟不上目标，q_error 会变大；
# 主端反馈力矩默认取 -K * q_error，用来阻碍操作者继续往该方向推。
ERROR_FB_GAIN = np.array([0.80, 0.80, 0.60, 0.18, 0.15, 0.10], dtype=float)

# 误差死区，单位 rad。小误差不反馈，避免从端正常滞后造成主端抖动。
ERROR_FB_DEADZONE = np.array([0.015, 0.015, 0.015, 0.020, 0.020, 0.020], dtype=float)

# 力矩反馈：使用从端电机反馈力矩减去启动时零偏后的残差。
# 注意：电机反馈力矩不等价于真实末端接触力，只能作为弱反馈/阻力感来源。
TORQUE_FB_GAIN = np.array([0.08, 0.08, 0.07, 0.035, 0.030, 0.020], dtype=float)

# 从端电机力矩死区，单位取决于电机反馈 Torque 的单位，一般可先按 N·m 理解。
TORQUE_FB_DEADZONE = np.array([0.15, 0.15, 0.12, 0.08, 0.06, 0.05], dtype=float)

# 电机力矩反馈方向。
# 如果你发现“从端受阻时主端反而被助推”，把这个值改为 +1.0。
# 默认 -1.0 表示产生反向阻力。
TORQUE_FB_SIGN = -1.0

# 主端反馈力矩最大值，单位与 MIT torque_set 一致。
# 第一次实验一定要保守，确认方向正确后再逐步增大。
MASTER_FB_TAU_MAX = np.array([0.80, 0.80, 0.60, 0.30, 0.22, 0.15], dtype=float)

# 主端反馈力矩低通滤波，越小越平滑，越大越跟手。
FEEDBACK_ALPHA = 0.12

# 反馈力矩总开关掩码；如果某个关节不想反馈，设为 0。
FEEDBACK_MASK = np.array([1, 1, 1, 1, 1, 1], dtype=float)

# 每隔多少次循环打印一次反馈信息。
FEEDBACK_PRINT_INTERVAL = 300


# =========================
# 4. CANFD 设备扫描函数
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
# 5. 角度处理函数
# =========================

def wrap_to_pi(q):
    q = np.asarray(q, dtype=float)
    return (q + math.pi) % (2 * math.pi) - math.pi


def angle_diff(q_target, q_now):
    """
    返回 q_target - q_now 的最短角度差，范围 [-pi, pi]。
    注意：这个函数只用于计算差值，不用于强行包裹绝对目标角。
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


def deadzone_vector(x, dz):
    """
    对向量做死区处理：
    |x| <= dz 时输出 0；
    |x| > dz 时输出 sign(x) * (|x|-dz)。
    """
    x = np.asarray(x, dtype=float)
    dz = np.asarray(dz, dtype=float)
    return np.sign(x) * np.maximum(np.abs(x) - dz, 0.0)


def get_dh(controller):
    snapshot = controller.get_status_snapshot()

    if snapshot is None:
        return None

    return np.array(snapshot["dh_rad"], dtype=float)


def get_motor_torque(controller):
    snapshot = controller.get_status_snapshot()

    if snapshot is None:
        return None

    motors = snapshot.get("motors", [])
    if len(motors) < 6:
        return None

    return np.array([float(m["tau"]) for m in motors[:6]], dtype=float)


# =========================
# 6. 从端安全目标写入
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

        large_clip_infos = [
            item for item in clip_infos
            if abs(item["clipped_motor"] - item["raw_motor"]) >= CLIP_PRINT_MIN_DELTA
        ]

        if PRINT_CLIP_INFO and large_clip_infos and loop_count % CLIP_PRINT_INTERVAL == 0:
            print("[INFO] 从端目标接近或超过电机限位，已进行安全夹紧：")
            for item in large_clip_infos:
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
# 7. 主端弱双向反馈力矩叠加
# =========================

def enable_master_external_feedback(controller):
    """
    给主端 ArmController 增加外部反馈力矩通道。

    原来的 gravity_comp_loop 只写：
        tau_master = tau_g

    这里把它替换为：
        tau_master = tau_g + external_feedback_tau

    注意：这个函数必须在 master.initialize_system() 之前调用，
    因为 initialize_system() 内部会启动重力补偿线程。
    """
    controller.external_feedback_tau = np.zeros(6, dtype=float)
    controller.external_feedback_lock = threading.RLock()

    def set_external_feedback_tau(self, tau):
        tau = np.asarray(tau, dtype=float).reshape(6)
        with self.external_feedback_lock:
            self.external_feedback_tau = tau.copy()

    def get_external_feedback_tau(self):
        with self.external_feedback_lock:
            return self.external_feedback_tau.copy()

    def gravity_comp_loop_with_feedback(self):
        assert self.can is not None
        assert self.robot is not None

        self.log("[GRAVITY] 重力补偿线程已启动：tau = tau_g + 弱双向反馈力矩")

        while not self.gravity_stop_event.is_set():
            try:
                with self.data_lock:
                    self.robot.Angle = self.robot.motor2dh(self.can.motors)

                    if not self.robot.set_robot():
                        time.sleep(GRAVITY_COMP_PERIOD_S)
                        continue

                    tau_g_motor = self.robot.Tau_G_Motor

                    with self.external_feedback_lock:
                        tau_fb = self.external_feedback_tau.copy()

                    for i, motor in enumerate(self.can.motors):
                        motor.MIT.position_set = 0.0
                        motor.MIT.velocity_set = 0.0
                        motor.MIT.kp_set = 0.0
                        motor.MIT.kd_set = 0.0
                        motor.MIT.torque_set = float(tau_g_motor[i] * GRAVITY_TORQUE_SCALE[i] + tau_fb[i])
                        motor.set()

                time.sleep(GRAVITY_COMP_PERIOD_S)

            except Exception as e:
                self.log(f"[ERR] 重力补偿/反馈线程异常: {e}")
                time.sleep(0.01)

        self.log("[GRAVITY] 重力补偿/反馈线程退出")

    controller.set_external_feedback_tau = MethodType(set_external_feedback_tau, controller)
    controller.get_external_feedback_tau = MethodType(get_external_feedback_tau, controller)
    controller.gravity_comp_loop = MethodType(gravity_comp_loop_with_feedback, controller)


def measure_slave_torque_zero(slave, sample_time_s=0.5, sample_hz=100.0):
    """
    采集从端静止时的电机反馈力矩均值作为零偏。
    这样后续反馈使用 tau_slave - tau_zero，避免把静态保持力矩一直反馈到主端。
    """
    samples = []
    dt = 1.0 / max(float(sample_hz), 1.0)
    end_time = time.time() + float(sample_time_s)

    print(f"[ZERO] 开始采集从端电机力矩零偏，持续 {sample_time_s:.2f}s...")

    while time.time() < end_time:
        tau = get_motor_torque(slave)
        if tau is not None and tau.shape == (6,):
            samples.append(tau.copy())
        time.sleep(dt)

    if not samples:
        print("[WARN] 从端电机力矩零偏采集失败，使用 0 作为零偏")
        return np.zeros(6, dtype=float)

    tau_zero = np.mean(np.vstack(samples), axis=0)

    print("[ZERO] 从端电机力矩零偏:")
    print(["{:.4f}".format(x) for x in tau_zero])

    return tau_zero


def compute_weak_bilateral_feedback(q_target, q_slave, tau_slave, tau_slave_zero, tau_fb_last):
    """
    根据从端反馈计算主端外部反馈力矩。

    输入：
        q_target: 当前给从端的目标 DH 角
        q_slave: 从端当前 DH 角
        tau_slave: 从端当前电机反馈力矩
        tau_slave_zero: 从端静止零偏力矩
        tau_fb_last: 上一次输出到主端的反馈力矩

    输出：
        tau_fb: 当前输出到主端 MIT 的反馈力矩
        info: 诊断信息
    """
    tau_fb_total = np.zeros(6, dtype=float)

    q_error = angle_diff(q_target, q_slave)
    q_error_eff = deadzone_vector(q_error, ERROR_FB_DEADZONE)

    tau_from_error = np.zeros(6, dtype=float)
    if ENABLE_ERROR_FEEDBACK:
        # q_error 为正：从端实际落后于正方向目标，主端施加反向阻力
        tau_from_error = -ERROR_FB_GAIN * SCALE * q_error_eff
        tau_fb_total += tau_from_error

    tau_residual = np.zeros(6, dtype=float)
    tau_residual_eff = np.zeros(6, dtype=float)
    tau_from_torque = np.zeros(6, dtype=float)

    if ENABLE_TORQUE_FEEDBACK and tau_slave is not None:
        tau_residual = np.asarray(tau_slave, dtype=float).reshape(6) - np.asarray(tau_slave_zero, dtype=float).reshape(6)
        tau_residual_eff = deadzone_vector(tau_residual, TORQUE_FB_DEADZONE)

        # TORQUE_FB_SIGN 默认 -1，如果发现方向反了，改成 +1
        tau_from_torque = TORQUE_FB_SIGN * TORQUE_FB_GAIN * SCALE * tau_residual_eff
        tau_fb_total += tau_from_torque

    tau_fb_total *= FEEDBACK_MASK
    tau_fb_total = np.clip(tau_fb_total, -MASTER_FB_TAU_MAX, MASTER_FB_TAU_MAX)

    # 对反馈力矩做低通滤波，避免主端手感发抖
    tau_fb = np.asarray(tau_fb_last, dtype=float).reshape(6) + FEEDBACK_ALPHA * (
        tau_fb_total - np.asarray(tau_fb_last, dtype=float).reshape(6)
    )

    info = {
        "q_error": q_error,
        "q_error_eff": q_error_eff,
        "tau_slave": tau_slave,
        "tau_residual": tau_residual,
        "tau_residual_eff": tau_residual_eff,
        "tau_from_error": tau_from_error,
        "tau_from_torque": tau_from_torque,
        "tau_fb_raw": tau_fb_total,
        "tau_fb": tau_fb,
    }

    return tau_fb, info


# =========================
# 8. 主程序
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

    # 关键：在主端初始化前替换重力补偿线程，使其支持叠加弱双向反馈力矩
    enable_master_external_feedback(master)

    tau_slave_zero = np.zeros(6, dtype=float)
    tau_fb_last = np.zeros(6, dtype=float)

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
        # 不再用 q_slave_home 作为 PV 目标，避免初始点边界附近反算超限
        if not slave.switch_mode(MODE_PV, None, PV_VEL_LIM):
            print("[ERR] 从端切换 PV 模式失败")
            return

        if AUTO_ZERO_SLAVE_TORQUE:
            tau_slave_zero = measure_slave_torque_zero(
                slave,
                sample_time_s=SLAVE_TORQUE_ZERO_TIME_S,
                sample_hz=SLAVE_TORQUE_ZERO_SAMPLE_HZ,
            )

        q_cmd_last = q_slave_home.copy()

        dt = 1.0 / CONTROL_HZ
        loop_count = 0

        print()
        print("开始弱双向同构遥操作")
        print("主端：MIT 重力补偿 + 从端弱反馈力矩")
        print("从端：PV 模式跟随")
        print(f"弱双向反馈：{'开启' if ENABLE_WEAK_BILATERAL else '关闭'}")
        print(f"误差反馈：{'开启' if ENABLE_ERROR_FEEDBACK else '关闭'}")
        print(f"电机力矩反馈：{'开启' if ENABLE_TORQUE_FEEDBACK else '关闭'}")
        print("按 Ctrl+C 停止")
        print()

        while True:
            loop_start = time.time()
            loop_count += 1

            q_master = get_dh(master)
            q_slave = get_dh(slave)
            tau_slave = get_motor_torque(slave)

            if q_master is None or q_slave is None:
                time.sleep(dt)
                continue

            # 主端相对初始姿态变化量
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

            # =========================
            # 从端 -> 主端：弱双向反馈
            # =========================
            if ENABLE_WEAK_BILATERAL:
                tau_fb, fb_info = compute_weak_bilateral_feedback(
                    q_target=q_target,
                    q_slave=q_slave,
                    tau_slave=tau_slave,
                    tau_slave_zero=tau_slave_zero,
                    tau_fb_last=tau_fb_last,
                )
                tau_fb_last = tau_fb.copy()
                master.set_external_feedback_tau(tau_fb)

                if loop_count % FEEDBACK_PRINT_INTERVAL == 0:
                    print("[FB] 从端反馈 -> 主端 MIT 反馈力矩")
                    print("  q_error rad       =", ["{:.4f}".format(x) for x in fb_info["q_error"]])
                    if tau_slave is not None:
                        print("  tau_slave         =", ["{:.4f}".format(x) for x in tau_slave])
                        print("  tau_slave-zero    =", ["{:.4f}".format(x) for x in fb_info["tau_residual"]])
                    print("  tau_fb_to_master  =", ["{:.4f}".format(x) for x in fb_info["tau_fb"]])
            else:
                master.set_external_feedback_tau(np.zeros(6, dtype=float))

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, dt - elapsed))

    except KeyboardInterrupt:
        print()
        print("收到 Ctrl+C，停止遥操作")

    finally:
        print("清零主端反馈力矩...")
        try:
            master.set_external_feedback_tau(np.zeros(6, dtype=float))
            time.sleep(0.05)
        except Exception as e:
            print(f"[WARN] 清零主端反馈力矩异常: {e}")

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
