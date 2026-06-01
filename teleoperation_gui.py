"""
单机双臂同构遥操作 GUI 版本

使用前提：
1. 本文件与 GUIyemian.py、USBCANFD.py、DMMotor.py、Robot.py、TreeStruct.py、zlgcan.py 放在同一目录。
2. 你已经把 USBCANFD.py 修改为支持 device_index/channel_index。
3. 你已经把 GUIyemian.py 中 ArmController 修改为支持：
   ArmController(device_index=..., channel_index=..., name=...)

功能：
- 扫描两个 ZLG USBCANFD 设备，并通过 serial 自动匹配主端/从端；
- 初始化主端机械臂和从端机械臂；
- 主端切换 MIT 模式，保留重力补偿；
- 从端切换 PV 模式并保持当前位置；
- 后台线程执行单机同构遥操作；
- GUI 中设置 CONTROL_HZ、PV_VEL_LIM、MAX_DELTA_PER_CYCLE、ALPHA、SCALE 等参数；
- 实时显示主端 DH、从端 DH、从端目标 DH；
- 停止遥操作后从端保持当前位置；
- 清理时关闭并失能两台机械臂。
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from typing import Any, Dict, Optional

import numpy as np

from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from zlgcan import ZCAN, ZCAN_USBCANFD_MINI
from GUIyemian import ArmController, MODE_MIT, MODE_PV


# =========================
# 1. 默认配置
# =========================

DEFAULT_MASTER_SERIAL = "9820ECA9B0E80D6418B0"
DEFAULT_SLAVE_SERIAL = "EBD6C68A50FF0DD4F0B0"
DEFAULT_DEVICE_SCAN_MAX = 3
DEFAULT_CHANNEL_INDEX = 0

DEFAULT_CONTROL_HZ = 100.0
DEFAULT_PV_VEL_LIM = 0.5
DEFAULT_MAX_DELTA_PER_CYCLE = 0.02
DEFAULT_ALPHA = 0.20
DEFAULT_SCALE = [1, 1, 1, 1, 1, 1]
DEFAULT_MOTOR_LIMIT_MARGIN = 0.003
DEFAULT_CLIP_PRINT_INTERVAL = 100

STATUS_EMIT_PERIOD_S = 0.10


# =========================
# 2. 通用函数
# =========================

def normalize_serial(serial: Any) -> str:
    if serial is None:
        return ""
    return str(serial).strip().upper()


def wrap_to_pi(q: Any) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return (q + math.pi) % (2 * math.pi) - math.pi


def angle_diff(q_target: Any, q_now: Any) -> np.ndarray:
    """返回 q_target - q_now 的最短角度差，范围 [-pi, pi]。"""
    return wrap_to_pi(np.asarray(q_target, dtype=float) - np.asarray(q_now, dtype=float))


def limit_delta_continuous(q_target: Any, q_last: Any, max_delta: float) -> np.ndarray:
    q_target = np.asarray(q_target, dtype=float)
    q_last = np.asarray(q_last, dtype=float)
    delta = angle_diff(q_target, q_last)
    delta = np.clip(delta, -float(max_delta), float(max_delta))
    return q_last + delta


def lowpass_continuous(q_target: Any, q_last: Any, alpha: float) -> np.ndarray:
    q_target = np.asarray(q_target, dtype=float)
    q_last = np.asarray(q_last, dtype=float)
    delta = angle_diff(q_target, q_last)
    return q_last + float(alpha) * delta


def get_dh(controller: ArmController) -> Optional[np.ndarray]:
    snapshot = controller.get_status_snapshot()
    if snapshot is None:
        return None
    dh_rad = snapshot.get("dh_rad", None)
    if dh_rad is None or len(dh_rad) != 6:
        return None
    return np.array(dh_rad, dtype=float)


def create_arm_controller(device_index: int, channel_index: int, name: str) -> ArmController:
    """
    创建 ArmController。

    注意：这里要求你本地的 GUIyemian.py 已经把 ArmController.__init__ 改成支持
    device_index、channel_index、name 参数。
    """
    try:
        return ArmController(device_index=device_index, channel_index=channel_index, name=name)
    except TypeError as e:
        raise TypeError(
            "当前 GUIyemian.py 中的 ArmController 似乎还不支持 "
            "ArmController(device_index=..., channel_index=..., name=...)。\n"
            "请确认你已经完成对 ArmController 和 USBCANFD 的 device_index/channel_index 修改。"
        ) from e


def scan_canfd_devices(max_index: int, log_func=None) -> Dict[str, Dict[str, Any]]:
    """
    扫描当前可打开的 ZLG USBCANFD 设备。

    返回：
        {
            "SERIAL": {
                "device_index": int,
                "hw_type": str,
                "can_num": int,
            }
        }
    """
    def log(msg: str) -> None:
        if log_func is not None:
            log_func(msg)
        else:
            print(msg)

    zcan = ZCAN()
    devices: Dict[str, Dict[str, Any]] = {}

    log("=" * 70)
    log("开始扫描 CANFD 设备...")
    log("=" * 70)

    for idx in range(int(max_index)):
        handle = zcan.OpenDevice(ZCAN_USBCANFD_MINI, idx, 0)

        if int(handle) == 0:
            log(f"device_index={idx}: 打开失败")
            continue

        info = None
        try:
            info = zcan.GetDeviceInf(handle)
        except Exception as e:
            log(f"device_index={idx}: 读取设备信息异常: {e}")
        finally:
            try:
                zcan.CloseDevice(handle)
            except Exception:
                pass

        if info is None:
            log(f"device_index={idx}: 打开成功，但读取设备信息失败")
            continue

        serial = normalize_serial(info.serial)
        hw_type = str(info.hw_type)
        can_num = int(info.can_num)

        log(f"device_index={idx}: 打开成功")
        log(f"  serial  = {serial}")
        log(f"  hw_type = {hw_type}")
        log(f"  can_num = {can_num}")

        if serial:
            devices[serial] = {
                "device_index": idx,
                "hw_type": hw_type,
                "can_num": can_num,
            }

    log("=" * 70)
    log(f"扫描完成，共发现 {len(devices)} 个可用 CANFD 设备")

    for serial, item in devices.items():
        log(f"serial={serial}, device_index={item['device_index']}, hw_type={item['hw_type']}")

    log("=" * 70)
    return devices


def get_device_index_by_serial(devices: Dict[str, Dict[str, Any]], target_serial: str, role_name: str, log_func=None) -> int:
    def log(msg: str) -> None:
        if log_func is not None:
            log_func(msg)
        else:
            print(msg)

    target_serial = normalize_serial(target_serial)

    if target_serial not in devices:
        log(f"[ERR] 没有找到 {role_name} CANFD 设备")
        log(f"[ERR] 目标 serial = {target_serial}")
        log("[ERR] 当前扫描到的 serial 有：")
        for serial in devices.keys():
            log(f"  {serial}")
        raise RuntimeError(f"未找到 {role_name} CANFD 设备: serial={target_serial}")

    device_index = int(devices[target_serial]["device_index"])
    log(f"[OK] {role_name} CANFD 匹配成功")
    log(f"     serial       = {target_serial}")
    log(f"     device_index = {device_index}")
    return device_index


def dh_to_motor_near_current_clamped(controller: ArmController, target_dh: Any, margin: float) -> tuple[np.ndarray, list[dict]]:
    """
    将目标 DH 角转换为电机角。

    与 robot.dh2motor() 的区别：
    1. 优先选择最接近当前电机位置的 2pi 等效电机角；
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

        raw = (target_dh[i] - zero_offset[i]) * ratio[i]
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


def safe_set_slave_target_dh(
    slave: ArmController,
    target_dh: Any,
    velocity_lim: float,
    margin: float,
    loop_count: int,
    clip_print_interval: int,
    log_func=None,
) -> bool:
    def log(msg: str) -> None:
        if log_func is not None:
            log_func(msg)
        else:
            print(msg)

    try:
        motor_target, clip_infos = dh_to_motor_near_current_clamped(slave, target_dh, margin)

        if clip_infos and loop_count % max(1, int(clip_print_interval)) == 0:
            log("[INFO] 从端目标接近或超过电机限位，已进行安全夹紧：")
            for item in clip_infos:
                log(
                    f"  关节{item['joint']}: "
                    f"raw_motor={item['raw_motor']:.4f}, "
                    f"clipped={item['clipped_motor']:.4f}, "
                    f"limit=[{item['limit'][0]:.4f}, {item['limit'][1]:.4f}]"
                )

        return bool(slave.set_pv_target_motor_position(motor_target.tolist(), float(velocity_lim)))

    except Exception as e:
        log(f"[WARN] safe_set_slave_target_dh 异常: {e}")
        return False


# =========================
# 3. Worker 线程
# =========================

class DeviceScanWorker(QThread):
    log_signal = Signal(str)
    done_signal = Signal(object)
    fail_signal = Signal(str)

    def __init__(self, max_index: int):
        super().__init__()
        self.max_index = int(max_index)

    def run(self) -> None:
        try:
            devices = scan_canfd_devices(self.max_index, self.log_signal.emit)
            self.done_signal.emit(devices)
        except Exception:
            self.fail_signal.emit(traceback.format_exc())


class ArmInitWorker(QThread):
    log_signal = Signal(str)
    done_signal = Signal(object, object)
    fail_signal = Signal(str)

    def __init__(self, master: ArmController, slave: ArmController, pv_vel_lim: float):
        super().__init__()
        self.master = master
        self.slave = slave
        self.pv_vel_lim = float(pv_vel_lim)

    def run(self) -> None:
        try:
            self.log_signal.emit("初始化主端机械臂...")
            if not self.master.initialize_system():
                raise RuntimeError("主端初始化失败")

            self.log_signal.emit("初始化从端机械臂...")
            if not self.slave.initialize_system():
                raise RuntimeError("从端初始化失败")

            time.sleep(1.0)

            q_master_home = get_dh(self.master)
            q_slave_home = get_dh(self.slave)

            if q_master_home is None or q_slave_home is None:
                raise RuntimeError("无法读取主端或从端初始 DH 关节角")

            self.log_signal.emit("主端初始 DH 角 rad: " + str(["{:.4f}".format(x) for x in q_master_home]))
            self.log_signal.emit("从端初始 DH 角 rad: " + str(["{:.4f}".format(x) for x in q_slave_home]))

            self.log_signal.emit("主端切换到 MIT 模式...")
            if not self.master.switch_mode(MODE_MIT, None, self.pv_vel_lim):
                raise RuntimeError("主端切换 MIT 模式失败")

            self.log_signal.emit("从端切换到 PV 模式，并保持当前位置...")
            if not self.slave.switch_mode(MODE_PV, None, self.pv_vel_lim):
                raise RuntimeError("从端切换 PV 模式失败")

            self.done_signal.emit(q_master_home, q_slave_home)

        except Exception:
            err = traceback.format_exc()
            try:
                self.slave.cleanup()
            except Exception:
                pass
            try:
                self.master.cleanup()
            except Exception:
                pass
            self.fail_signal.emit(err)


class TeleopWorker(QThread):
    log_signal = Signal(str)
    status_signal = Signal(object)
    stopped_signal = Signal(str)
    fail_signal = Signal(str)

    def __init__(
        self,
        master: ArmController,
        slave: ArmController,
        q_master_home: np.ndarray,
        q_slave_home: np.ndarray,
        control_hz: float,
        pv_vel_lim: float,
        max_delta_per_cycle: float,
        alpha: float,
        scale: np.ndarray,
        motor_limit_margin: float,
        clip_print_interval: int,
    ):
        super().__init__()
        self.master = master
        self.slave = slave
        self.q_master_home = np.asarray(q_master_home, dtype=float).reshape(6)
        self.q_slave_home = np.asarray(q_slave_home, dtype=float).reshape(6)
        self.control_hz = float(control_hz)
        self.pv_vel_lim = float(pv_vel_lim)
        self.max_delta_per_cycle = float(max_delta_per_cycle)
        self.alpha = float(alpha)
        self.scale = np.asarray(scale, dtype=float).reshape(6)
        self.motor_limit_margin = float(motor_limit_margin)
        self.clip_print_interval = int(clip_print_interval)
        self._stop_requested = False

    def request_stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        q_cmd_last = self.q_slave_home.copy()
        dt = 1.0 / max(1e-6, self.control_hz)
        loop_count = 0
        last_status_time = 0.0
        last_hz_time = time.time()
        loop_for_hz = 0
        actual_hz = 0.0

        original_slave_log = self.slave.log

        def filtered_slave_log(msg: str) -> None:
            # 遥操作循环中每个周期都会写 PV 目标，原 ArmController 会大量打印。
            # 这里过滤掉高频重复日志，保留异常、夹紧、模式切换等关键日志。
            if str(msg).startswith("[PV] 已写入目标电机角"):
                return
            original_slave_log(msg)

        self.slave.log = filtered_slave_log

        try:
            self.log_signal.emit("开始单机同构遥操作")
            self.log_signal.emit("主端：MIT 重力补偿，可拖动；从端：PV 模式跟随")

            while not self._stop_requested:
                loop_start = time.time()
                loop_count += 1
                loop_for_hz += 1

                q_master = get_dh(self.master)
                if q_master is None:
                    time.sleep(dt)
                    continue

                delta_master = angle_diff(q_master, self.q_master_home)
                q_target_raw = self.q_slave_home + self.scale * delta_master
                q_target = limit_delta_continuous(q_target_raw, q_cmd_last, self.max_delta_per_cycle)
                q_target = lowpass_continuous(q_target, q_cmd_last, self.alpha)

                ok = safe_set_slave_target_dh(
                    self.slave,
                    q_target,
                    self.pv_vel_lim,
                    self.motor_limit_margin,
                    loop_count,
                    self.clip_print_interval,
                    self.log_signal.emit,
                )

                if ok:
                    q_cmd_last = q_target.copy()
                else:
                    if loop_count % 50 == 0:
                        self.log_signal.emit("[WARN] 从端目标写入失败，保持当前位置")
                    try:
                        self.slave.set_pv_hold_current_position(self.pv_vel_lim)
                    except Exception as e:
                        self.log_signal.emit(f"[WARN] 从端保持当前位置异常: {e}")

                now = time.time()
                if now - last_hz_time >= 1.0:
                    actual_hz = loop_for_hz / max(1e-6, now - last_hz_time)
                    loop_for_hz = 0
                    last_hz_time = now

                if now - last_status_time >= STATUS_EMIT_PERIOD_S:
                    q_slave = get_dh(self.slave)
                    self.status_signal.emit({
                        "loop_count": loop_count,
                        "actual_hz": actual_hz,
                        "q_master": q_master,
                        "q_slave": q_slave,
                        "q_target": q_cmd_last.copy(),
                    })
                    last_status_time = now

                elapsed = time.time() - loop_start
                time.sleep(max(0.0, dt - elapsed))

            self.log_signal.emit("已收到停止遥操作请求")

        except Exception:
            self.fail_signal.emit(traceback.format_exc())

        finally:
            self.slave.log = original_slave_log
            try:
                self.slave.set_pv_hold_current_position(self.pv_vel_lim)
            except Exception as e:
                self.log_signal.emit(f"[WARN] 停止后从端保持当前位置异常: {e}")
            self.stopped_signal.emit("遥操作已停止，从端已保持当前位置")


class CleanupWorker(QThread):
    log_signal = Signal(str)
    done_signal = Signal()
    fail_signal = Signal(str)

    def __init__(self, master: Optional[ArmController], slave: Optional[ArmController]):
        super().__init__()
        self.master = master
        self.slave = slave

    def run(self) -> None:
        try:
            if self.slave is not None:
                self.log_signal.emit("清理从端机械臂...")
                self.slave.cleanup()
            if self.master is not None:
                self.log_signal.emit("清理主端机械臂...")
                self.master.cleanup()
            self.done_signal.emit()
        except Exception:
            self.fail_signal.emit(traceback.format_exc())


# =========================
# 4. GUI 主窗口
# =========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("单机双臂同构遥操作 GUI")
        self.resize(1280, 820)

        self.devices: Dict[str, Dict[str, Any]] = {}
        self.master: Optional[ArmController] = None
        self.slave: Optional[ArmController] = None
        self.q_master_home: Optional[np.ndarray] = None
        self.q_slave_home: Optional[np.ndarray] = None

        self.scan_worker: Optional[DeviceScanWorker] = None
        self.init_worker: Optional[ArmInitWorker] = None
        self.teleop_worker: Optional[TeleopWorker] = None
        self.cleanup_worker: Optional[CleanupWorker] = None

        self._build_ui()
        self._set_initialized(False)
        self._set_teleop_running(False)

    # -------------------------
    # UI 构建
    # -------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        splitter = QSplitter(Qt.Vertical)
        root.addWidget(splitter)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        splitter.addWidget(top_widget)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        top_layout.addWidget(left_panel, 1)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        top_layout.addWidget(right_panel, 1)

        self._build_device_group(left_layout)
        self._build_param_group(left_layout)
        self._build_control_group(left_layout)

        self._build_status_group(right_layout)
        self._build_device_table_group(right_layout)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        splitter.addWidget(log_group)

        splitter.setSizes([460, 360])

    def _build_device_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("CANFD 设备配置")
        layout = QGridLayout(group)
        parent_layout.addWidget(group)

        self.master_serial_edit = QLineEdit(DEFAULT_MASTER_SERIAL)
        self.slave_serial_edit = QLineEdit(DEFAULT_SLAVE_SERIAL)

        self.scan_max_spin = QSpinBox()
        self.scan_max_spin.setRange(1, 16)
        self.scan_max_spin.setValue(DEFAULT_DEVICE_SCAN_MAX)

        self.channel_index_spin = QSpinBox()
        self.channel_index_spin.setRange(0, 8)
        self.channel_index_spin.setValue(DEFAULT_CHANNEL_INDEX)

        layout.addWidget(QLabel("主端 Serial"), 0, 0)
        layout.addWidget(self.master_serial_edit, 0, 1, 1, 3)
        layout.addWidget(QLabel("从端 Serial"), 1, 0)
        layout.addWidget(self.slave_serial_edit, 1, 1, 1, 3)
        layout.addWidget(QLabel("扫描最大 index"), 2, 0)
        layout.addWidget(self.scan_max_spin, 2, 1)
        layout.addWidget(QLabel("通道 index"), 2, 2)
        layout.addWidget(self.channel_index_spin, 2, 3)

        self.btn_scan = QPushButton("扫描 CANFD 设备")
        self.btn_init = QPushButton("初始化双臂并切换模式")
        self.btn_cleanup = QPushButton("清理/关闭双臂")

        self.btn_scan.clicked.connect(self.on_scan_clicked)
        self.btn_init.clicked.connect(self.on_init_clicked)
        self.btn_cleanup.clicked.connect(self.on_cleanup_clicked)

        layout.addWidget(self.btn_scan, 3, 0, 1, 2)
        layout.addWidget(self.btn_init, 3, 2, 1, 2)
        layout.addWidget(self.btn_cleanup, 4, 0, 1, 4)

    def _build_param_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("遥操作参数")
        layout = QGridLayout(group)
        parent_layout.addWidget(group)

        self.control_hz_spin = QDoubleSpinBox()
        self.control_hz_spin.setRange(1.0, 500.0)
        self.control_hz_spin.setDecimals(1)
        self.control_hz_spin.setSingleStep(10.0)
        self.control_hz_spin.setValue(DEFAULT_CONTROL_HZ)

        self.pv_vel_spin = QDoubleSpinBox()
        self.pv_vel_spin.setRange(0.001, 5.0)
        self.pv_vel_spin.setDecimals(3)
        self.pv_vel_spin.setSingleStep(0.05)
        self.pv_vel_spin.setValue(DEFAULT_PV_VEL_LIM)

        self.max_delta_spin = QDoubleSpinBox()
        self.max_delta_spin.setRange(0.0001, 1.0)
        self.max_delta_spin.setDecimals(5)
        self.max_delta_spin.setSingleStep(0.001)
        self.max_delta_spin.setValue(DEFAULT_MAX_DELTA_PER_CYCLE)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.001, 1.0)
        self.alpha_spin.setDecimals(3)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(DEFAULT_ALPHA)

        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 0.1)
        self.margin_spin.setDecimals(5)
        self.margin_spin.setSingleStep(0.001)
        self.margin_spin.setValue(DEFAULT_MOTOR_LIMIT_MARGIN)

        self.clip_interval_spin = QSpinBox()
        self.clip_interval_spin.setRange(1, 10000)
        self.clip_interval_spin.setValue(DEFAULT_CLIP_PRINT_INTERVAL)

        layout.addWidget(QLabel("CONTROL_HZ"), 0, 0)
        layout.addWidget(self.control_hz_spin, 0, 1)
        layout.addWidget(QLabel("PV_VEL_LIM"), 0, 2)
        layout.addWidget(self.pv_vel_spin, 0, 3)
        layout.addWidget(QLabel("MAX_DELTA_PER_CYCLE"), 1, 0)
        layout.addWidget(self.max_delta_spin, 1, 1)
        layout.addWidget(QLabel("ALPHA"), 1, 2)
        layout.addWidget(self.alpha_spin, 1, 3)
        layout.addWidget(QLabel("MOTOR_LIMIT_MARGIN"), 2, 0)
        layout.addWidget(self.margin_spin, 2, 1)
        layout.addWidget(QLabel("CLIP_PRINT_INTERVAL"), 2, 2)
        layout.addWidget(self.clip_interval_spin, 2, 3)

        self.scale_spins = []
        for i in range(6):
            spin = QDoubleSpinBox()
            spin.setRange(-3.0, 3.0)
            spin.setDecimals(2)
            spin.setSingleStep(0.1)
            spin.setValue(float(DEFAULT_SCALE[i]))
            self.scale_spins.append(spin)
            layout.addWidget(QLabel(f"SCALE J{i + 1}"), 3 + i // 3, (i % 3) * 2)
            layout.addWidget(spin, 3 + i // 3, (i % 3) * 2 + 1)

    def _build_control_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("运行控制")
        layout = QGridLayout(group)
        parent_layout.addWidget(group)

        self.btn_start = QPushButton("开始遥操作")
        self.btn_stop = QPushButton("停止遥操作并保持从端")
        self.btn_clear_log = QPushButton("清空日志")

        self.btn_start.clicked.connect(self.on_start_clicked)
        self.btn_stop.clicked.connect(self.on_stop_clicked)
        self.btn_clear_log.clicked.connect(lambda: self.log_edit.clear())

        self.check_disable_on_exit = QCheckBox("清理时失能并关闭设备")
        self.check_disable_on_exit.setChecked(True)

        self.state_label = QLabel("状态：未初始化")
        self.state_label.setStyleSheet("font-weight: bold;")

        layout.addWidget(self.btn_start, 0, 0)
        layout.addWidget(self.btn_stop, 0, 1)
        layout.addWidget(self.btn_clear_log, 0, 2)
        layout.addWidget(self.check_disable_on_exit, 1, 0, 1, 2)
        layout.addWidget(self.state_label, 1, 2)

    def _build_status_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("双臂 DH 状态")
        layout = QVBoxLayout(group)
        parent_layout.addWidget(group, 1)

        self.status_summary_label = QLabel("实际循环频率：-- Hz | loop：--")
        layout.addWidget(self.status_summary_label)

        self.status_table = QTableWidget(6, 5)
        self.status_table.setHorizontalHeaderLabels(["关节", "主端 DH(rad)", "从端 DH(rad)", "目标 DH(rad)", "SCALE"])
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for r in range(6):
            for c in range(5):
                item = QTableWidgetItem("--")
                item.setTextAlignment(Qt.AlignCenter)
                self.status_table.setItem(r, c, item)
            self.status_table.item(r, 0).setText(f"J{r + 1}")
        layout.addWidget(self.status_table)

    def _build_device_table_group(self, parent_layout: QVBoxLayout) -> None:
        group = QGroupBox("设备扫描结果")
        layout = QVBoxLayout(group)
        parent_layout.addWidget(group, 1)

        self.device_table = QTableWidget(0, 5)
        self.device_table.setHorizontalHeaderLabels(["匹配角色", "device_index", "serial", "hw_type", "can_num"])
        self.device_table.verticalHeader().setVisible(False)
        self.device_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.device_table)

    # -------------------------
    # UI 辅助
    # -------------------------

    def append_log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        text = str(msg)
        if not text.startswith("["):
            text = f"[{ts}] {text}"
        self.log_edit.append(text)
        self.log_edit.moveCursor(QTextCursor.End)

    def set_status_item(self, row: int, col: int, text: str) -> None:
        item = self.status_table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            item.setTextAlignment(Qt.AlignCenter)
            self.status_table.setItem(row, col, item)
        item.setText(text)

    def _set_initialized(self, initialized: bool) -> None:
        self.btn_start.setEnabled(initialized)
        self.btn_cleanup.setEnabled(initialized)
        self.btn_init.setEnabled(not initialized)
        self.btn_scan.setEnabled(not initialized)
        self.check_disable_on_exit.setEnabled(not initialized)
        if initialized:
            self.state_label.setText("状态：已初始化，等待开始遥操作")
        else:
            self.state_label.setText("状态：未初始化")

    def _set_teleop_running(self, running: bool) -> None:
        self.btn_start.setEnabled((not running) and self.master is not None and self.slave is not None)
        self.btn_stop.setEnabled(running)
        self.btn_cleanup.setEnabled((not running) and self.master is not None and self.slave is not None)
        self.btn_init.setEnabled((not running) and self.master is None and self.slave is None)
        self.btn_scan.setEnabled((not running) and self.master is None and self.slave is None)
        self.control_hz_spin.setEnabled(not running)
        self.pv_vel_spin.setEnabled(not running)
        self.max_delta_spin.setEnabled(not running)
        self.alpha_spin.setEnabled(not running)
        self.margin_spin.setEnabled(not running)
        self.clip_interval_spin.setEnabled(not running)
        for spin in self.scale_spins:
            spin.setEnabled(not running)
        if running:
            self.state_label.setText("状态：遥操作运行中")

    def read_scale(self) -> np.ndarray:
        return np.array([float(spin.value()) for spin in self.scale_spins], dtype=float)

    def update_device_table(self) -> None:
        master_serial = normalize_serial(self.master_serial_edit.text())
        slave_serial = normalize_serial(self.slave_serial_edit.text())

        self.device_table.setRowCount(len(self.devices))
        for row, (serial, item) in enumerate(self.devices.items()):
            if serial == master_serial:
                role = "主端"
            elif serial == slave_serial:
                role = "从端"
            else:
                role = "未匹配"

            values = [
                role,
                str(item.get("device_index", "")),
                serial,
                str(item.get("hw_type", "")),
                str(item.get("can_num", "")),
            ]
            for col, text in enumerate(values):
                table_item = QTableWidgetItem(text)
                table_item.setTextAlignment(Qt.AlignCenter)
                self.device_table.setItem(row, col, table_item)

    def update_status_table(self, q_master=None, q_slave=None, q_target=None, actual_hz=None, loop_count=None) -> None:
        scale = self.read_scale()
        for i in range(6):
            self.set_status_item(i, 4, f"{scale[i]:.2f}")
            if q_master is not None:
                self.set_status_item(i, 1, f"{float(q_master[i]):.4f}")
            if q_slave is not None:
                self.set_status_item(i, 2, f"{float(q_slave[i]):.4f}")
            if q_target is not None:
                self.set_status_item(i, 3, f"{float(q_target[i]):.4f}")

        hz_txt = "--" if actual_hz is None else f"{float(actual_hz):.1f}"
        loop_txt = "--" if loop_count is None else str(int(loop_count))
        self.status_summary_label.setText(f"实际循环频率：{hz_txt} Hz | loop：{loop_txt}")

    # -------------------------
    # 按钮回调
    # -------------------------

    def on_scan_clicked(self) -> None:
        if self.scan_worker is not None and self.scan_worker.isRunning():
            self.append_log("[WARN] 正在扫描，请稍后")
            return

        self.btn_scan.setEnabled(False)
        self.append_log("开始扫描 CANFD 设备")
        self.scan_worker = DeviceScanWorker(self.scan_max_spin.value())
        self.scan_worker.log_signal.connect(self.append_log)
        self.scan_worker.done_signal.connect(self.on_scan_done)
        self.scan_worker.fail_signal.connect(self.on_scan_failed)
        self.scan_worker.start()

    def on_scan_done(self, devices: object) -> None:
        self.devices = dict(devices)
        self.update_device_table()
        self.btn_scan.setEnabled(True)
        self.append_log("[DONE] 设备扫描完成")

        try:
            get_device_index_by_serial(self.devices, self.master_serial_edit.text(), "主端", self.append_log)
            get_device_index_by_serial(self.devices, self.slave_serial_edit.text(), "从端", self.append_log)
        except Exception as e:
            self.append_log(f"[WARN] serial 自动匹配未完全成功: {e}")

    def on_scan_failed(self, err: str) -> None:
        self.btn_scan.setEnabled(True)
        self.append_log("[ERR] 设备扫描异常：")
        self.append_log(err)

    def on_init_clicked(self) -> None:
        if not self.devices:
            QMessageBox.warning(self, "提示", "请先点击“扫描 CANFD 设备”，确认主端和从端 serial 都能匹配。")
            return

        if self.master is not None or self.slave is not None:
            QMessageBox.warning(self, "提示", "双臂已经初始化。如需重新初始化，请先清理/关闭双臂。")
            return

        try:
            master_index = get_device_index_by_serial(self.devices, self.master_serial_edit.text(), "主端", self.append_log)
            slave_index = get_device_index_by_serial(self.devices, self.slave_serial_edit.text(), "从端", self.append_log)
            if master_index == slave_index:
                raise RuntimeError("主端和从端匹配到了同一个 device_index，请检查 serial 配置")

            channel_index = int(self.channel_index_spin.value())
            self.master = create_arm_controller(master_index, channel_index, "master")
            self.slave = create_arm_controller(slave_index, channel_index, "slave")

            self.master.log_signal.connect(lambda msg: self.append_log("[主端] " + msg))
            self.slave.log_signal.connect(lambda msg: self.append_log("[从端] " + msg))

            disable_on_exit = self.check_disable_on_exit.isChecked()
            self.master.disable_on_exit = disable_on_exit
            self.slave.disable_on_exit = disable_on_exit

            self.btn_init.setEnabled(False)
            self.btn_scan.setEnabled(False)
            self.append_log("开始初始化双臂并切换模式")

            self.init_worker = ArmInitWorker(self.master, self.slave, self.pv_vel_spin.value())
            self.init_worker.log_signal.connect(self.append_log)
            self.init_worker.done_signal.connect(self.on_init_done)
            self.init_worker.fail_signal.connect(self.on_init_failed)
            self.init_worker.start()

        except Exception as e:
            self.append_log(f"[ERR] 初始化前检查失败: {e}")
            self.master = None
            self.slave = None
            self._set_initialized(False)

    def on_init_done(self, q_master_home: object, q_slave_home: object) -> None:
        self.q_master_home = np.asarray(q_master_home, dtype=float).reshape(6)
        self.q_slave_home = np.asarray(q_slave_home, dtype=float).reshape(6)
        self.update_status_table(q_master=self.q_master_home, q_slave=self.q_slave_home, q_target=self.q_slave_home)
        self._set_initialized(True)
        self._set_teleop_running(False)
        self.append_log("[DONE] 双臂初始化完成：主端 MIT，从端 PV 保持当前位置")

    def on_init_failed(self, err: str) -> None:
        self.append_log("[ERR] 双臂初始化失败：")
        self.append_log(err)
        self.master = None
        self.slave = None
        self.q_master_home = None
        self.q_slave_home = None
        self._set_initialized(False)
        self._set_teleop_running(False)

    def on_start_clicked(self) -> None:
        if self.master is None or self.slave is None or self.q_master_home is None or self.q_slave_home is None:
            QMessageBox.warning(self, "提示", "请先初始化双臂。")
            return

        if self.teleop_worker is not None and self.teleop_worker.isRunning():
            self.append_log("[WARN] 遥操作已经在运行")
            return

        self.teleop_worker = TeleopWorker(
            master=self.master,
            slave=self.slave,
            q_master_home=self.q_master_home,
            q_slave_home=self.q_slave_home,
            control_hz=self.control_hz_spin.value(),
            pv_vel_lim=self.pv_vel_spin.value(),
            max_delta_per_cycle=self.max_delta_spin.value(),
            alpha=self.alpha_spin.value(),
            scale=self.read_scale(),
            motor_limit_margin=self.margin_spin.value(),
            clip_print_interval=self.clip_interval_spin.value(),
        )
        self.teleop_worker.log_signal.connect(self.append_log)
        self.teleop_worker.status_signal.connect(self.on_teleop_status)
        self.teleop_worker.stopped_signal.connect(self.on_teleop_stopped)
        self.teleop_worker.fail_signal.connect(self.on_teleop_failed)

        self._set_teleop_running(True)
        self.teleop_worker.start()

    def on_stop_clicked(self) -> None:
        if self.teleop_worker is not None and self.teleop_worker.isRunning():
            self.append_log("正在请求停止遥操作...")
            self.teleop_worker.request_stop()
        else:
            self.append_log("[INFO] 当前没有正在运行的遥操作")

    def on_teleop_status(self, status: object) -> None:
        data = dict(status)
        self.update_status_table(
            q_master=data.get("q_master"),
            q_slave=data.get("q_slave"),
            q_target=data.get("q_target"),
            actual_hz=data.get("actual_hz"),
            loop_count=data.get("loop_count"),
        )

    def on_teleop_stopped(self, msg: str) -> None:
        self.append_log(f"[DONE] {msg}")
        self._set_teleop_running(False)
        if self.master is not None and self.slave is not None:
            self._set_initialized(True)

    def on_teleop_failed(self, err: str) -> None:
        self.append_log("[ERR] 遥操作线程异常：")
        self.append_log(err)

    def on_cleanup_clicked(self) -> None:
        if self.teleop_worker is not None and self.teleop_worker.isRunning():
            self.append_log("清理前先停止遥操作...")
            self.teleop_worker.request_stop()
            self.teleop_worker.wait(1500)

        if self.master is None and self.slave is None:
            self.append_log("[INFO] 当前没有需要清理的机械臂")
            return

        reply = QMessageBox.question(
            self,
            "确认清理",
            "是否清理并关闭主端/从端机械臂？\n如果勾选了“清理时失能并关闭设备”，会尝试失能电机并关闭 CANFD。",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        disable_on_exit = self.check_disable_on_exit.isChecked()
        if self.master is not None:
            self.master.disable_on_exit = disable_on_exit
        if self.slave is not None:
            self.slave.disable_on_exit = disable_on_exit

        self.btn_cleanup.setEnabled(False)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.append_log("开始清理双臂...")

        self.cleanup_worker = CleanupWorker(self.master, self.slave)
        self.cleanup_worker.log_signal.connect(self.append_log)
        self.cleanup_worker.done_signal.connect(self.on_cleanup_done)
        self.cleanup_worker.fail_signal.connect(self.on_cleanup_failed)
        self.cleanup_worker.start()

    def on_cleanup_done(self) -> None:
        self.append_log("[DONE] 双臂清理完成")
        self.master = None
        self.slave = None
        self.q_master_home = None
        self.q_slave_home = None
        self._set_initialized(False)
        self._set_teleop_running(False)
        self.btn_scan.setEnabled(True)
        self.btn_init.setEnabled(True)

    def on_cleanup_failed(self, err: str) -> None:
        self.append_log("[ERR] 双臂清理异常：")
        self.append_log(err)
        self.master = None
        self.slave = None
        self.q_master_home = None
        self.q_slave_home = None
        self._set_initialized(False)
        self._set_teleop_running(False)
        self.btn_scan.setEnabled(True)
        self.btn_init.setEnabled(True)

    # -------------------------
    # 关闭窗口
    # -------------------------

    def closeEvent(self, event) -> None:
        if self.teleop_worker is not None and self.teleop_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "确认退出",
                "遥操作仍在运行，是否停止遥操作并退出？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.teleop_worker.request_stop()
            self.teleop_worker.wait(1500)

        if self.master is not None or self.slave is not None:
            reply = QMessageBox.question(
                self,
                "确认退出",
                "是否在退出前清理并关闭双臂？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                try:
                    if self.slave is not None:
                        self.slave.cleanup()
                    if self.master is not None:
                        self.master.cleanup()
                except Exception as e:
                    self.append_log(f"[WARN] 退出清理异常: {e}")

        event.accept()


# =========================
# 5. 程序入口
# =========================

def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
