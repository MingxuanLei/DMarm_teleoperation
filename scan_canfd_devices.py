#运行代码结果，白色机械臂做主端，对应index1,serial=9820ECA9B0E80D6418B0
#黑色机械臂做从端，对应index0，serial=EBD6C68A50FF0DD4F0B0
from zlgcan import ZCAN, ZCAN_USBCANFD_MINI


def scan_devices(max_index=8):
    zcan = ZCAN()

    print("开始扫描 CANFD 设备...")
    print("=" * 60)

    found = []

    for idx in range(max_index):
        handle = zcan.OpenDevice(ZCAN_USBCANFD_MINI, idx, 0)

        if int(handle) == 0:
            print(f"device_index={idx}: 打开失败")
            continue

        info = zcan.GetDeviceInf(handle)

        if info is not None:
            print(f"device_index={idx}: 打开成功")
            print(f"  handle     = {int(handle)}")
            print(f"  serial     = {info.serial}")
            print(f"  hw_type    = {info.hw_type}")
            print(f"  can_num    = {info.can_num}")
            print(f"  hw_version = {info.hw_version}")
            print(f"  fw_version = {info.fw_version}")
            found.append((idx, info.serial, info.hw_type))
        else:
            print(f"device_index={idx}: 打开成功，但读取设备信息失败")

        zcan.CloseDevice(handle)

    print("=" * 60)
    print(f"共发现 {len(found)} 个可打开的 CANFD 设备")

    for idx, serial, hw_type in found:
        print(f"device_index={idx}, serial={serial}, hw_type={hw_type}")


if __name__ == "__main__":
    scan_devices()