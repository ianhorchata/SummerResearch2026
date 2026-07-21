#!/usr/bin/env python3
"""Reusable controller for Hiwonder/LewanSoul LX bus servos via the BusLinker board.

LX-series servos (LX-224, LX-15D, etc.) use the LewanSoul serial bus protocol:
  * 115200 baud, half-duplex single-wire TTL bridged over USB
  * frames: 0x55 0x55 ID LENGTH CMD [PARAMS...] CHECKSUM
  * positions are 0-1000 (maps to 0-240 degrees, ~500 = center)
  * motion is TIME-based: "go to position over N milliseconds"

Only dependency is pyserial:

    pip install pyserial

Usage as a library:

    from hiwonder_servo import HiwonderServoBus

    with HiwonderServoBus("/dev/ttyACM0") as bus:
        print(bus.scan())
        print(bus.read_position(1))
        bus.move(1, 500, time_ms=1000)   # center over 1 s

Usage as a CLI:

    python3 hiwonder_servo.py scan
    python3 hiwonder_servo.py ping 1
    python3 hiwonder_servo.py read 1
    python3 hiwonder_servo.py move 1 500 --time 1000
    python3 hiwonder_servo.py sweep 1 --low 200 --high 800 --time 800 --cycles 3
    python3 hiwonder_servo.py torque 1 --off
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Tuple

try:
    import serial  # pyserial
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Install it with:\n    pip install pyserial\n"
        f"(import error: {exc})"
    )

# -- LewanSoul LX protocol commands ---------------------------------------
CMD_MOVE_TIME_WRITE = 1
CMD_MOVE_TIME_READ = 2
CMD_MOVE_TIME_WAIT_WRITE = 7
CMD_MOVE_START = 11
CMD_ID_WRITE = 13
BROADCAST_ID = 254
CMD_ID_READ = 14
CMD_ANGLE_OFFSET_READ = 19
CMD_ANGLE_LIMIT_WRITE = 20
CMD_ANGLE_LIMIT_READ = 21
CMD_VIN_LIMIT_WRITE = 22
CMD_VIN_LIMIT_READ = 23
CMD_TEMP_MAX_LIMIT_WRITE = 24
CMD_TEMP_MAX_LIMIT_READ = 25
CMD_TEMP_READ = 26
CMD_VIN_READ = 27
CMD_POS_READ = 28
CMD_LOAD_OR_UNLOAD_WRITE = 31
CMD_LOAD_OR_UNLOAD_READ = 32
CMD_LED_ERROR_WRITE = 35
CMD_LED_ERROR_READ = 36

# NOTE: The LewanSoul LX protocol has NO command to read live fault status.
# Command 36 (LED_ERROR) is only the *alarm-enable mask* (which conditions blink
# the LED), so it cannot be used to detect active faults. Health is instead
# derived by comparing live temperature/voltage against the servo's own limits,
# and by checking whether the servo has auto-unloaded its torque.

POS_MIN, POS_MAX = 0, 1000
DEG_PER_UNIT = 240 / 1000  # 0.24 degrees per position unit
ALARM_BITS = {0x01: "over-temp", 0x02: "over-voltage", 0x04: "locked-rotor"}


class ServoError(RuntimeError):
    """Raised when a servo command fails or a servo does not respond."""


class HiwonderServoBus:
    """Direct LewanSoul-protocol driver for a serial bus of LX servos."""

    def __init__(self, port: str = "/dev/ttyACM0", baudrate: int = 115200,
                 timeout: float = 0.05):
        self.port = port
        self.baudrate = baudrate
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
        except serial.SerialException as exc:
            raise ServoError(f"Failed to open {port!r}: {exc}")

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    def __enter__(self) -> "HiwonderServoBus":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # -- low-level packet I/O ---------------------------------------------
    @staticmethod
    def _frame(servo_id: int, command: int, params: List[int]) -> bytes:
        length = len(params) + 3
        body = [servo_id, length, command, *params]
        checksum = (~sum(body)) & 0xFF
        return bytes([0x55, 0x55, *body, checksum])

    def _recv_frame(self, deadline: float) -> Optional[Tuple[int, int, List[int]]]:
        """Read one 0x55 0x55 frame. Returns (id, command, params) or None."""
        while time.time() < deadline:
            if self.ser.read(1) == b"\x55" and self.ser.read(1) == b"\x55":
                break
        else:
            return None
        head = self.ser.read(2)  # id, length
        if len(head) < 2:
            return None
        servo_id, length = head[0], head[1]
        rest = self.ser.read(length - 1)  # command + params + checksum
        if len(rest) < length - 1 or length < 3:
            return None
        command = rest[0]
        params = list(rest[1:-1])
        return servo_id, command, params

    def _send(self, servo_id: int, command: int, params: Optional[List[int]] = None) -> None:
        self.ser.reset_input_buffer()
        self.ser.write(self._frame(servo_id, command, params or []))

    def _query(self, servo_id: int, command: int, expected_params: int,
               params: Optional[List[int]] = None) -> List[int]:
        """Send a read command and return the response params (ignores TX echo)."""
        self._send(servo_id, command, params)
        deadline = time.time() + 0.25
        while True:
            frame = self._recv_frame(deadline)
            if frame is None:
                raise ServoError(f"no response to command {command} on id {servo_id}")
            rid, rcmd, rparams = frame
            # Skip our own echoed command (half-duplex bus): real reply carries params.
            if rcmd == command and len(rparams) >= expected_params:
                return rparams

    @staticmethod
    def _s16(lo: int, hi: int) -> int:
        val = lo | (hi << 8)
        return val - 0x10000 if val & 0x8000 else val

    # -- public API --------------------------------------------------------
    def ping(self, servo_id: int) -> bool:
        try:
            self._query(servo_id, CMD_ID_READ, 1)
            return True
        except ServoError:
            return False

    def scan(self, id_range=range(0, 21)) -> List[int]:
        return [sid for sid in id_range if self.ping(sid)]

    def set_id(self, current_id: int, new_id: int) -> None:
        """Persist a new ID (0-253) to the servo's EEPROM.

        Only ONE servo should be on the bus when using the broadcast ID (254),
        otherwise every servo will adopt the same new ID.
        """
        if not 0 <= new_id <= 253:
            raise ServoError(f"new id {new_id} out of range (0-253)")
        self._send(current_id, CMD_ID_WRITE, [new_id])

    def read_position(self, servo_id: int) -> int:
        p = self._query(servo_id, CMD_POS_READ, 2)
        return self._s16(p[0], p[1])

    def read_voltage(self, servo_id: int) -> float:
        p = self._query(servo_id, CMD_VIN_READ, 2)
        return (p[0] | (p[1] << 8)) / 1000.0  # mV -> V

    def read_temperature(self, servo_id: int) -> int:
        return self._query(servo_id, CMD_TEMP_READ, 1)[0]

    def read_angle_offset(self, servo_id: int) -> int:
        raw = self._query(servo_id, CMD_ANGLE_OFFSET_READ, 1)[0]
        return raw - 256 if raw & 0x80 else raw  # signed byte (-125..125)

    def read_angle_limits(self, servo_id: int) -> Tuple[int, int]:
        """Return the stored (min, max) position limits (0-1000)."""
        p = self._query(servo_id, CMD_ANGLE_LIMIT_READ, 4)
        low = p[0] | (p[1] << 8)
        high = p[2] | (p[3] << 8)
        return low, high

    def set_angle_limits(self, servo_id: int, low: int, high: int) -> None:
        """Persist new (min, max) position limits (0-1000) in the servo's EEPROM."""
        low = max(POS_MIN, min(POS_MAX, int(low)))
        high = max(POS_MIN, min(POS_MAX, int(high)))
        params = [low & 0xFF, (low >> 8) & 0xFF, high & 0xFF, (high >> 8) & 0xFF]
        self._send(servo_id, CMD_ANGLE_LIMIT_WRITE, params)

    def set_torque(self, servo_id: int, on: bool = True) -> None:
        self._send(servo_id, CMD_LOAD_OR_UNLOAD_WRITE, [1 if on else 0])

    def is_torque_on(self, servo_id: int) -> bool:
        return bool(self._query(servo_id, CMD_LOAD_OR_UNLOAD_READ, 1)[0])

    def read_temp_limit(self, servo_id: int) -> int:
        """Maximum allowed temperature (deg C) stored in the servo."""
        return self._query(servo_id, CMD_TEMP_MAX_LIMIT_READ, 1)[0]

    def set_temp_limit(self, servo_id: int, temp_c: int) -> None:
        """Persist maximum allowed temperature (50-100 deg C) in the servo."""
        temp_c = max(50, min(100, int(temp_c)))
        self._send(servo_id, CMD_TEMP_MAX_LIMIT_WRITE, [temp_c])

    def read_voltage_limits(self, servo_id: int) -> Tuple[float, float]:
        """Stored (min, max) input-voltage limits in volts."""
        p = self._query(servo_id, CMD_VIN_LIMIT_READ, 4)
        vmin = (p[0] | (p[1] << 8)) / 1000.0
        vmax = (p[2] | (p[3] << 8)) / 1000.0
        return vmin, vmax

    def set_voltage_limits(self, servo_id: int, min_mv: int, max_mv: int) -> None:
        """Persist input-voltage limits in millivolts."""
        min_mv = max(4500, min(12000, int(min_mv)))
        max_mv = max(4500, min(12000, int(max_mv)))
        if min_mv >= max_mv:
            raise ServoError("minimum voltage limit must be less than maximum")
        params = [min_mv & 0xFF, (min_mv >> 8) & 0xFF,
                  max_mv & 0xFF, (max_mv >> 8) & 0xFF]
        self._send(servo_id, CMD_VIN_LIMIT_WRITE, params)

    def read_alarm_mask(self, servo_id: int) -> int:
        """Return which alarm conditions are enabled as a bitmask."""
        return self._query(servo_id, CMD_LED_ERROR_READ, 1)[0]

    def set_alarm_mask(self, servo_id: int, mask: int) -> None:
        """Persist enabled alarm conditions. 1=temp, 2=voltage, 4=locked rotor."""
        if not 0 <= int(mask) <= 7:
            raise ServoError("alarm mask must be 0-7")
        self._send(servo_id, CMD_LED_ERROR_WRITE, [int(mask)])

    @staticmethod
    def format_alarm_mask(mask: int) -> str:
        enabled = [name for bit, name in ALARM_BITS.items() if mask & bit]
        return ",".join(enabled) if enabled else "none"

    def health(self, servo_id: int) -> List[str]:
        """Return a list of real issues (empty = healthy), derived from measurements.

        The protocol can't report *why* a servo faulted, so this infers it:
        compares live temp/voltage to the servo's own limits and reports if the
        servo has auto-unloaded its torque (the tell-tale of overload protection).
        """
        issues: List[str] = []
        temp = self.read_temperature(servo_id)
        tmax = self.read_temp_limit(servo_id)
        if temp >= tmax:
            issues.append(f"over-temperature ({temp}>={tmax}C)")
        volt = self.read_voltage(servo_id)
        vmin, vmax = self.read_voltage_limits(servo_id)
        if volt < vmin:
            issues.append(f"under-voltage ({volt:.2f}<{vmin:.2f}V)")
        if volt > vmax:
            issues.append(f"over-voltage ({volt:.2f}>{vmax:.2f}V)")
        if not self.is_torque_on(servo_id):
            issues.append("torque auto-unloaded (likely overload/stall protection)")
        return issues

    def move(self, servo_id: int, position: int, time_ms: int = 1000) -> None:
        """Move to `position` (0-1000) over `time_ms` milliseconds."""
        position = max(POS_MIN, min(POS_MAX, int(position)))
        time_ms = max(0, min(30000, int(time_ms)))
        params = [position & 0xFF, (position >> 8) & 0xFF,
                  time_ms & 0xFF, (time_ms >> 8) & 0xFF]
        self._send(servo_id, CMD_MOVE_TIME_WRITE, params)

    def move_many(self, targets: dict, time_ms: int = 1000, sync: bool = True) -> None:
        """Move several servos at once. `targets` maps {servo_id: position}.

        With sync=True (default) each move is preloaded with MOVE_TIME_WAIT_WRITE
        and then released together via a broadcast MOVE_START, so all servos begin
        moving on the same trigger. With sync=False each command takes effect as it
        arrives.
        """
        if not targets:
            return
        command = CMD_MOVE_TIME_WAIT_WRITE if sync else CMD_MOVE_TIME_WRITE
        for servo_id, position in targets.items():
            position = max(POS_MIN, min(POS_MAX, int(position)))
            t = max(0, min(30000, int(time_ms)))
            params = [position & 0xFF, (position >> 8) & 0xFF,
                      t & 0xFF, (t >> 8) & 0xFF]
            self._send(servo_id, command, params)
        if sync:
            self._send(BROADCAST_ID, CMD_MOVE_START)

    def move_deg(self, servo_id: int, degrees: float, time_ms: int = 1000) -> None:
        """Move to an angle in degrees (0-240) over `time_ms` milliseconds."""
        self.move(servo_id, round(degrees / DEG_PER_UNIT), time_ms)

    def move_at_speed(self, servo_id: int, position: int, deg_per_s: float = 60.0) -> int:
        """Move to `position` capped at `deg_per_s`, auto-computing the travel time.

        Keeps the motor in its high-torque (lower-speed) regime so it can drive
        against gravity. Returns the time_ms used.
        """
        position = max(POS_MIN, min(POS_MAX, int(position)))
        current = self.read_position(servo_id)
        distance_deg = abs(position - current) * DEG_PER_UNIT
        time_ms = max(20, int(distance_deg / max(deg_per_s, 1.0) * 1000))
        self.move(servo_id, position, time_ms)
        return time_ms

    def read_degrees(self, servo_id: int) -> float:
        return self.read_position(servo_id) * DEG_PER_UNIT


# -- CLI -------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Control Hiwonder/LewanSoul LX bus servos.")
    p.add_argument("--port", default="/dev/ttyACM0", help="serial device (default: /dev/ttyACM0)")
    p.add_argument("--baud", type=int, default=115200, help="baud rate (default: 115200)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("scan", help="scan a range of IDs and list responders")
    sc.add_argument("--max", type=int, default=20, help="highest ID to probe (default: 20)")

    sa = sub.add_parser("readall", help="scan the bus and print status for every servo found")
    sa.add_argument("--max", type=int, default=20, help="highest ID to probe (default: 20)")

    sp = sub.add_parser("ping", help="ping a single servo")
    sp.add_argument("id", type=int)

    sr = sub.add_parser("read", help="read a servo's position, voltage, temperature")
    sr.add_argument("id", type=int)

    sm = sub.add_parser("move", help="move a servo to a position (0-1000)")
    sm.add_argument("id", type=int)
    sm.add_argument("position", type=int)
    sm.add_argument("--time", type=int, default=1000, help="travel time in ms")
    sm.add_argument("--speed", type=float, default=None,
                    help="cap speed in deg/s (auto-computes time; overrides --time)")

    smm = sub.add_parser("move-many", help="move several servos at once, e.g. move-many 1:500 2:300")
    smm.add_argument("targets", nargs="+", metavar="ID:POS", help="ID:position pairs (position 0-1000)")
    smm.add_argument("--time", type=int, default=1000, help="travel time in ms")
    smm.add_argument("--no-sync", action="store_true", help="send sequentially instead of a synced start")

    sd = sub.add_parser("move-deg", help="move a servo to an angle (0-240 degrees)")
    sd.add_argument("id", type=int)
    sd.add_argument("degrees", type=float)
    sd.add_argument("--time", type=int, default=1000, help="travel time in ms")

    si = sub.add_parser("set-id", help="change a servo's ID (persisted to EEPROM)")
    si.add_argument("current", type=int, help="current ID (use 254 to broadcast)")
    si.add_argument("new", type=int, help="new ID (0-253)")

    sf = sub.add_parser("fault", help="read fault/torque status for one or more servos")
    sf.add_argument("ids", type=int, nargs="*", help="servo IDs (default: scan the bus)")

    spr = sub.add_parser("protection", help="read voltage/temp/alarm protection settings")
    spr.add_argument("id", type=int)

    stl = sub.add_parser("set-temp-limit", help="write max temperature limit (50-100 C)")
    stl.add_argument("id", type=int)
    stl.add_argument("temp", type=int)

    svl = sub.add_parser("set-voltage-limits", help="write voltage limits in millivolts")
    svl.add_argument("id", type=int)
    svl.add_argument("min_mv", type=int)
    svl.add_argument("max_mv", type=int)

    sam = sub.add_parser("set-alarm-mask", help="write alarm mask: 1=temp, 2=voltage, 4=locked-rotor")
    sam.add_argument("id", type=int)
    sam.add_argument("mask", type=int)

    smon = sub.add_parser("monitor", help="continuously print voltage/position/torque/faults (Ctrl-C to stop)")
    smon.add_argument("ids", type=int, nargs="+", help="servo IDs to watch")
    smon.add_argument("--hz", type=float, default=10.0, help="samples per second (default: 10)")

    sl = sub.add_parser("limits", help="read stored angle limits and offset")
    sl.add_argument("id", type=int)

    ss = sub.add_parser("set-limits", help="write stored angle limits (0-1000)")
    ss.add_argument("id", type=int)
    ss.add_argument("low", type=int)
    ss.add_argument("high", type=int)

    sw = sub.add_parser("sweep", help="sweep a servo back and forth")
    sw.add_argument("id", type=int)
    sw.add_argument("--low", type=int, default=200)
    sw.add_argument("--high", type=int, default=800)
    sw.add_argument("--time", type=int, default=800, help="travel time per leg in ms")
    sw.add_argument("--cycles", type=int, default=3)

    st = sub.add_parser("torque", help="enable/disable holding torque on one or more servos")
    st.add_argument("ids", type=int, nargs="*", help="one or more servo IDs")
    st.add_argument("--all", action="store_true", help="target every servo found on the bus")
    st.add_argument("--off", action="store_true", help="release torque (default: enable)")
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        with HiwonderServoBus(args.port, args.baud) as bus:
            if args.cmd == "scan":
                found = bus.scan(range(0, args.max + 1))
                print("Found servo IDs:", found if found else "none (check power/jumpers/baud)")
            elif args.cmd == "readall":
                found = bus.scan(range(0, args.max + 1))
                if not found:
                    print("No servos found (check power/jumpers/baud).")
                else:
                    print(f"{'ID':>3}  {'pos':>5}  {'deg':>6}  {'volt':>5}  {'temp':>4}  "
                          f"{'limits':>11}  off")
                    for sid in found:
                        try:
                            pos = bus.read_position(sid)
                            volt = bus.read_voltage(sid)
                            temp = bus.read_temperature(sid)
                            low, high = bus.read_angle_limits(sid)
                            off = bus.read_angle_offset(sid)
                            print(f"{sid:>3}  {pos:>5}  {pos * DEG_PER_UNIT:>6.1f}  "
                                  f"{volt:>5.2f}  {temp:>4}  {f'{low}-{high}':>11}  {off}")
                        except ServoError as exc:
                            print(f"{sid:>3}  error: {exc}")
            elif args.cmd == "ping":
                print(f"ID {args.id}: {'responded' if bus.ping(args.id) else 'NO RESPONSE'}")
            elif args.cmd == "read":
                pos = bus.read_position(args.id)
                print(f"position   : {pos} (0-1000)  = {pos * DEG_PER_UNIT:.1f} deg")
                print(f"voltage (V): {bus.read_voltage(args.id)}")
                print(f"temp (C)   : {bus.read_temperature(args.id)}")
            elif args.cmd == "move":
                if args.speed is not None:
                    t = bus.move_at_speed(args.id, args.position, args.speed)
                    print(f"ID {args.id} -> {args.position} at {args.speed} deg/s ({t} ms)")
                else:
                    bus.move(args.id, args.position, args.time)
                    print(f"ID {args.id} -> {args.position} over {args.time} ms")
            elif args.cmd == "move-many":
                targets = {}
                for pair in args.targets:
                    try:
                        sid_str, pos_str = pair.split(":")
                        targets[int(sid_str)] = int(pos_str)
                    except ValueError:
                        print(f"error: bad ID:POS pair {pair!r} (expected e.g. 1:500)",
                              file=sys.stderr)
                        return 1
                bus.move_many(targets, args.time, sync=not args.no_sync)
                pairs = ", ".join(f"{k}->{v}" for k, v in targets.items())
                print(f"moved {pairs} over {args.time} ms")
            elif args.cmd == "move-deg":
                bus.move_deg(args.id, args.degrees, args.time)
                print(f"ID {args.id} -> {args.degrees} deg over {args.time} ms")
            elif args.cmd == "set-id":
                bus.set_id(args.current, args.new)
                ok = bus.ping(args.new)
                print(f"ID {args.current} -> {args.new} "
                      f"({'confirmed' if ok else 'no response at new ID — re-scan'})")
            elif args.cmd == "fault":
                ids = args.ids or bus.scan()
                for sid in ids:
                    issues = bus.health(sid)
                    print(f"ID {sid}: {'; '.join(issues) if issues else 'healthy'}")
            elif args.cmd == "protection":
                vmin, vmax = bus.read_voltage_limits(args.id)
                tmax = bus.read_temp_limit(args.id)
                mask = bus.read_alarm_mask(args.id)
                print(f"voltage limits: {vmin:.2f}-{vmax:.2f} V")
                print(f"temp limit    : {tmax} C")
                print(f"alarm mask    : {mask} ({bus.format_alarm_mask(mask)})")
            elif args.cmd == "set-temp-limit":
                bus.set_temp_limit(args.id, args.temp)
                print(f"ID {args.id} temp limit set to {bus.read_temp_limit(args.id)} C")
            elif args.cmd == "set-voltage-limits":
                bus.set_voltage_limits(args.id, args.min_mv, args.max_mv)
                vmin, vmax = bus.read_voltage_limits(args.id)
                print(f"ID {args.id} voltage limits set to {vmin:.2f}-{vmax:.2f} V")
            elif args.cmd == "set-alarm-mask":
                bus.set_alarm_mask(args.id, args.mask)
                mask = bus.read_alarm_mask(args.id)
                print(f"ID {args.id} alarm mask set to {mask} ({bus.format_alarm_mask(mask)})")
            elif args.cmd == "monitor":
                period = 1.0 / max(args.hz, 0.1)
                hdr = "   ".join(f"id{sid}:[volt pos temp torque]" for sid in args.ids)
                print(f"{'t(s)':>6}  {hdr}   (Ctrl-C to stop)")
                start = time.time()
                try:
                    while True:
                        cells = []
                        for sid in args.ids:
                            try:
                                v = bus.read_voltage(sid)
                                p = bus.read_position(sid)
                                tp = bus.read_temperature(sid)
                                tq = "on" if bus.is_torque_on(sid) else "OFF"
                                cells.append(f"{v:5.2f} {p:4d} {tp:3d}C {tq:>3}")
                            except ServoError:
                                cells.append("  --   --   --  --")
                        print(f"{time.time() - start:6.1f}  " + "   ".join(cells))
                        time.sleep(period)
                except KeyboardInterrupt:
                    print("\nstopped.")
            elif args.cmd == "limits":
                low, high = bus.read_angle_limits(args.id)
                print(f"angle limits: {low}-{high} "
                      f"({low * DEG_PER_UNIT:.1f}-{high * DEG_PER_UNIT:.1f} deg)")
                print(f"angle offset: {bus.read_angle_offset(args.id)}")
            elif args.cmd == "set-limits":
                bus.set_angle_limits(args.id, args.low, args.high)
                print(f"ID {args.id} angle limits set to {args.low}-{args.high}")
            elif args.cmd == "sweep":
                leg = args.time / 1000.0 + 0.05
                for _ in range(args.cycles):
                    bus.move(args.id, args.high, args.time)
                    time.sleep(leg)
                    bus.move(args.id, args.low, args.time)
                    time.sleep(leg)
                print("sweep done")
            elif args.cmd == "torque":
                ids = bus.scan() if args.all else args.ids
                if not ids:
                    print("No servo IDs given. Pass IDs (e.g. `torque 1 2 3`) or --all.")
                    return 1
                for sid in ids:
                    bus.set_torque(sid, not args.off)
                state = "OFF" if args.off else "ON"
                print(f"torque {state} for ID(s): {', '.join(map(str, ids))}")
    except ServoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
