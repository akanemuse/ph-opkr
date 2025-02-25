from cereal import car
from common.conversions import Conversions as CV
from opendbc.can.parser import CANParser
from opendbc.can.can_define import CANDefine
from selfdrive.car.interfaces import CarStateBase
from selfdrive.car.chrysler.values import DBC, STEER_THRESHOLD


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    can_define = CANDefine(DBC[CP.carFingerprint]["pt"])
    self.shifter_values = can_define.dv["GEAR"]["PRNDL"]

  def update(self, cp, cp_cam):

    ret = car.CarState.new_message()

    self.frame = int(cp.vl["EPS_STATUS"]["COUNTER"])

    ret.doorOpen = any([cp.vl["DOORS"]["DOOR_OPEN_FL"],
                        cp.vl["DOORS"]["DOOR_OPEN_FR"],
                        cp.vl["DOORS"]["DOOR_OPEN_RL"],
                        cp.vl["DOORS"]["DOOR_OPEN_RR"]])
    ret.seatbeltUnlatched = cp.vl["SEATBELT_STATUS"]["SEATBELT_DRIVER_UNLATCHED"] == 1

    ret.brakePressed = cp.vl["BRAKE_2"]["BRAKE_PRESSED_2"] == 5  # human-only
    ret.brake = 0
    ret.gas = cp.vl["ACCEL_GAS_134"]["ACCEL_134"]
    ret.gasPressed = ret.gas > 1e-5

    ret.espDisabled = (cp.vl["TRACTION_BUTTON"]["TRACTION_OFF"] == 1)

    ret.wheelSpeeds = self.get_wheel_speeds(
      cp.vl["WHEEL_SPEEDS"]["WHEEL_SPEED_FL"],
      cp.vl["WHEEL_SPEEDS"]["WHEEL_SPEED_FR"],
      cp.vl["WHEEL_SPEEDS"]["WHEEL_SPEED_RL"],
      cp.vl["WHEEL_SPEEDS"]["WHEEL_SPEED_RR"],
      unit=1,
    )
    ret.vEgoRaw = (cp.vl["SPEED_1"]["SPEED_LEFT"] + cp.vl["SPEED_1"]["SPEED_RIGHT"]) / 2.
    ret.vEgo, ret.aEgo = self.update_speed_kf(ret.vEgoRaw)
    ret.standstill = not ret.vEgoRaw > 0.001

    ret.leftBlinker = cp.vl["STEERING_LEVERS"]["TURN_SIGNALS"] == 1
    ret.rightBlinker = cp.vl["STEERING_LEVERS"]["TURN_SIGNALS"] == 2
    ret.steeringAngleDeg = cp.vl["STEERING"]["STEER_ANGLE"]
    ret.steeringRateDeg = cp.vl["STEERING"]["STEERING_RATE"]
    ret.gearShifter = self.parse_gear_shifter(self.shifter_values.get(cp.vl["GEAR"]["PRNDL"], None))

    ret.cruiseState.enabled = cp.vl["ACC_2"]["ACC_STATUS_2"] == 7  # ACC is green.
    ret.cruiseState.available = ret.cruiseState.enabled  # FIXME: for now same as enabled
    ret.cruiseState.speed = cp.vl["DASHBOARD"]["ACC_SPEED_CONFIG_KPH"] * CV.KPH_TO_MS
    # CRUISE_STATE is a three bit msg, 0 is off, 1 and 2 are Non-ACC mode, 3 and 4 are ACC mode, find if there are other states too
    #ret.cruiseState.nonAdaptive = cp.vl["DASHBOARD"]["CRUISE_STATE"] in (1, 2)

    ret.steeringTorque = cp.vl["EPS_STATUS"]["TORQUE_DRIVER"]
    ret.steeringTorqueEps = cp.vl["EPS_STATUS"]["TORQUE_MOTOR"]
    ret.steeringPressed = abs(ret.steeringTorque) > STEER_THRESHOLD
    steer_state = cp.vl["EPS_STATUS"]["LKAS_STATE"]
    ret.steerFaultPermanent = steer_state == 4 or (steer_state == 0 and ret.vEgo > self.CP.minSteerSpeed)

    ret.genericToggle = bool(cp.vl["STEERING_LEVERS"]["HIGH_BEAM_FLASH"])

    if self.CP.enableBsm:
      ret.leftBlindspot = cp.vl["BLIND_SPOT_WARNINGS"]["BLIND_SPOT_LEFT"] == 1
      ret.rightBlindspot = cp.vl["BLIND_SPOT_WARNINGS"]["BLIND_SPOT_RIGHT"] == 1

    self.lkas_counter = cp_cam.vl["LKAS_COMMAND"]["COUNTER"]
    self.lkas_car_model = cp_cam.vl["LKAS_HUD"]["CAR_MODEL"]
    self.lkas_status_ok = cp_cam.vl["LKAS_HEARTBIT"]["LKAS_STATUS_OK"]

    return ret

  @staticmethod
  def get_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("PRNDL", "GEAR"),
      ("DOOR_OPEN_FL", "DOORS"),
      ("DOOR_OPEN_FR", "DOORS"),
      ("DOOR_OPEN_RL", "DOORS"),
      ("DOOR_OPEN_RR", "DOORS"),
      ("BRAKE_PRESSED_2", "BRAKE_2"),
      ("ACCEL_134", "ACCEL_GAS_134"),
      ("SPEED_LEFT", "SPEED_1"),
      ("SPEED_RIGHT", "SPEED_1"),
      ("WHEEL_SPEED_FL", "WHEEL_SPEEDS"),
      ("WHEEL_SPEED_RR", "WHEEL_SPEEDS"),
      ("WHEEL_SPEED_RL", "WHEEL_SPEEDS"),
      ("WHEEL_SPEED_FR", "WHEEL_SPEEDS"),
      ("STEER_ANGLE", "STEERING"),
      ("STEERING_RATE", "STEERING"),
      ("TURN_SIGNALS", "STEERING_LEVERS"),
      ("ACC_STATUS_2", "ACC_2"),
      ("HIGH_BEAM_FLASH", "STEERING_LEVERS"),
      ("ACC_SPEED_CONFIG_KPH", "DASHBOARD"),
      ("CRUISE_STATE", "DASHBOARD"),
      ("TORQUE_DRIVER", "EPS_STATUS"),
      ("TORQUE_MOTOR", "EPS_STATUS"),
      ("LKAS_STATE", "EPS_STATUS"),
      ("COUNTER", "EPS_STATUS",),
      ("TRACTION_OFF", "TRACTION_BUTTON"),
      ("SEATBELT_DRIVER_UNLATCHED", "SEATBELT_STATUS"),
    ]

    checks = [
      # sig_address, frequency
      ("BRAKE_2", 50),
      ("EPS_STATUS", 100),
      ("SPEED_1", 100),
      ("WHEEL_SPEEDS", 50),
      ("STEERING", 100),
      ("ACC_2", 50),
      ("GEAR", 50),
      ("ACCEL_GAS_134", 50),
      ("DASHBOARD", 15),
      ("STEERING_LEVERS", 10),
      ("SEATBELT_STATUS", 2),
      ("DOORS", 1),
      ("TRACTION_BUTTON", 1),
    ]

    if CP.enableBsm:
      signals += [
        ("BLIND_SPOT_RIGHT", "BLIND_SPOT_WARNINGS"),
        ("BLIND_SPOT_LEFT", "BLIND_SPOT_WARNINGS"),
      ]
      checks.append(("BLIND_SPOT_WARNINGS", 2))

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 0)

  @staticmethod
  def get_cam_can_parser(CP):
    signals = [
      # sig_name, sig_address
      ("COUNTER", "LKAS_COMMAND"),
      ("CAR_MODEL", "LKAS_HUD"),
      ("LKAS_STATUS_OK", "LKAS_HEARTBIT")
    ]
    checks = [
      ("LKAS_COMMAND", 100),
      ("LKAS_HEARTBIT", 10),
      ("LKAS_HUD", 4),
    ]

    return CANParser(DBC[CP.carFingerprint]["pt"], signals, checks, 2)
