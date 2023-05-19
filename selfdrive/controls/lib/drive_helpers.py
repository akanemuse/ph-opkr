import math
from cereal import car
from common.numpy_fast import clip, interp
from common.realtime import DT_MDL
from common.conversions import Conversions as CV
from selfdrive.modeld.constants import T_IDXS

from common.params import Params
from decimal import Decimal

# from chanhojung's idea, parameterized by opkr
if Params().get("DesiredCurvatureLimit", encoding="utf8") is not None:
  DESIRED_CURVATURE_LIMIT = float(Decimal(Params().get("DesiredCurvatureLimit", encoding="utf8")) * Decimal('0.01'))
else:
  DESIRED_CURVATURE_LIMIT = DT_MDL

# kph
V_CRUISE_MAX = 128.748 # 80 mph
V_CRUISE_MIN = 30
V_CRUISE_DELTA = 10
V_CRUISE_ENABLE_MIN = 30
LAT_MPC_N = 16
LON_MPC_N = 32
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0

# EU guidelines
MAX_LATERAL_JERK = 5.0

CRUISE_LONG_PRESS = 50
CRUISE_NEAREST_FUNC = {
  car.CarState.ButtonEvent.Type.accelCruise: math.ceil,
  car.CarState.ButtonEvent.Type.decelCruise: math.floor,
}
CRUISE_INTERVAL_SIGN = {
  car.CarState.ButtonEvent.Type.accelCruise: +1,
  car.CarState.ButtonEvent.Type.decelCruise: -1,
}


class MPC_COST_LAT:
  PATH = 1.0
  HEADING = 1.0
  STEER_RATE = 1.0


def apply_deadzone(error, deadzone):
  if error > deadzone:
    error -= deadzone
  elif error < - deadzone:
    error += deadzone
  else:
    error = 0.
  return error


def rate_limit(new_value, last_value, dw_step, up_step):
  return clip(new_value, last_value + dw_step, last_value + up_step)


def update_v_cruise(v_cruise_kph, buttonEvents, button_timers, enabled, metric):
  # handle button presses. TODO: this should be in state_control, but a decelCruise press
  # would have the effect of both enabling and changing speed is checked after the state transition
  if not enabled:
    return v_cruise_kph

  button_type = None

  # convert to mph for processing
  current_mph = round(v_cruise_kph * 0.621371)

  # grab buttons with legacy long press code we dont actually use anymore
  for b in buttonEvents:
    if b.type.raw in button_timers and not b.pressed:
      if button_timers[b.type.raw] > CRUISE_LONG_PRESS:
        return v_cruise_kph # end long press
      button_type = b.type.raw
      break
  else:
    for k in button_timers.keys():
      if button_timers[k] and button_timers[k] % CRUISE_LONG_PRESS == 0:
        button_type = k
        break

  # was a button pressed?
  if button_type:
    if button_type == car.CarState.ButtonEvent.Type.accelCruise:
      if current_mph >= 70: # make smaller changes up when at high speeds
        current_mph += 2
      else:
        current_mph += 4
    elif button_type == car.CarState.ButtonEvent.Type.decelCruise:
      if current_mph > 70:
        current_mph = 70 # hop right down to 70 when pressing slow above 70
      else:
        current_mph -= 4
    
    # apply limits
    if current_mph > 80:
      current_mph = 80
    elif current_mph < 26:
      current_mph = 26

    v_cruise_kph = current_mph * 1.60934 # back to kph

  return v_cruise_kph


def initialize_v_cruise(v_ego, buttonEvents, v_cruise_last):
  for b in buttonEvents:
    # 250kph or above probably means we never had a set speed
    if b.type == car.CarState.ButtonEvent.Type.accelCruise and v_cruise_last < 250:
      return v_cruise_last

  return int(round(clip(v_ego * CV.MS_TO_KPH, V_CRUISE_ENABLE_MIN, V_CRUISE_MAX)))


def get_lag_adjusted_curvature(CP, v_ego, psis, curvatures, curvature_rates):
  if len(psis) != CONTROL_N:
    psis = [0.0]*CONTROL_N
    curvatures = [0.0]*CONTROL_N
    curvature_rates = [0.0]*CONTROL_N

  # TODO this needs more thought, use .2s extra for now to estimate other delays
  delay = max(0.01, CP.steerActuatorDelay)
  current_curvature = curvatures[0]
  psi = interp(delay, T_IDXS[:CONTROL_N], psis)
  desired_curvature_rate = curvature_rates[0]

  # MPC can plan to turn the wheel and turn back before t_delay. This means
  # in high delay cases some corrections never even get commanded. So just use
  # psi to calculate a simple linearization of desired curvature
  curvature_diff_from_psi = psi / (max(v_ego, 1e-1) * delay) - current_curvature
  desired_curvature = current_curvature + 2 * curvature_diff_from_psi

  v_ego = max(v_ego, 0.1)
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego**2)
  safe_desired_curvature_rate = clip(desired_curvature_rate,
                                          -max_curvature_rate,
                                          max_curvature_rate)
  safe_desired_curvature = clip(desired_curvature,
                                     current_curvature - max_curvature_rate * DESIRED_CURVATURE_LIMIT,
                                     current_curvature + max_curvature_rate * DESIRED_CURVATURE_LIMIT)

  return safe_desired_curvature, safe_desired_curvature_rate
