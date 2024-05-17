from gymnasium import spaces
from default_env import DefaultEnv
import numpy as np

from msfs_rt_env import RealTimeFlightSimInterface


class GliderEnv(DefaultEnv):

    def __init__(self):

        super().__init__()

        heading = spaces.Box(low=0.0, high=360.0, shape=(1,), dtype=np.float64)
        bank = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float64)
        pitch = spaces.Box(low=-180.0, high=180.0, shape=(1,), dtype=np.float64)
        turnCoord = spaces.Box(low=-127.0, high=127, shape=(1,), dtype=np.int64)
        airspeed = spaces.Box(low=-10.0, high=np.inf, shape=(1,), dtype=np.float64)
        vario = spaces.Box(low=-np.inf, high=200.0, shape=(1,), dtype=np.float64)
        alt = spaces.Box(low=0.0, high=13000.0, shape=(1,), dtype=np.float64)

        self.observationSpace = spaces.Tuple(
            (heading, bank, pitch, turnCoord, airspeed, vario, alt)
        )

        self.actionSpace = spaces.Box(
            low=-1.0000, high=1.0000, shape=(3,), dtype=np.float64
        )

        self.defaultAction = np.array([0, 0, 0], dtype=np.float64)

        self.simVars = [
            dict(name="Plane Heading Degrees True", units="degrees"),
            dict(name="Plane Bank Degrees", units="degrees"),
            dict(name="Plane Pitch Degrees", units="degrees"),
            dict(name="Turn Coordinator Ball"),
            dict(name="Airspeed Indicated", units="knots"),
            dict(name="Variometer Rate", units="feet per second"),
            dict(name="Indicated Altitude", units="feet"),
        ]

        self.startPos = [53.562170301218835, -2.872581622824867]
        self.startHead = 0

    def get_observation_space(self) -> spaces.Tuple:

        return self.observationSpace

    def get_action_space(self) -> spaces.Box:

        return self.actionSpace

    def get_default_action(self) -> np.array:

        return self.defaultAction

    def observation(self):

        sv = self.sc.get_simdata(self.simVars)

        heading = np.array([sv["Plane Heading Degrees True"]], dtype=np.float64)
        bank = np.array([sv["Plane Bank Degrees"]], dtype=np.float64)
        pitch = np.array([sv["Plane Pitch Degrees"]], dtype=np.float64)
        turnCoord = np.array([sv["Turn Coordinator Ball"]], dtype=np.int64)
        airspeed = np.array([sv["Airspeed Indicated"]], dtype=np.float64)
        vario = np.array([sv["Variometer Rate"]], dtype=np.float64)
        alt = np.array([sv["Indicated Altitude"]], dtype=np.float64)

        return [heading, bank, pitch, turnCoord, airspeed, vario, alt]

    def control(self, controlInput):
        """
        Transmits the chosen action to MSFS
        controlInput: Array [Aileron, Elevator, Rudder]
        """
        self.sc.set_simdatum("Aileron Position", controlInput[0] * 16000)
        self.sc.set_simdatum("Elevator Position", controlInput[1] * 16000)
        self.sc.set_simdatum("Rudder Pedal Position", controlInput[2] * 16000)

    def reset_env(self, seed=None, options=None):
        pos = self.startPos
        head = self.startHead
        self.sc.set_simdatum("Plane Latitude", pos[0], units="degrees")
        self.sc.set_simdatum("Plane Longitude", pos[1], units="degrees")
        self.sc.set_simdatum("Plane Heading Degrees True", head)
        self.sc.set_simdatum("Plane Bank Degrees", 0)
        self.sc.set_simdatum("Plane Pitch Degrees", 0)
        self.sc.set_simdatum("Airspeed True", 75, units="knots")
        self.sc.set_simdatum("Plane Alt Above Ground", 1500, units="feet")
        self.sc.set_simdatum("Vertical Speed", 0)
        self.sc.set_simdatum("Aileron Position", 0)
        self.sc.set_simdatum("Elevator Position", 0)
        self.sc.set_simdatum("Rudder Pedal Position", 0)

    def reward(self, observation):

        turnCoord = observation[3][0]
        turnCoordReward = 0.5 * (1 - abs(turnCoord / 75))
        if turnCoordReward <= 0.1:
            turnCoordReward = 0.1
        # print(f"Turn Coordinator: {turnCoordReward}")

        vario = observation[5][0]
        if vario > 0:
            varioRew = (1 / (((-0.25 * vario) + 1) ** 2)) + 1
        else:
            varioRew = 0

        # print(f"vario: {varioRew}")

        alt = observation[6][0]
        altDiscount = alt / 1500
        # print(f"alt: {altDiscount}")

        reward = altDiscount * (varioRew + turnCoordReward)
        print(
            f"Reward: {reward}    | Turn{turnCoordReward} Vario:{varioRew} Alt:{altDiscount}"
        )
        return reward

    def terminate(self):

        altAboveGround = self.sc.get_simdatum("Plane Alt Above Ground", units="feet")

        terminated = False
        if altAboveGround < 150:
            terminated = True

        return terminated


class GliderInterface(RealTimeFlightSimInterface):

    def __init__(self):
        super().__init__(GliderEnv())
