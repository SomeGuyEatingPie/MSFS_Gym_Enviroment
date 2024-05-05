from simconnect import SimConnect
import numpy as np
import math

class MSFS():

    def __init__(self):
        
        f = False
        while f == False:
            try:
                self.sc = SimConnect(name='RTGym')
                f =True
            except:
                print("Error, cannot connect to simconnect. Press ENTER to try again...")
                input()

        self.simsimvars = [
                dict(name="Plane Heading Degrees True", units="degrees"),
                dict(name="Plane Bank Degrees", units= "degrees"),
                dict(name="Plane Pitch Degrees", units="degrees"),
                dict(name="Turn Coordinator Ball"),
                dict(name="Airspeed Indicated", units="knots"),
                dict(name="Variometer Rate", units="feet per second"),
                dict(name="Plane Alt Above Ground", units= "feet"),
                dict(name="Indicated Altitude", units="feet"),
                dict(name="Plane Latitude", units="degrees"),
                dict(name="Plane Longitude", units="degrees")
            ]
            
        

        self.defaultAction = np.array([0, 0, 0], dtype='int')

        self.startPos = [53.562170301218835, -2.872581622824867]
        self.startHead = 0

    
    def observation(self):

        
        sv = self.sc.get_simdata(self.simsimvars)

        heading = np.array([sv["Plane Heading Degrees True"]], dtype=float)
        bank = np.array([sv["Plane Bank Degrees"]], dtype=float)
        pitch = np.array([sv["Plane Pitch Degrees"]], dtype=float)
        turnCoord = np.array([sv["Turn Coordinator Ball"]], dtype=int)
        airspeed = np.array([sv["Airspeed Indicated"]], dtype=float)
        vario = np.array([sv["Variometer Rate"]], dtype=float)
        trueAlt = np.array([sv["Plane Alt Above Ground"]], dtype=float)
        alt = np.array([sv["Indicated Altitude"]], dtype=float)
        pos = np.array([sv["Plane Latitude"],sv["Plane Longitude"]], dtype=float)


        return [heading, bank, pitch, turnCoord, airspeed, vario, trueAlt, alt, pos]
    

    def control(self, controlInput):
        """
        Transmits the chosen action to MSFS
        controlInput: Array [Aileron, Elevator, Rudder]
        """
        self.sc.set_simdatum("Aileron Position", controlInput[0]*16000)
        self.sc.set_simdatum("Elevator Position", controlInput[1]*16000)
        self.sc.set_simdatum("Rudder Pedal Position", controlInput[2]*16000)

    
    def reset_env(self, seed=None, options=None):
        pos = self.startPos
        head = self.startHead
        self.sc.set_simdatum("Plane Latitude", pos[0], units = "degrees")
        self.sc.set_simdatum("Plane Longitude", pos[1], units = "degrees")
        self.sc.set_simdatum("Plane Heading Degrees True", head)
        self.sc.set_simdatum("Plane Bank Degrees", 0)
        self.sc.set_simdatum("Plane Pitch Degrees", 0)
        self.sc.set_simdatum("Airspeed True", 75)
        self.sc.set_simdatum("Plane Alt Above Ground", 1500)

    
    def reward(self, observation):
        
        #1500ft starting alt
        alt = observation[4][0]
        altRew = 1.5* math.sqrt(alt-140)-50
        #75kts starting airspeed
            #50kts Stall speed
        speed = observation[4][0]

        vario = observation[5]
        speedRew = 2*math.log(speed - 50) -2.795

        reward = speedRew*altRew*vario

        return reward
    
    def end_episode(self, observation):

        return observation[6] < 150
    
def test_simconnect():

    env = MSFS()
    env.reset_env()
    observation = env.observation()
    print("--------Observation---------")
    print(observation)
    reward = env.reward(observation)
    print("--------Reward--------")
    print(reward)
    env.end_episode(observation)
    control = (0.5, 0.5,0.5)
    env.control(control)

#test_simconnect()