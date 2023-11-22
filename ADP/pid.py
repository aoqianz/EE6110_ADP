import math
import numpy as np

class PID():
    def __init__(self,p,i,d,p2,i2,d2):
        self.p = p
        self.i = i
        self.d = d
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0

        self.p2 = p2
        self.i2 = i2
        self.d2 = d2
        self.integral2 = 0
        self.derivative2 = 0
        self.prev_error2 = 0

    def cal_gain(self, obs):
        angle = obs[2]
        # angle_vel = obs[3]
        self.integral += angle
        self.derivative = angle - self.prev_error
        self.prev_error = angle

        pid = self.p*angle + self.i*self.integral + self.d*self.derivative

        return pid

    def cal_gain2(self, obs):
        angle = obs[2]
        self.integral += angle
        self.derivative = angle - self.prev_error
        self.prev_error = angle

        x = obs[0]
        self.integral2 += x
        self.derivative2 = x - self.prev_error2
        self.prev_error2 = x

        pid = self.p*angle + self.i*self.integral + self.d*self.derivative
        pid2 = self.p2 * x + self.i2 * self.integral2 + self.d2 * self.derivative2

        return pid + pid2

class PID2():
    def __init__(self,p,i,d):
        self.p = p
        self.i = i
        self.d = d
        self.integral = 0
        self.derivative = 0
        self.prev_error = 0

    def cal_gain2(self, obs):
        angle = 0-obs[0]
        # print(angle)
        angle_vel = -obs[1]
        self.integral += angle
        self.derivative = angle - self.prev_error
        self.prev_error = angle

        pid = self.p*angle + self.i*self.integral + self.d*angle_vel

        return pid