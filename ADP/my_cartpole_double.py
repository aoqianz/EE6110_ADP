"""
Original file: cartpole.py
Initial Modifications by Fredrik Gustafsson
Modified to match latest gym environment and rendering by Aoqian Zhang
"""

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c
"""
import math
from typing import Optional, Union
import logging
import math
import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.error import DependencyNotInstalled
from gymnasium.envs.classic_control import utils
import numpy as np
from scipy.integrate import ode

sin = np.sin
cos = np.cos

class CartpoleDouble(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'render_fps': 50
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.g = -9.81  # gravity constant
        self.m0 = 1.0  # mass of cart
        self.m1 = 0.5  # mass of pole 1
        self.m2 = 0.5  # mass of pole 2
        self.L1 = 1  # length of pole 1
        self.L2 = 1  # length of pole 2
        self.l1 = self.L1 / 2  # distance from pivot point to center of mass
        self.l2 = self.L2 / 2  # distance from pivot point to center of mass
        self.I1 = self.m1 * (self.L1 ^ 2) / 12  # moment of inertia of pole 1 w.r.t its center of mass
        self.I2 = self.m2 * (self.L2 ^ 2) / 12  # moment of inertia of pole 2 w.r.t its center of mass
        self.tau = 0.02  # seconds between state updates
        self.counter = 0
        self.force_mag = 20.0
        # Angle at which to fail the episode
        # self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # # (never fail the episode based on the angle)
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 3

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.theta_threshold_radians,
            self.theta_threshold_radians,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max])

        self.render_mode = render_mode
        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        # Just need to initialize the relevant attributes
        # self.configure()
        self.render_mode = render_mode

        self.screen_width = 800
        self.screen_height = 600
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    # def configure(self, display=None):
    #     self.display = display

    def step(self, action):
        # assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        u = float(max(min(self.force_mag, action), -self.force_mag))
        self.counter += 1

        # (state_dot = func(state))
        def func(t, state, u):
            x, theta, phi, x_dot, theta_dot, phi_dot = state
            # x = state.item(0)
            # theta = self.angle_normalize(state.item(1))
            # phi = self.angle_normalize(state.item(2))
            # x_dot = state.item(3)
            # theta_dot = state.item(4)
            # phi_dot = state.item(5)
            state = np.matrix(
                [[x], [theta], [phi], [x_dot], [theta_dot], [phi_dot]])  # this is needed for some weird reason

            d1 = self.m0 + self.m1 + self.m2
            d2 = self.m1 * self.l1 + self.m2 * self.L1
            d3 = self.m2 * self.l2
            d4 = self.m1 * pow(self.l1, 2) + self.m2 * pow(self.L1, 2) + self.I1
            d5 = self.m2 * self.L1 * self.l2
            d6 = self.m2 * pow(self.l2, 2) + self.I2
            f1 = (self.m1 * self.l1 + self.m2 * self.L1) * self.g
            f2 = self.m2 * self.l2 * self.g

            D = np.matrix([[d1, d2 * cos(theta), d3 * cos(phi)],
                           [d2 * cos(theta), d4, d5 * cos(theta - phi)],
                           [d3 * cos(phi), d5 * cos(theta - phi), d6]])

            C = np.matrix([[0, -d2 * sin(theta) * theta_dot, -d3 * sin(phi) * phi_dot],
                           [0, 0, d5 * sin(theta - phi) * phi_dot],
                           [0, -d5 * sin(theta - phi) * theta_dot, 0]])

            G = np.matrix([[0], [-f1 * sin(theta)], [-f2 * sin(phi)]])

            H = np.matrix([[1], [0], [0]])

            I = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            O_3_3 = np.matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            O_3_1 = np.matrix([[0], [0], [0]])

            A_tilde = np.bmat([[O_3_3, I], [O_3_3, -np.linalg.inv(D) * C]])
            B_tilde = np.bmat([[O_3_1], [np.linalg.inv(D) * H]])
            W = np.bmat([[O_3_1], [np.linalg.inv(D) * G]])
            state_dot = A_tilde * state + B_tilde * u + W
            return state_dot

        solver = ode(func)
        solver.set_integrator("dop853")  # (Runge-Kutta)
        solver.set_f_params(u)
        t0 = 0
        state0 = state
        solver.set_initial_value(state0, t0)
        solver.integrate(self.tau)
        state = solver.y

        # state_dot = func(0, state, u)
        # state = state + self.tau*state_dot
        x = state.item(0)
        theta = self.angle_normalize(state.item(1))
        phi = self.angle_normalize(state.item(2))
        x_dot = state.item(3)
        theta_dot = state.item(4)
        phi_dot = state.item(5)
        self.state = (x, theta, phi, x_dot,theta_dot,phi_dot)
        done = x < -self.x_threshold or x > self.x_threshold or theta > 90 * 2 * np.pi / 360 or theta < -90 * 2 * np.pi / 360
        done = bool(done)

        xout = bool(x < -self.x_threshold+0.1 or x > self.x_threshold-0.1)
        truncated = self.counter >= 3000
        # cost = 10 * self.angle_normalize(theta) + \
        #        10 * self.angle_normalize(phi)

        reward = cos(theta) + cos(phi) - 0.05*(x ** 2) - 0.001 * (theta_dot ** 2) - 0.001 * (phi_dot ** 2) - 0.001 * (x_dot ** 2) - 100*int(xout)
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, done, truncated, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.01, 0.01  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(6,))
        self.state[1] = np.random.uniform(-0.1, 0.1)
        self.counter = 0
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (1 * self.L1)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 150  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[1])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        # Draw the second pole (new code)
        # Adjust the rotation for the second pole as needed. This example assumes `x[3]` contains the rotation for the second pole.
        top_left_tip = pole_coords[1]  # This is the top left point after rotation
        top_right_tip = pole_coords[2]  # This is the top right point after rotation
        tip_of_first_pole = ((top_left_tip[0] + top_right_tip[0])/2, top_left_tip[1])
        second_pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            # Rotate the second pole around the tip of the first pole
            # assuming x[3] is the angle of the second pole relative to the first pole
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            # Offset the position to the tip of the first pole
            coord = (coord[0] + tip_of_first_pole[0], coord[1] + tip_of_first_pole[1])
            second_pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, second_pole_coords, (102, 204, 255))
        gfxdraw.filled_polygon(self.surf, second_pole_coords, (102, 204, 255))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.aacircle(
            self.surf,
            int(tip_of_first_pole[0]),
            int(tip_of_first_pole[1]),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(tip_of_first_pole[0]),
            int(tip_of_first_pole[1]),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )


    def angle_normalize(self, angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi