
import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from gymnasium.experimental.vector import VectorEnv
from gymnasium.vector.utils import batch_space


class CartPoleReal(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.frictioncart = 0.01  # Added cart friction
        self.frictionpole = 0.01  # Added pole friction
        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 3
        self.steps = 0
        self.Nc = 1
        self.actuator_noise = 0.05 #5% actuator noise
        self.sensor_noise = 0.05 #5% sensor noise
        self.max_steps = 3000
        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        # err_msg = f"{action!r} ({type(action)}) invalid"
        # assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        action = action * (1+self.make_actuator_noise())
        force = float(max(min(self.force_mag, action), -self.force_mag))
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (-force - self.polemass_length * theta_dot * theta_dot * (sintheta + self.frictioncart * np.sign(self.Nc*x_dot)*costheta)) / self.total_mass \
               + self.frictioncart * self.gravity * np.sign(self.Nc*x_dot)
        thetaacc = (self.gravity * sintheta + costheta * temp - self.frictionpole * theta_dot / self.polemass_length) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta * (costheta - self.frictioncart * np.sign(self.Nc*x_dot)) / self.total_mass))

        Nc_new = self.total_mass * self.gravity - self.polemass_length*(thetaacc*sintheta + theta_dot*theta_dot*costheta)
        if np.sign(Nc_new) != np.sign(self.Nc):
            self.Nc = Nc_new
            temp = (-force - self.polemass_length * theta_dot * theta_dot * (
                        sintheta + self.frictioncart * np.sign(self.Nc*x_dot) * costheta)) / self.total_mass \
                   + self.frictioncart * self.gravity * np.sign(self.Nc*x_dot)
            thetaacc = (
                                   self.gravity * sintheta + costheta * temp - self.frictionpole * theta_dot / self.polemass_length) / \
                       (self.length * (4.0 / 3.0 - self.masspole * costheta * (
                                   costheta - self.frictioncart * np.sign(self.Nc*x_dot)) / self.total_mass))
        self.Nc = Nc_new

        xacc = (force + self.polemass_length * (theta_dot * theta_dot * sintheta - thetaacc*costheta) - self.frictioncart * np.sign(self.Nc*x_dot)) / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        theta = self.angle_normalize(theta)
        self.state = (x, x_dot, theta, theta_dot)

        x_noise = float(x*(1+self.make_sensor_noise()))
        theta_noise = float(theta*(1+self.make_sensor_noise()))
        state_noise = (x_noise, x_dot, theta_noise, theta_dot)

        x_out = bool(
            x < -self.x_threshold
            or x > self.x_threshold
        )
        terminated = bool(
            x < -self.x_threshold-0.1
            or x > self.x_threshold+0.1
        )
        self.steps += 1
        truncated = self.steps >= self.max_steps

        if abs(theta_noise) < 0.4:
            reward = 1.0 - 0.2 * abs(theta_noise) - 0.01 * abs(x_noise) - 0.1 * abs(theta_dot) - 0.2 * (x_dot) - 100 * int(x_out)
        else:
            R_angle = math.cos(theta_noise)
            R_vel = -0.01 * (theta_dot ** 2) - 0.01 * (x_dot ** 2)
            R_x = -0.05 * (x_noise ** 2)
            R_term = -100 * int(x_out)
            # if abs(theta_noise) < 0.01 and abs(theta_dot) < 0.001 and abs(x_dot) < 0.01:
            #     R_bonus = 10
            # else:
            #     R_bonus = 0
            reward = R_angle + R_vel + R_x + R_term
        # reward = np.cos(theta) - 0.001 * theta_dot**2 - 0.0001 * force**2 - 100 * int(x_out)
        if self.render_mode == "human":
            self.render()
        return np.array(state_noise, dtype=np.float32), reward, terminated, truncated, {}

    def angle_normalize(self,angle):
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def make_actuator_noise(self):
        return np.random.uniform(low=-self.actuator_noise, high=self.actuator_noise, size=(1,))

    def make_sensor_noise(self):
        return np.random.uniform(low=-self.sensor_noise, high=self.sensor_noise, size=(1,))

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
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        # self.state[0] = self.np_random.uniform(low=-self.x_threshold*0.5, high=self.x_threshold*0.5)
        self.state[2] = 180 * 2 * math.pi / 360
        self.steps_beyond_terminated = None
        self.steps = 0
        self.Nc = 1
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
        polelen = scale * (2 * self.length)
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
        carty = 100  # TOP OF CART
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
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

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

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
