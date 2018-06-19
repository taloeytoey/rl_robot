import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS


# import os, sys
# current_path = os.getcwd()
# sys.path.insert(0, os.path.join( current_path, "pymunk-4.0.0" ) )

import pymunk

from pymunk.vec2d import Vec2d
# from pymunk.pygame_util import draw
from pymunk.pygame_util import DrawOptions as draw
import pymunk.pygame_util

# PyGame init
width = 1000
height = 700
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()

# Turn off alpha since we don't use it.
screen.set_alpha(None)

# Showing sensors and redrawing slows things down.
show_sensors = True
draw_screen = True


class GameState:
    def __init__(self):
        # Global-ish.
        self.crashed = False

        # Physics stuff.
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)

        # Create the car.
        self.create_car(100, 100, 0.9)

        # Record steps.
        self.num_steps = 0

        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.
            s.group = 1
            s.collision_type = 1
            s.color = (255, 0, 0)
        self.space.add(static)

        # Create some obstacles, semi-randomly.
        # We'll create three and they'll move around to prevent over-fitting.
        self.obstacles = []

        self.obstacles.append(self.create_obstacle(450, 400, 80, (204, 153, 255)))
        self.obstacles.append(self.create_obstacle(800, 250, 80, (255, 153, 255)))
        self.obstacles.append(self.create_obstacle(700, 500, 80, (153, 153, 255)))
        self.obstacles.append(self.create_obstacle(200, 200, 80, (125, 225, 235)))
        self.obstacles.append(self.create_obstacle(200, 500, 80, (102, 204, 255)))

        # Create cat.
        #self.create_cat()

    def create_obstacle(self, x, y, r, color):
        # c_body = pymunk.Body(pymunk.inf, pymunk.inf)
        c_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        c_shape = pymunk.Circle(c_body, r)
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = color
        self.space.add(c_body, c_shape)
        return c_body


    #def create_cat(self):
    #    inertia = pymunk.moment_for_circle(1, 0.01, 30, (0, 0))
    #    self.cat_body = pymunk.Body(1, inertia)
    #    self.cat_body.position = 50, height - 100
    #    self.cat_shape = pymunk.Circle(self.cat_body, 30)
    #    self.cat_shape.color = THECOLORS["orange"]
    #    self.cat_shape.elasticity = 1.0
    #    self.cat_shape.angle = 0.5
    #    direction = Vec2d(1, 0).rotated(self.cat_body.angle)
    #    self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0.01, 30, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 30)
        self.car_shape.color = (135, 147, 154)
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        # self.car_body.apply_impulse(driving_direction)
        self.car_body.apply_impulse_at_world_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        if action == 0:  # Turn left.
            self.car_body.angle -= .5
        elif action == 1:  # Turn right.
            self.car_body.angle += .5

        # Move obstacles.
        #if self.num_steps % 100 == 0:
        #    self.move_obstacles()

        # Move cat.
        #if self.num_steps % 5 == 0:
        #    self.move_cat()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 50 * driving_direction

        # Update the screen and stuff.
        screen.fill((255, 255, 255))
        # draw(screen, self.space)
        options = pymunk.pygame_util.DrawOptions(screen)
        self.space.debug_draw(options)

        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        # Get the current location and the readings there.
        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        state = np.array(readings)

        # Set the reward.
        # Car crashed when any reading == 1
        if self.car_is_crashed(readings):
            self.crashed = True
            reward = float(-50)
            self.recover_from_crash(driving_direction)
        else:
            # Higher readings are better, so return the sum.
            # reward = int(self.sum_readings(readings))
            reward = float(int(self.sum_readings(readings) / 10))
        self.num_steps += 1

        return reward, state

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(20, 100)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    #def move_cat(self):
    #    speed = random.randint(10, 100)
    #    self.cat_body.angle -= random.randint(-1, 1)
    #    direction = Vec2d(1, 0).rotated(self.cat_body.angle)
    #    self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        """
        We hit something, so recover.
        """
        while self.crashed:
            # Go backwards.
            self.car_body.velocity = -50 * driving_direction
            self.crashed = False
            for i in range(5):
                self.car_body.angle += .2  # Turn a little.
                screen.fill((255, 0, 0))  # Red is scary!
                # draw(screen, self.space)
                options = pymunk.pygame_util.DrawOptions(screen)
                self.space.debug_draw(options)

                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

    def sum_readings(self, readings):
        """Sum the number of non-zero readings."""
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        """
        Instead of using a grid of boolean(ish) sensors, sonar readings
        simply return N "distance" readings, one for each sonar
        we're simulating. The distance is a count of the first non-zero
        reading starting at the object. For instance, if the fifth sensor
        in a sonar "arm" is non-zero, then that arm returns a distance of 5.
        """
        # Make our arms.
        arm_left1 = self.make_sonar_arm(x, y)
        arm_middle = arm_left1
        arm_right1 = arm_left1




        # Rotate them and get readings.
        readings.append(float(self.get_arm_distance(arm_left1, x, y, angle, 0.75)))
        readings.append(float(self.get_arm_distance(arm_middle, x, y, angle, 0)))
        readings.append(float(self.get_arm_distance(arm_right1, x, y, angle, -0.75)))

        if show_sensors:
            pygame.display.update()

        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        # Used to count the distance.
        i = 0

        # Look at each point and see if we've hit something.
        for point in arm:
            i += 1

            # Move the point to the right spot.
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            # Check if we've hit something. Return the current i (distance)
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i  # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (0, 0, 0), (rotated_p), 2)

        # Return the distance for the arm.
        return i

    def make_sonar_arm(self, x, y):
        spread = 10  # Default spread.
        distance = 20  # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we'll rotate it about the
        # center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))

        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == (255, 255, 255):
            return 0
        else:
            return 1

if __name__ == "__main__":
    game_state = GameState()
    while True:
        #game_state.frame_step((random.randint(0, 2)))
        game_state.frame_step(1)
