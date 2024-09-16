import pygame
import numpy as np
import random
import math

# Initialize pygame
pygame.init()

# Define constants for the environment
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
CAR_WIDTH, CAR_HEIGHT = 40, 20
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Discretization step for state space
GRID_SIZE = 40  # Increased grid size for better learning

# Speed boundaries
MIN_SPEED = 20
MAX_SPEED = 50

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Car RL Racing Simulation")

# Clock for controlling frame rate
clock = pygame.time.Clock()


# Define the car class
class Car:
    def __init__(self):
        self.start_position()

    def start_position(self):
        # Reset car to the starting position at the middle of the track
        self.x = 200  # Middle x position
        self.y = 200  # Middle y position
        self.angle = 1  # Initial angle
        self.speed = 10  # Start with minimum speed (20 units)
        self.max_speed = 22  # Maximum speed is 50
        self.acceleration = 1  # Increase acceleration for larger speed range
        self.brake = 2  # Stronger brake to handle speed range
        self.turn_angle = 45  # Increase turn angle to promote turning
        self.waypoints_passed = set()  # To track passed waypoints

    def update(self, action):
        if action == 1:  # Turn left
            self.angle += self.turn_angle
        elif action == 2:  # Turn right
            self.angle -= self.turn_angle
        elif action == 3:  # Brake
            self.speed = max(self.speed - self.brake, MIN_SPEED)  # Decrease speed but not below MIN_SPEED

        # Accelerate when moving (unless braking)
        if action != 3:
            self.speed = min(self.speed + self.acceleration, self.max_speed)  # Cap speed at MAX_SPEED

        # Move the car straight or with the current angle
        rad_angle = math.radians(self.angle)
        self.x += self.speed * math.sin(rad_angle)
        self.y -= self.speed * math.cos(rad_angle)

        # Keep car within bounds
        self.x = max(0, min(SCREEN_WIDTH - CAR_WIDTH, self.x))
        self.y = max(0, min(SCREEN_HEIGHT - CAR_HEIGHT, self.y))

    def draw(self):
        car_rect = pygame.Rect(self.x, self.y, CAR_WIDTH, CAR_HEIGHT)
        pygame.draw.rect(screen, RED, car_rect)


# Define the environment (track with waypoints and checkpoints)
class RaceTrack:
    def __init__(self):
        # Define a narrower outer and inner track
        self.outer_track = [
            [(200, 150), (600, 150)],  # Top outer line
            [(600, 150), (600, 450)],  # Right outer line
            [(600, 450), (200, 450)],  # Bottom outer line
            [(200, 450), (200, 150)]  # Left outer line
        ]

        self.inner_track = [
            [(250, 200), (550, 200)],  # Top inner line (closer to the outer line)
            [(550, 200), (550, 400)],  # Right inner line
            [(550, 400), (250, 400)],  # Bottom inner line
            [(250, 400), (250, 200)]  # Left inner line
        ]

        # Define checkpoints/waypoints along the track
        self.waypoints = [
            (250, 175),  # Start
            (550, 175),  # Top right corner
            (550, 425),  # Bottom right corner
            (250, 425),  # Bottom left corner
        ]

    def draw(self):
        # Draw the outer track
        for segment in self.outer_track:
            pygame.draw.line(screen, WHITE, segment[0], segment[1], 5)

        # Draw the inner track
        for segment in self.inner_track:
            pygame.draw.line(screen, WHITE, segment[0], segment[1], 5)

        # Draw the waypoints
        for wp in self.waypoints:
            pygame.draw.circle(screen, BLUE, wp, 10)

    def is_off_track(self, car):
        car_rect = pygame.Rect(car.x, car.y, CAR_WIDTH, CAR_HEIGHT)
        return not self.is_inside_track(car_rect)

    def is_inside_track(self, car_rect):
        # This is a simple collision check with bounding boxes for outer and inner track
        outer_box = pygame.Rect(200, 150, 400, 300)
        inner_box = pygame.Rect(250, 200, 300, 200)
        return outer_box.contains(car_rect) and not inner_box.contains(car_rect)

    def check_waypoint(self, car):
        """Check if the car has passed a waypoint and return the waypoint index."""
        for i, wp in enumerate(self.waypoints):
            if math.hypot(car.x - wp[0], car.y - wp[1]) < 15:
                return i
        return None


# Q-learning agent with action masking for penalties
class QLearningAgent:
    def __init__(self):
        # Discretized Q-table (smaller, to fit the grid size)
        self.q_table = np.zeros(((SCREEN_WIDTH // GRID_SIZE), (SCREEN_HEIGHT // GRID_SIZE), 4))  # 4 actions
        self.alpha = 0.1  # Higher learning rate to encourage faster learning
        self.gamma = 0.9  # Discount factor
        self.epsilon = 1.0  # Start with high exploration rate

        # Store actions that led to penalties in specific states
        self.penalty_actions = {}

    def discretize(self, x, y):
        """Discretize the car's position to fit in the Q-table."""
        return int(x // GRID_SIZE), int(y // GRID_SIZE)

    def choose_action(self, car):
        # Discretize the car's current position
        x, y = self.discretize(car.x, car.y)

        # Epsilon-greedy strategy
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Random action (exploration)

        # Mask out penalty actions in the current state
        if (x, y) in self.penalty_actions:
            valid_actions = [a for a in range(4) if a not in self.penalty_actions[(x, y)]]
            if valid_actions:
                return random.choice(valid_actions)  # Pick from non-penalized actions
            else:
                return np.argmax(self.q_table[x, y])  # Fall back to best action

        return np.argmax(self.q_table[x, y])  # Best action (exploitation)

    def learn(self, car, action, reward, next_car):
        # Discretize the current and next car's positions
        x, y = self.discretize(car.x, car.y)
        next_x, next_y = self.discretize(next_car.x, next_car.y)

        # Best next action from the next state
        best_next_action = np.argmax(self.q_table[next_x, next_y])

        # Q-learning formula
        self.q_table[x, y, action] += self.alpha * (
                reward + self.gamma * self.q_table[next_x, next_y, best_next_action] - self.q_table[x, y, action]
        )

        # If the reward is strongly negative (penalty), mark this action as penalized
        if reward < 0:
            if (x, y) not in self.penalty_actions:
                self.penalty_actions[(x, y)] = set()
            self.penalty_actions[(x, y)].add(action)  # Mark this action as penalized

        # Gradually reduce epsilon to reduce exploration over time
        self.epsilon = max(0.01, self.epsilon * 0.995)  # Decay epsilon faster to switch to exploitation


import os

# Create a directory to store the frames
if not os.path.exists('frames'):
    os.makedirs('frames')

frame_count = 0  # Variable to keep track of frame numbers
# Simulation loop
def main():
    global frame_count
    car = Car()
    track = RaceTrack()
    agent = QLearningAgent()

    running = True
    while running:
        screen.fill(BLACK)
        track.draw()
        car.draw()

        # Save the current frame
        pygame.image.save(screen, f"frames/frame_{frame_count:05d}.png")
        frame_count += 1
        # Agent chooses an action
        action = agent.choose_action(car)

        # Store car's previous state
        prev_car = Car()
        prev_car.x, prev_car.y, prev_car.angle, prev_car.speed = car.x, car.y, car.angle, car.speed

        # Update car's state based on action
        car.update(action)

        # Check if the car is off track and give reward
        if track.is_off_track(car):
            reward = -50  # Reduced penalty for going off track to encourage learning
            car.start_position()  # Reset the car to starting position
        else:
            reward = 1  # Reward for staying on track

        # Add extra reward for turning
        if action == 1 or action == 2:
            reward += 10  # Reward for making turns

        # Check if the car has passed a waypoint
        waypoint_idx = track.check_waypoint(car)
        if waypoint_idx is not None and waypoint_idx not in car.waypoints_passed:
            car.waypoints_passed.add(waypoint_idx)
            reward += 20  # Reward for passing a waypoint

        # Learn from the experience
        agent.learn(prev_car, action, reward, car)

        # Check for user input to close the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update the display
        pygame.display.update()

        # Control frame rate
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
