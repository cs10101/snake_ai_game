# these are all the librearies we need to import for the game
import pygame # this is used to create the game window and handle graphics
import random # this is used to generate random positions for the food
from enum import Enum # this is used to define the direction of the snake
from collections import namedtuple # this is used to create a simple class to represent points on the screen
import numpy as np

# initialize pygame and fonts 
pygame.init()

# we are using the 'ariel.ttf' file to render text in the game
# you can also use pygame.font.SysFont if you don't have the ttf file
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

# this is a class to represent the directions the snake can move in
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# Point is a simple class to represent a point on the screen
# it has two attributes: x and y
Point = namedtuple('Point', 'x, y')

# rgb colors
# these are the colors we will use in the game
# each variable takes three values to represent RGB colours
WHITE = (255, 255, 255)
RED   = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# BLOCK_SIZE is the size of each block in the game
# the snake and food will be made up of these blocks
# the higher the value, the bigger the blocks
BLOCK_SIZE = 20

# SPEED is the speed of the game
# higher values make the game faster
SPEED = 60

# this is the main class for the game
# the class allows us to create a game object
# and contains all the methods to run the game
class SnakeGameAI:
    
    # this is the constructor method
    # its role is to initialize the game
    # it sets up the display, initializes the game state
    # and places the first food item on the screen
    def __init__(self, w=640, h=480):

        # set width and height of the game window
        self.w = w
        self.h = h

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Version 1.0')

        # this line keeps track of how long the game has been running
        self.clock = pygame.time.Clock()
        self.reset()
        
        

    # this is the reset function which will be used to reset the game if the user want to reset the game
    def reset(self):
        # init game state
        # this line basically sets the snake to alwasys start moving to the right
        self.direction = Direction.RIGHT
        
        # the head is always the front of the snake
        # this line sets the initial position of the snake in the middle of the screen
        self.head = Point(self.w/2, self.h/2)

        # the actual body of the snake is represented as a list of points in a 2D space
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

        
    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt = None):

        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        score_text = font.render("Score: " + str(self.score), True, WHITE)

        # Display the number of games played in a session
        

        self.display.blit(score_text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):
        # [straigh, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change

        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn

        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn

        self.direction = new_dir
        

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            
