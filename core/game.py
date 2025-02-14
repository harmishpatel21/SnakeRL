import pygame
import numpy as np 
from collections import namedtuple 
from config import Config, Direction


# Constants
pygame.init()
font = pygame.font.Font('C:/Windows/Fonts/arial.ttf', 25)
Point = namedtuple('Point', 'x,y')

# # Colors
# WHITE = (255, 255, 255)
# RED = (200, 0, 0)
# BLUE1 = (0, 0, 255)
# BLUE2 = (0, 100, 255)
# BLACK = (0, 0, 0)

# BLOCK_SIZE = 20
# SPEED = 10 # Lower is faster

# class Direction(Enum):
#     RIGHT = 0
#     LEFT = 1
#     UP = 2
#     DOWN = 3 

class SnakeGame:
    def __init__(self):
        self.w = Config.BLOCK_SIZE * Config.GRID_WIDTH
        self.h = Config.BLOCK_SIZE * Config.GRID_HEIGHT
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        ## Initial game state
        self.direction = Direction.RIGHT
        start_x = (Config.GRID_WIDTH // 2) * Config.BLOCK_SIZE
        start_y = (Config.GRID_HEIGHT // 2) * Config.BLOCK_SIZE
        self.head = Point(start_x, start_y)
        self.snake = [self.head,
                      Point(start_x - Config.BLOCK_SIZE, start_y),
                      Point(start_x - (2*Config.BLOCK_SIZE),start_y)]
        self.score = 0
        self.food = None 
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = np.random.randint(0, Config.GRID_WIDTH) * Config.BLOCK_SIZE
        y = np.random.randint(0, Config.GRID_HEIGHT) * Config.BLOCK_SIZE
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
        
        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head)

        # 3. Check if game over
        reward = 0
        game_over = False 

        if self.is_collision() or self.frame_iteration > Config.MAX_FRAME_ITERATIONS*len(self.snake):
            game_over = True 
            reward = Config.COLLISION_PENALTY
            return reward, game_over, self.score
        
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = Config.FOOD_REWARD
            self._place_food()
        else:
            reward = Config.MOVE_PENALTY
            self.snake.pop()
        
        # 5. update UI and clock
        self._update_ui()
        self.clock.tick(Config.SPEED)

        # 6. Return game state
        return reward, game_over, self.score
    
    def is_collision(self, pt=None):
        pt = pt or self.head 
        # hits boundary
        if (pt.x > Config.GRID_WIDTH * Config.BLOCK_SIZE or pt.x < 0 or
            pt.y > Config.GRID_HEIGHT * Config.BLOCK_SIZE or pt.y < 0):
            return True 
        # hits itself
        if pt in self.snake[1:]:
            return True
        return False 
    
    def _update_ui(self):
        self.display.fill(Config.BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, Config.BLUE1,
                             pygame.Rect(pt.x, pt.y, Config.BLOCK_SIZE, Config.BLOCK_SIZE))
            pygame.draw.rect(self.display, Config.BLUE2, 
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        
        pygame.draw.rect(self.display, Config.RED, 
                         pygame.Rect(self.food.x, self.food.y, Config.BLOCK_SIZE, Config.BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, Config.WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()

    def _move(self, action):
        # Action: [straight, right turn, left turn]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]): # No change
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]): # Right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += Config.BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= Config.BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += Config.BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= Config.BLOCK_SIZE
            
        self.head = Point(x, y)

    def get_state(self):
        # Simplified state representation for AI
        head = self.head
        point_l = Point(head.x - Config.BLOCK_SIZE, head.y)
        point_r = Point(head.x + Config.BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - Config.BLOCK_SIZE)
        point_d = Point(head.x, head.y + Config.BLOCK_SIZE)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or
            (dir_l and self.is_collision(point_l)) or
            (dir_u and self.is_collision(point_u)) or
            (dir_d and self.is_collision(point_d)),
            
            # Danger right
            (dir_u and self.is_collision(point_r)) or
            (dir_d and self.is_collision(point_l)) or
            (dir_l and self.is_collision(point_u)) or
            (dir_r and self.is_collision(point_d)),
            
            # Danger left
            (dir_d and self.is_collision(point_r)) or
            (dir_u and self.is_collision(point_l)) or
            (dir_r and self.is_collision(point_u)) or
            (dir_l and self.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location
            self.food.x < self.head.x,  # Food left
            self.food.x > self.head.x,  # Food right
            self.food.y < self.head.y,  # Food up
            self.food.y > self.head.y   # Food down
        ]
        
        return np.array(state, dtype=int)