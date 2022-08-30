from game import Direction, SnakeGame, Point
from collections import deque
import numpy as np

MAX_MEMORY = 100_100
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def _init_(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #when shit gets full it pops element

    def get_state(self, game):
        head = game.snake[0]
        #compute surrounding points
        point_l = Point(head.x-20, head.y)
        point_r = Point(head.x+20, head.y)
        point_u = Point(head.x, head.y-20)
        point_d = Point(head.x, head.y+20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #danger_straight
            (dir_r and game.is_collision(point_r))or
            (dir_l and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #danger_right
            (dir_r and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_u))or
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l)),

            #danger_left
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d))or
            (dir_u and game.is_collision(point_l))or
            (dir_d and game.is_collision(point_r)),

            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #food_location
            game.food.x < game.head.x, #food is on left
            game.food.x > game.head.x, #food is on right
            game.food.y > game.head.y, #food is up
            game.food.y < game.head.y #food is down
        ]

        return np.array(state, dtype=int)

    def train_short_memory(self, state, action, reward, next_state, done):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memory(self):
        pass

def train():
    play_scores = []
    plot_mean_scores = []
    total_scores = []
    record = 0
    
    agent = Agent()
    game = SnakeGame()

    while True:
        #get old state
        state_old = agent.get_state(game)

        #get move
        final_move = agent.get_action(state_old)

        reward, done, score =  game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory on one move
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            #if game_over
            game.reset()
            agent.n_games +=1
            agent.train_long_memory()

            if score > record :
                record = score
                #agent.model.save()
            
            print("Game", agent.n_games, "Score", score, "Record", record)

            #plotting here




if __name__ == "__main__":
    train()