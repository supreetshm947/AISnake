from game import Direction, SnakeGame, Point
from collections import deque
import numpy as np
import random
import torch
import helper
from model import Linear_QNet, QTrainer
MAX_MEMORY = 100_100
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #when shit gets full it pops element
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

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
        self.trainer.train_step(state, action, reward, next_state, done)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if(len(self.memory) < BATCH_SIZE):
            mini_samples = self.memory
        else:
            mini_samples = random.sample(self.memory, BATCH_SIZE)
        
        states, actions, rewards, next_states, dones = zip(*mini_samples)

        self.trainer.train_step(states, actions, rewards, next_states, dones)
    
    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            index = random.randint(0,2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            index = torch.argmax(self.model(state0)).item()
        
        move[index] = 1
        return move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_scores = 0
    record = 0
    
    agent = Agent()
    game = SnakeGame()

    while True:
        #get old state
        state_old = agent.get_state(game)

        #get move - either gives a random move or predicted move - exploration exploitation tradeoff
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
            plot_scores.append(score)
            total_scores += score
            mean_score = total_scores/agent.n_games
            plot_mean_scores.append(mean_score)
            helper.plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()