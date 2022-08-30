from game import Direction, SnakeGame, Point
from collections import deque

MAX_MEMORY = 100_100
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def _init_(self):
        self.n_games = 0
        self.epsilon = 0 #randomness
        self.gamma = 0 #discount rate
        self.memory = deque(maxlen=MAX_MEMORY) #when shit gets full it pops element

def train():
    play_scores = []
    plot_mean_scores = []
    total_scores = []
    

if __name__ == "__main__":
    train()