# AISnake
Using Reinforcement Learning to play Snake Game.

<img src="https://github.com/supreetshm947/AISnake/blob/main/Demo.gif" width="320" height="240" />

Training Pseudocode:
```
state = get_state(game) #state is an 11 bit vector
action = get_move(state) #predict action or move based on current state of the game
reward, game_over, score = game.play_step(action) #play the move and calculate and return the reward of current move, game_over flag and score until the current iteration 
new_state = get_state(game)
model_train(state, action, new_state, reward, next_state, game_over)
```
