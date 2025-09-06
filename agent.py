import torch # this library is how we will build our neural network
import random # this will allow us to intorduce randomnesss into our agent and game
import numpy as np # this is for numerical opperations

from game import SnakeGameAI, Direction, Point # here we are importing the SnakeGameAI class from the game file
from collections import deque # this is a ds where we store memories
from helper import plot # here we are importing the 'plot' function from the 'helper file

from model import Linear_QNet, QTrainer # here we are importing the Linear_QNet and QTrainer classes from the model file

MAX_MEMORY = 100_000 # this is the maximum number of memories we will store
BATCH_SIZE = 1000 # this is the number of memories we will use to train out model
LR = 0.002 # this is our learning rate


device = torch.device("cpu") # we train the model on the CPU due to stability issues with the MPS backend
print('Using CPU only (MPS disabled due to stability issues).')

# this is the agent class
# this is our snake which will be learning how to play the game
class Agent:

    # this is the constructor method and is how we create the agent object
    def __init__(self):
        self.n_games = 0 # this varaible is the number of games the agent has played during training
        self.epsilon = 0 # this is the randomness factor, the higher the value of epsilon the more random the actions of the agent will be
        self.gamma = 0.9 # discount rate, this number must always be smaller than 1 and is used to balance immediate and future rewards
        self.memory = deque(maxlen=MAX_MEMORY) # this is the memory ds where we will store our experiences
        self.model = Linear_QNet(11, 128, 3).to(device) # this is the actual neural network model, it has 11 input nodes, 128 hidden nodes and 3 output nodes

        # this is the trainer object which will be used to train the model
        # we can see that the QTrainer class takes the model, learning rate, the gamma value, and the device the training should take place on
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma ,device = device)
        
        # try loading an existing model if it already exists
        try:
            # this line will look for a 'model.pth' file in the current directory and then load it
            self.model.load(device=device)

            # if one is found, we use this print statemnt to tell the user
            print('Loaded saved model. Continuing model training...')
        
        # if there is no model found then we catch the exception and then print a message to the user
        except:
            print('No saved model found. Starting new model trining...')

    # this method is how we get the current state of the game
    # the state is represented as a list of 11 values which are either 0 or 1
    # these values represent things like the direction the snake is moving, where the food is
    def get_state(self, game):

        # the head variable represents the head of the snake
        head = game.snake[0]

        # we use the following four variables to check if there is any danger around the snakes head
        # each point is used to represent a possible position the snake could move to
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x - 20, head.y - 20)
        point_d = Point(head.x - 20, head.y + 20)

        # these boolean variables are used to check which direction the snake is currently moving in
        # for example if 'dir_l' is true, then the snake is moving to the left
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # this is the state list which contains all 11 values
        # each value is either a 0 or 1 which represents true or false
        state = [
            # danger straight
            # this section is the snake looking for danger infront of itself
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            # danger right
            # this section is the snake looking for danger to its right side
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            # danger left
            # this section is the snake looking for danger to its left side
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            # move direction
            # this is where we encode which direction the snake is currently moving in
            # because the snake can move in one direction at a time, only one of these values will be true
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            # this is where we encode the location of the food relative to the snake
            # for example if the food is the left of the snake then the first value will be true
            # it is possible for more than one of these values to be true because the food location can be diagonal to the snakes head
            game.food.x < game.head.x, # is food to the left
            game.food.x > game.head.x, # is food to the right
            game.food.y < game.head.y, # is food above
            game.food.y > game.head.y # is food below
        ]

        # here we convert the state list to a numpy array and then return it
        # we convert to a numpy array because it groups the arrays into a single block in memory which makes it faster to process
        return np.array(state, dtype = int)

    # this method is used to store the agents past experiences
    # the agent will use these experiences to learn from later on
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # this method is used to train the agent with a batch of memories
    # the agent will randomly select a batch of memories and then use them to train the model
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # returns a list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)


    # this method is used to train the agent with a single memory
    # this is used to train the agent after every step taken
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # below is how we calculate the epsilon decay
        # after each game the value will decrease and allow the agent to make less random moves by allowing it to explot what it has learned in its neural network
        # this means that at the start of the training the agent will make a lot of random
        self.epsilon = 80 - self.n_games

        # this is the move array which will be returned by this method
        # the three values represent the directions the snake can move in
        # for example, [1,0,0] = straight
        final_move = [0,0,0]

        # this if statement is what decides if the agent will make a random move or if it will exploit what its learned in its neural network
        # the higher the value of epsilon, the more likely the agent will make a random move
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1

        # if the agent chooses not to make a random move, then it will use its neural network to predict the best move
        else:
            state0 = torch.tensor(np.array(state), dtype = torch.float).to(device)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(num_games = None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    #agent.model.save("initial_model.pth")

    max_games = num_games if num_games else float("inf")

    while agent.n_games < max_games:
        # get the old state of the current state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform the move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train the long memory, plot results
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # this if statement is used to check if the model has beaten its high score
            # if the model has beaten its high score, it will overwrite the exisiting model and save the new model to model.pth
            if score > record:
                record = score
                agent.model.save()

            # this is the print statment what will print stats of the training session to the terminl.
            print(f"Game {agent.n_games} | Score: {score} | Record: {record} | Epsilon: {agent.epsilon} | Average Score: {round((total_score / agent.n_games), 2)}")

            plot_scores.append(score)

            total_score += score

            mean_score = total_score / agent.n_games

            plot_mean_scores.append(mean_score)

            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':

    # we need to imprt the sys library to be allow the program to read terminal inputs
    import sys
    
    # default is that the agent runs forever
    num_games = None

    # if the user provides a number in the terminal, the program will set the agent to play the selected number of games
    # and then quit the program once its done
    if len(sys.argv) > 1: # this line checks if there is more than one argument in the terminal input

        try: # we try to convert the terminal input to an integer value
            num_games = int(sys.argv[1]) # if we can do this then we assign the value to the num_games variable
        except ValueError: # this except statement will catch if the user enters anything that cannot be converted into an integer value, like a word or a letter
            print("Invalid number of games. Running forever now.")

    # here we call the train function and pass in the num_games variable to tell the agent how many games to play
    train(num_games = num_games)
