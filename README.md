# AI Snake — Deep Q-Learning

An AI agent that learns to play Snake using Deep Q-Learning, built with
Python, PyTorch and Pygame.

The agent learns through interaction with the game rather than being given
a predefined strategy. During training, it balances exploration of random
actions with exploitation of actions predicted by its neural network.

## Features

- Snake environment built using Pygame
- Deep Q-Learning agent implemented with PyTorch
- Experience replay for training from previous game states
- Epsilon-greedy exploration with decreasing randomness during training
- Automatic saving and loading of trained models
- Live tracking of score and average score during training
- Option to train for a fixed number of games or indefinitely

## How It Works

The agent observes the current state of the game using 11 input values
representing information such as:

- danger around the snake
- the snake's current direction
- the position of the food relative to the snake

The neural network receives these 11 values and produces three outputs
representing the possible actions:

- continue straight
- turn right
- turn left

The agent initially explores the environment by making more random
decisions. As training progresses, the exploration rate decreases and
the agent increasingly uses its neural network to select actions.

After each action, the agent receives a reward based on the outcome.
Experiences are stored in memory and later sampled in batches using
experience replay.

## Technologies

- Python 3.12
- PyTorch
- Pygame
- NumPy
- Matplotlib

## Installation

Clone the repository:

```bash
git clone <YOUR-REPOSITORY-URL>
cd snake-pygame

## Setting Up the Virtual Environment

It is recommended to run the project inside a Python virtual environment. This keeps the project's dependencies separate from other Python installations on your computer.

This project has been tested using **Python 3.12**.

### 1. Create the Virtual Environment

From inside the project directory, run:

```bash
python3 -m venv .venv
```

This will create a `.venv` directory containing the project's local Python environment.

### 2. Activate the Virtual Environment

#### macOS / Linux

```bash
source .venv/bin/activate
```

#### Windows

```powershell
.venv\Scripts\activate
```

Once activated, you should see `(.venv)` at the beginning of your terminal prompt.

### 3. Install the Dependencies

With the virtual environment activated, install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

The main dependencies used by the project are:

- PyTorch
- Pygame
- NumPy
- Matplotlib

---

## Running the Project

The AI can either train indefinitely or for a specified number of games.

### Train Indefinitely

Run:

```bash
python agent.py
```

The agent will continue playing and training until the program is manually stopped.

### Train for a Fixed Number of Games

You can provide the desired number of games as a command-line argument.

For example:

```bash
python agent.py 500
```

This will run the training process for 500 games before stopping.

You can replace `500` with any desired number of games.

---

## Saved Models

During training, the neural network is automatically saved so that its progress can be preserved between sessions.

Saved models are stored in:

```text
models/
```

If a saved model is available when the program starts, it will automatically be loaded and training will continue from the existing model.

If no saved model is found, the program will start training a new model from scratch.

The `models/` directory is generated locally and is excluded from Git using `.gitignore`.

---

## Training Results

During training, Matplotlib is used to display the performance of the agent.

The graph tracks:

- The score achieved during each game
- The average score achieved over the training session

When a finite training session finishes, the final training graph is saved as:

```text
outputs/snake_training_final.png
```

The `outputs/` directory is automatically created when required and is excluded from Git because it contains generated training results.

---

## Project Structure

```text
snake-pygame/
├── agent.py
├── game.py
├── helper.py
├── model.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/          # Generated during training
└── outputs/         # Generated training results
```

### `agent.py`

Contains the reinforcement learning agent and the main training loop. It is responsible for:

- Reading the current game state
- Selecting actions
- Managing experience replay memory
- Controlling epsilon-based exploration
- Training the neural network
- Tracking training performance

### `game.py`

Contains the Snake game environment built using Pygame.

It handles:

- Snake movement
- Food generation
- Collision detection
- Game state
- Scoring
- Rendering the game window

### `model.py`

Contains the PyTorch neural network and Q-learning training logic.

The neural network uses:

- **11 input neurons** representing the current game state
- **128 hidden neurons**
- **3 output neurons** representing the possible actions

The three possible actions are:

1. Continue straight
2. Turn right
3. Turn left

### `helper.py`

Contains the Matplotlib functionality used to visualise the agent's performance during training.

---

## How the Agent Learns

The project uses **Deep Q-Learning**, a reinforcement learning technique where the agent learns which actions are valuable through interaction with the game.

For every move, the agent observes the current state of the game and chooses one of three possible actions.

The agent receives feedback in the form of rewards and uses this information to improve its neural network.

### Experience Replay

Previous experiences are stored in memory.

During training, the agent randomly selects batches of these experiences and trains on them again. This allows the agent to learn from previous situations rather than relying only on its most recent move.

### Exploration and Exploitation

The agent uses an epsilon-greedy strategy.

At the beginning of training, the agent is more likely to make random moves. This allows it to explore different actions and discover which behaviours produce better results.

As training progresses, the amount of randomness decreases and the agent increasingly relies on its neural network to choose actions.

---

## Technologies Used

- **Python 3.12**
- **PyTorch** — neural network and Deep Q-Learning
- **Pygame** — Snake game environment and graphics
- **NumPy** — numerical operations and game-state representation
- **Matplotlib** — training performance visualisation

---

## Future Improvements

Potential future improvements to the project include:

- Creating a standalone desktop application
- Adding a mode for watching a trained AI without continuing training
- Adding configuration options for training parameters
- Additional training statistics and visualisations
- Automated testing
- Experimenting with different neural network architectures
- Experimenting with reinforcement learning hyperparameters

---

## Author

**Christopher Scott**

BA Artificial Intelligence & Data Science  
Glasgow Caledonian University

