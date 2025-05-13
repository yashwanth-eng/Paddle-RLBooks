import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

#cliff walking env
class environment:
    def __init__(self, nRow, nCol, nA):

        self.nRow = nRow
        self.nCol = nCol

        
        self.nS = nRow * nCol  # Number of states
        self.nA = nA  # Number of actions
        self.actions =[0, 1, 2, 3] # Actions: 0=left, 1=down, 2=right, 3=up
        self.V = np.zeros(self.nS)  # Value function
        self.Q = np.zeros((self.nS, self.nA)) # Q-value function
        self.state = 0  # Initial state

        self.grid_reset()

        

        self.render()


    def grid_reset(self):

        self.grid  = np.zeros((self.nRow, self.nCol))  # Initialize the grid
     
        self.grid[0, 1:(self.nCol - 1)] = 1
        # lets make bottom row except for first and last column as cliff

        self.grid[0,0] = 3 # start
        self.grid[0, self.nCol - 1] = 2 # goal



    def state_to_grid(self, state):
        row = state // self.nCol
        col = state % self.nCol
        return row, col
        

        

    def reset(self):
        self.state = 0  # Reset to initial state
        self.grid_reset()
        self.render()
        return self.state

    def step(self, action):

        # if state is top edge or left edge or right edge, up , left, right action should not be taken

        #top edge
        if ((self.nRow - 1) * self.nCol) <= self.state < self.nRow * self.nCol:
            if action == 3:
                return self.state, -100, True
        #left edge
        if self.state % self.nCol == 0:
            if action == 0:
                return self.state, -100, True
        #right edge
        if (self.state + 1) % self.nCol == 0:
            if action == 2:
                return self.state, -100, True
        #bottom edge
        if self.state < self.nCol:
            if action == 1:
                return self.state, -100, True
            


        # Define the transition and reward logic here
        if action == 0:
            self.state -= 1
        elif action == 1:
            self.state += self.nCol
        elif action == 2:
            self.state += 1
        elif action == 3:
            self.state -= self.nCol


        self.update_plot(self.state)

        # if the state is in the cliff, return -100
        if self.state in range(1, self.nCol - 1):
            return self.state, -100, True
        
        if self.state == self.nCol - 1:
            # Reached the goal state
            return self.state, 100, True

        return self.state, -1, False  # Return next state, reward, and done flag
    

    def choose_action(env):
        # Define the action space
        # Randomly choose an action
        chosen_action = np.random.choice(4)
        
        return chosen_action


    def epsilon_greedy(env, state, epsilon=0.1):
        """
        Epsilon-greedy policy for action selection.

        Args:
            env: The environment object.
            state: The current state.
            Q: The Q-table (a dictionary or 2D array).
            epsilon: The probability of choosing a random action.

        Returns:
            The selected action.
        """
        if np.random.rand() < epsilon:
            # Choose a random action
            action = np.random.choice(env.actions)
        else:
            # Choose the action with the highest Q-value for the current state
            action = np.argmax(env.Q[state])

        return action


    def get_policy(env):
        policy = np.zeros(env.nS, dtype=int)  # Initialize policy array
        for state in range(env.nS):
            action = np.argmax(env.Q[state])
            policy[state] = action

        return policy
    
    def render(self):

        fig, self.ax = plt.subplots()

        color_map = ['green','red', 'blue', 'yellow']

        # Use a custom colormap with green, blue, and orange
        cmap = mcolors.ListedColormap(color_map)
        bounds = [0, 1, 2, 3, 4]  # Boundaries for each color (0 -> green, 1 -> red, 2 -> blue, 3 -> orange)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Display the grid as a color matrix
        cax = self.ax.matshow(self.grid, cmap=cmap, norm=norm, interpolation='nearest')

        # Optionally, add a colorbar to show the color mapping
        cbar = fig.colorbar(cax, ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['green','red', 'blue', 'yellow'])

        # Set gridlines for better visibility
        self.ax.set_xticks(np.arange(0, self.nCol, 1))
        self.ax.set_yticks(np.arange(0, self.nRow, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xticks(np.arange(-.5, self.nCol, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.nRow, 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        plt.show()

    def update_plot(self, state):
        """
        Updates the plot to highlight the given state.

        Args:
            state: The state to highlight on the grid.
        """
        # Convert the state to grid coordinates
        row, col = self.state_to_grid(state)

        # Highlight the current state with yellow (value 4)
        self.grid[row, col] = 4

        # Update the plot
        self.ax.clear()  # Clear the previous plot
        color_map = ['green', 'red', 'blue', 'yellow']
        cmap = mcolors.ListedColormap(color_map)
        bounds = [0, 1, 2, 3, 4]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Display the updated grid
        cax = self.ax.matshow(self.grid, cmap=cmap, norm=norm, interpolation='nearest')


        # Set gridlines for better visibility
        self.ax.set_xticks(np.arange(0, self.nCol, 1))
        self.ax.set_yticks(np.arange(0, self.nRow, 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_xticks(np.arange(-.5, self.nCol, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.nRow, 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        # Redraw the plot
        plt.pause(0.01)



#monte carlo with value funcation
def monte_carlo(env, num_episodes=1000, gamma=0.9):

    for episode in range(num_episodes):
        # Generate an episode
        env.reset()
        done = False
        rewards = []
        states_visited = []
        states_visited.append(env.state)

        while not done:
            action = env.epsilon_greedy(env.state)
            next_state, reward, done = env.step(action)

            env.Q[env.state, action] += reward

            print(next_state, reward, done, action)
            rewards.append(reward)
            states_visited.append(next_state)
        

        # Calculate the cumulative reward for each state
        G = 0
        cumulative_rewards = []
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            cumulative_rewards.insert(0, G)
            state = states_visited[t]
            # Update the value function
            env.V[state] += (G - env.V[state]) / (episode + 1)

    return env.get_policy(), env.V

Cliff_walker = environment(4, 12, 4)
policy, V = monte_carlo(Cliff_walker, num_episodes=1000, gamma=0.9)
print("Final Policy:", policy)
print("Final Value Function:", V.reshape(4, 12))