import numpy as np
# Removed unused import
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

#cliff walking env
class environment:
    def __init__(self, nRow, nCol, nA):

        self.nRow = nRow
        self.nCol = nCol

        self.same_state = False

        
        self.nS = nRow * nCol  # Number of states
        self.nA = nA  # Number of actions
        self.actions =[0, 1, 2, 3] # Actions: 0=left, 1=down, 2=right, 3=up
        self.V = np.zeros(self.nS)  # Value function
        self.Q = np.zeros((self.nS, self.nA)) # Q-value function
        self.state = 0  # Initial state

        self.current_episode_states = []

        fig, self.ax = plt.subplots()

        self.grid_reset()


    def grid_reset(self):

        self.grid  = np.zeros((self.nRow, self.nCol))  # Initialize the grid
        self.current_episode_states = []
     
        self.grid[0, 1:(self.nCol - 1)] = 1
        # lets make bottom row except for first and last column as cliff

        self.grid[0,0] = 3 # start
        self.grid[0, self.nCol - 1] = 2 # goal
        self.render()



    def state_to_grid(self, state):
        row = state // self.nCol
        col = state % self.nCol
        return row, col
    

    def grid_to_state(self, row, col):
        state = row * self.nCol + col
        

        

    def reset(self):
        self.state = 0  # Reset to initial state
        self.grid_reset()
        return self.state
    

    def is_invalid_action(self, action):
        """
        Checks if the given action is invalid based on the current state.

        Args:
            action: The action to check (0=left, 1=down, 2=right, 3=up).

        Returns:
            bool indicating whether the action takes the agent out of bounds of the grid.
        """
        # Bottom edge
        if ((self.nRow - 1) * self.nCol) <= self.state < self.nRow * self.nCol:
            if action == 1:
                return True
        # Left edge
        if self.state % self.nCol == 0:
            if action == 0:
                return -1, -100, True
        # Right edge
        if (self.state + 1) % self.nCol == 0:
            if action == 2:
                return True
        # Top edge
        if self.state < self.nCol:
            if action == 3:
                return True
        return False

    def step(self, action):

        # if state is top edge or left edge or right edge, up , left, right action should not be taken        

        # Check if the action is invalid
        invalid_action = self.is_invalid_action(action)
        if invalid_action:
            # If the action is invalid, return the current state and a penalty
            return self.state, -100, False
        
        current_state = self.state

        row, col = self.state_to_grid(current_state)
        self.update_grid(row, col, 3)

        # Actions: 0=left, 1=down, 2=right, 3=up
        # Define the transition and reward logic here
        if action == 0:
            next_state = self.state - 1
        elif action == 1:
            next_state = self.state + self.nCol
        elif action == 2:
            next_state = self.state + 1
        elif action == 3:
            next_state = self.state - self.nCol


        if next_state in self.current_episode_states:

            if self.same_state == False:
                self.same_state = True
                # If the state is already visited, return the current state and a penalty
                return self.state, -100, False


        self.current_episode_states.append(next_state)

        row, col = self.state_to_grid(next_state)
        self.update_grid(row, col, 4)

        self.state = next_state  # Update the current state

        self.same_state = False

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


    def epsilon_greedy(env, state, epsilon=0.00001):
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

        return int(action)


    def get_policy(env):
        policy = np.zeros(env.nS, dtype=int)  # Initialize policy array
        for state in range(env.nS):
            action = np.argmax(env.Q[state])
            policy[state] = action

        return policy
    
    def render(self):

        self.ax.clear()  # Clear the previous plot
        color_map = ['green', 'red', 'blue', 'yellow', 'orange']
        cmap = mcolors.ListedColormap(color_map)
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Display the updated grid
        cax = self.ax.matshow(self.grid, cmap=cmap, norm=norm, interpolation='nearest')
  

        # Add text annotations to the top of the plot
        self.ax.text(0.5, 1.05, 'Cliff Walking Environment', transform=self.ax.transAxes, 
                 ha='center', va='center', fontsize=12, fontweight='bold')


        # Set gridlines for better visibility
        self.ax.set_xticks(np.arange(0, self.nCol, 1))
        self.ax.set_yticks(np.arange(0, self.nRow, 1))
        self.ax.set_xticks(np.arange(-.5, self.nCol, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.nRow, 1), minor=True)
        self.ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        plt.pause(0.05)  # Pause to update the plot

    def update_grid(self, row, col, value):
        """
        Updates the grid at the specified position and refreshes the plot.

        Args:
            row: The row index to update.
            col: The column index to update.
            value: The new value to set at the specified position.
        """
        # Update the grid
        self.grid[row, col] = value

        # Refresh the plot
        self.ax.clear()  # Clear the previous plot
        color_map = ['green', 'red', 'blue', 'yellow', 'orange']
        cmap = mcolors.ListedColormap(color_map)
        bounds = [0, 1, 2, 3, 4, 5]
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
        plt.pause(0.05)

#monte carlo with value funcation
def monte_carlo(env, num_episodes=1000, gamma=0.9):

    for episode in range(num_episodes):
        print("Episode:", episode)
        # Generate an episode
        env.reset()
        done = False
        rewards = []
        states_visited = []
       

        while not done:

            current_state = env.state
             # Actions: 0=left, 1=down, 2=right, 3=up
            action = env.epsilon_greedy(env.state, epsilon=0.1)
            next_state, reward, done = env.step(action)

            env.Q[current_state, action] += reward

            # print(next_state, reward, done, action)
           
            rewards.append(reward)
            states_visited.append(current_state)
        

        # Calculate the cumulative reward for each state
        G = 0
        cumulative_rewards = []
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            cumulative_rewards.insert(0, G)
            state = states_visited[t]
            # Update the value function
            env.V[state] += (G - env.V[state]) / (num_episodes + 1)

    return env.get_policy(), env.V

Cliff_walker = environment(4, 5, 4)
policy, V = monte_carlo(Cliff_walker, num_episodes=100, gamma=0.9)
print("Final Policy:", policy.reshape(4, 5))
print("Final Value Function:", V.reshape(4, 5))


plt.show()