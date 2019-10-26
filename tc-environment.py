import numpy as np
import random

from itertools import groupby
from itertools import product


class TicTacToeEnvironment:
    def __init__(self):
        """initialise the board"""

        # initialise state as an array
        self.state = [np.nan for _ in range(9)]

        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)]

        self.reset()

    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""

        # define possible index collections in each of (horizontal, vertical, diagonal) directions

        # 3 horizontal rows of a 3 X 3 playing board
        horizontal_indices = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        # 3 vertical rows of a 3 X 3 playing board
        vertical_indices = [[0, 3, 6], [1, 4, 7], [2, 5, 8]]

        # 2 diagonal rows : (top left --> bottom right) & (top right --> bottom left)
        diagonal_indices = [[0, 4, 8], [2, 4, 6]]

        # sum across each group of indices should be equal to 15 to win the game
        horizontal_sum = [np.sum(np.array(curr_state)[i]) for i in horizontal_indices]
        vertical_sum = [np.sum(np.array(curr_state)[i]) for i in vertical_indices]
        diagonal_sum = [np.sum(np.array(curr_state)[i]) for i in diagonal_indices]

        horizontal_win = list(filter(lambda x: x == 15, horizontal_sum))
        vertical_win = list(filter(lambda x: x == 15, vertical_sum))
        diagonal_win = list(filter(lambda x: x == 15, diagonal_sum))

        #  game is won if sum across any direction is equal to 15
        if len(horizontal_win) != 0 or len(vertical_win) != 0 or len(diagonal_win) != 0:
            return True
        else:
            return False

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, "Win"

        elif len(self.allowed_positions(curr_state)) == 0:
            return True, "Tie"

        else:
            return False, "Resume"

    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]

    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        # fetch all allowed values used in the game
        used_values = [val for val in curr_state if not np.isnan(val)]

        # RL agent is only allowed to play odd numbers : {1,3,5,7,9}
        # fetch numbers which an agent can still play i.e.
        # odd numbers which have not been played by the agent so far
        agent_values = [
            val
            for val in self.all_possible_numbers
            if val not in used_values and val % 2 != 0
        ]

        # environment is only allowed to play even numbers : {2,4,6,8}
        # fetch numbers which environment can still play i.e.
        # even numbers which have not been played by the environment so far
        env_values = [
            val
            for val in self.all_possible_numbers
            if val not in used_values and val % 2 == 0
        ]

        return (agent_values, env_values)

    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        allowed_positions = self.allowed_positions(curr_state)
        allowed_values = self.allowed_values(curr_state)

        # action space of a given space is the cartesian product of all allowed positions and allowed values
        agent_actions = product(allowed_positions, allowed_values[0])
        env_actions = product(allowed_positions, allowed_values[1])

        return (agent_actions, env_actions)

    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """

        # current new state variable from existing state
        new_state = [i for i in curr_state]

        # update current action
        new_state[curr_action[0]] = curr_action[1]

        return new_state

    def reset(self):
        return self.state
