import logging
import numpy as np
import unittest

from tc_environment import TicTacToeEnvironment

logger = logging.getLogger("__name__")


class TicTacToeEnvironmentTest(unittest.TestCase):
    def setUp(self):
        # is_winning
        self.state_check_total_15 = [3, 1, 8, 4, 5, 2, np.nan, np.nan, 7]
        self.state_check_total_not_15 = [3, 1, 4, 8, np.nan, 2, np.nan, np.nan, 7]

        self.state_agent_win = [3, 7, 1, np.nan, np.nan, np.nan, 2, 4, 6]
        self.state_env_win = [3, 7, 1, 5, np.nan, 8, 5, 4, 6]

        # is_terminal
        self.state_terminal = [1, 2, 5, 4, 6, 3, 7, 8, 9]
        self.state_non_terminal = [
            1,
            3,
            5,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ]

        # allowed_values
        self.state_allowed_values = [1, 3, np.nan, np.nan, 2, 8, np.nan, np.nan, np.nan]
        self.state_no_allowed_values = [1, 2, 5, 4, 6, 3, 7, 8, 9]

        # action_space
        # agent to play : 9
        # env to play : 6
        # possible positions : 6 & 7
        self.state_sample_action_space = [3, 1, 8, 4, 5, 2, np.nan, np.nan, 7]

        # state_transition
        self.state_old_state = [1, 3, np.nan, np.nan, 2, 8, np.nan, np.nan, np.nan]

        self.tic_tac_toe_env = TicTacToeEnvironment()

    def test_is_winning(self):
        self.assertTrue(
            self.tic_tac_toe_env.is_winning(curr_state=self.state_check_total_15)
        )

        self.assertFalse(
            self.tic_tac_toe_env.is_winning(curr_state=self.state_check_total_not_15)
        )

    def test_is_terminal(self):
        has_reached_terminal, result = self.tic_tac_toe_env.is_terminal(
            self.state_terminal
        )

        self.assertTrue(has_reached_terminal)
        self.assertEqual(result, "Tie")

        has_reached_terminal, result = self.tic_tac_toe_env.is_terminal(
            self.state_non_terminal
        )

        self.assertFalse(has_reached_terminal)
        self.assertEqual(result, "Resume")

    def test_allowed_positions(self):
        t_allowed_positions = self.tic_tac_toe_env.allowed_positions(self.state_env_win)

        self.assertListEqual(t_allowed_positions, [4])

    def test_allowed_values(self):
        t_agent_values, t_env_values = self.tic_tac_toe_env.allowed_values(
            curr_state=self.state_allowed_values
        )

        self.assertListEqual(t_agent_values, [5, 7, 9])
        self.assertListEqual(t_env_values, [4, 6])

        t_agent_values, t_env_values = self.tic_tac_toe_env.allowed_values(
            curr_state=self.state_no_allowed_values
        )

        self.assertListEqual(t_agent_values, [])
        self.assertListEqual(t_env_values, [])

    def test_action_space(self):
        t_agent_actions, t_env_actions = self.tic_tac_toe_env.action_space(
            curr_state=self.state_sample_action_space
        )

        self.assertListEqual(
            [ac for i, ac in enumerate(t_agent_actions)], [(6, 9), (7, 9)]
        )
        self.assertListEqual(
            [ac for i, ac in enumerate(t_env_actions)], [(6, 6), (7, 6)]
        )

    def test_state_transition(self):
        t_new_state = self.tic_tac_toe_env.state_transition(
            curr_state=self.state_old_state, curr_action=[3, 5]
        )

        self.assertListEqual(
            t_new_state, [1, 3, np.nan, 5, 2, 8, np.nan, np.nan, np.nan]
        )


if __name__ == "__main__":
    unittest.main()
