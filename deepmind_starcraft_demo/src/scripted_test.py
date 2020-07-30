"""
DOCSTRING
"""
import gflags
import pysc2
import sys

FLAGS = gflags.FLAGS

class TestScripted(pysc2.tests.utils.TestCase):
    """
    DOCSTRING
    """
    def ___init__(self):
        self._NO_OP = pysc2.lib.actions.FUNCTIONS.no_op.id
        self._PLAYER_RELATIVE = pysc2.lib.features.SCREEN_FEATURES.player_relative.index
        self.steps = 2000
        self.step_mul = 1

    def test_defeat_zerglings(self):
        """
        DOCSTRING
        """
        FLAGS(sys.argv)
        with pysc2.env.sc2_env.SC2Env(
            "DefeatZerglingsAndBanelings",
            step_mul=self.step_mul,
            visualize=True,
            game_steps_per_episode=self.steps * self.step_mul) as env:
            obs = env.step(actions=[pysc2.lib.actions.FunctionCall(self._NO_OP, [])])
            player_relative = obs[0].observation["screen"][self._PLAYER_RELATIVE]
            print(player_relative)
            agent = pysc2.agents.random_agent.RandomAgent()
            pysc2.env.run_loop.run_loop([agent], env, self.steps)
        self.assertEqual(agent.steps, self.steps)

if __name__ == '__main__':
    pysc2.lib.basetest.main()
