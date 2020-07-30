"""
DOCSTRING
"""
import baselines
import dill
import gflags
import numpy
import os
import pysc2
import sys
import tempfile
import tensorflow
import zipfile

FLAGS = gflags.FLAGS

class ActWrapper:
    """
    DOCSTRING
    """
    def __init__(self, act):
        self._act = act
        #self._act_params = act_params

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    @staticmethod
    def load(path, act_params, num_cpu=16):
        with open(path, "rb") as f:
            model_data = dill.load(f)
        act = baselines.deepq.build_act(**act_params)
        sess = U.make_session(num_cpu=num_cpu)
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)
            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            baselines.common.tf_util.load_state(os.path.join(td, "model"))
        return ActWrapper(act)

    def save(self, path):
        """
        Save model to a pickle located at `path`.
        """
        with tempfile.TemporaryDirectory() as td:
            baselines.common.tf_util.save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            dill.dump((model_data), f)

class DeepQMineralShards:
    """
    DOCSTRING
    """
    def __init__(self):
        self._PLAYER_RELATIVE = pysc2.lib.features.SCREEN_FEATURES.player_relative.index
        self._PLAYER_FRIENDLY = 1
        self._NO_OP = pysc2.lib.actions.FUNCTIONS.no_op.id
        self._MOVE_SCREEN = pysc2.lib.actions.FUNCTIONS.Move_screen.id
        self._SELECT_ARMY = pysc2.lib.actions.FUNCTIONS.select_army.id
        self._NOT_QUEUED = [0]
        self._SELECT_ALL = [0]
        self.UP, self.DOWN, self.LEFT, self.RIGHT = 'up', 'down', 'left', 'right'

    def load(path, act_params, num_cpu=16):
        """
        Load act function that was returned by learn function.

        Parameters
        ----------
        path: str
            path to the act function pickle
        num_cpu: int
            number of cpus to use for executing the policy

        Returns
        -------
        act: ActWrapper
            function that takes a batch of observations and returns actions.
        """
        return ActWrapper.load(path, num_cpu=num_cpu, act_params=act_params)

    def learn(
        env,
        q_func,
        num_actions=4,
        lr=5e-4,
        max_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=1,
        batch_size=32,
        print_freq=1,
        checkpoint_freq=10000,
        learning_starts=1000,
        gamma=1.0,
        target_network_update_freq=500,
        prioritized_replay=False,
        prioritized_replay_alpha=0.6,
        prioritized_replay_beta0=0.4,
        prioritized_replay_beta_iters=None,
        prioritized_replay_eps=1e-6,
        num_cpu=16,
        param_noise=False,
        param_noise_threshold=0.05,
        callback=None):
        """
        Train a deepq model.

        Parameters
        -------
        env: pysc2.env.SC2Env
            environment to train on
        q_func: (tensorflow.Variable, int, str, bool) -> tensorflow.Variable
            the model that takes the following inputs:
                observation_in: object
                     the output of observation placeholder
                num_actions: int
                    number of actions
                scope: str
                reuse: bool
                    should be passed to outer variable scope
            and returns a tensor of shape (batch_size, num_actions) with values of every action.
        lr: float
            learning rate for adam optimizer
        max_timesteps: int
            number of env steps to optimizer for
        buffer_size: int
            size of the replay buffer
        exploration_fraction: float
            fraction of entire training period over which the exploration rate is annealed
        exploration_final_eps: float
            final value of random action probability
        train_freq: int
            update the model every `train_freq` steps.
            set to None to disable printing
        batch_size: int
            size of a batched sampled from replay buffer for training
        print_freq: int
            how often to print out training progress
            set to None to disable printing
        checkpoint_freq: int
            how often to save the model. This is so that the best version is restored
            at the end of the training. If you do not wish to restore the best version at
            the end of the training set this variable to None.
        learning_starts: int
            how many steps of the model to collect transitions for before learning starts
        gamma: float
            discount factor
        target_network_update_freq: int
            update the target network every `target_network_update_freq` steps.
        prioritized_replay: True
            if True prioritized replay buffer will be used.
        prioritized_replay_alpha: float
            alpha parameter for prioritized replay buffer
        prioritized_replay_beta0: float
            initial value of beta for prioritized replay buffer
        prioritized_replay_beta_iters: int
            number of iterations over which beta will be annealed from initial value
            to 1.0. If set to None equals to max_timesteps.
        prioritized_replay_eps: float
            epsilon to add to the TD errors when updating priorities.
        num_cpu: int
            number of cpus to use for training
        callback: (locals, globals) -> None
            function called at every steps with state of the algorithm.
            If callback returns true training stops.

        Returns
        -------
        act: ActWrapper
            Wrapper over act function. Adds ability to save it and load it.
            See header of baselines/deepq/categorical.py for details on the act function.
        """
        # Create all the functions necessary to train the model
        sess = baselines.common.tf_util.make_session(num_cpu=num_cpu)
        sess.__enter__()
        def make_obs_ph(name):
          return baselines.common.tf_util.BatchInput((64, 64), name=name)
        act, train, update_target, debug = baselines.deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=num_actions,
            optimizer=tensorflow.train.AdamOptimizer(learning_rate=lr),
            gamma=gamma,
            grad_norm_clipping=10)
        act_params = {
            'make_obs_ph': make_obs_ph,
            'q_func': q_func,
            'num_actions': num_actions,}
        # create the replay buffer
        if prioritized_replay:
            replay_buffer = baselines.deepq.replay_buffer.PrioritizedReplayBuffer(
                buffer_size, alpha=prioritized_replay_alpha)
            if prioritized_replay_beta_iters is None:
              prioritized_replay_beta_iters = max_timesteps
            beta_schedule = baselines.common.schedules.LinearSchedule(
                prioritized_replay_beta_iters,
                initial_p=prioritized_replay_beta0,
                final_p=1.0)
        else:
            replay_buffer = baselines.deepq.replay_buffer.ReplayBuffer(buffer_size)
            beta_schedule = None
        # create the schedule for exploration starting from 1.
        exploration = baselines.common.schedules.LinearSchedule(
            schedule_timesteps=int(exploration_fraction * max_timesteps),
            initial_p=1.0, final_p=exploration_final_eps)
        # initialize the parameters and copy them to the target network.
        baselines.common.tf_util.initialize()
        update_target()
        episode_rewards = [0.0]
        #episode_minerals = [0.0]
        saved_mean_reward = None
        path_memory = numpy.zeros((64,64))
        obs = env.reset()
        # select all marines first
        obs = env.step(actions=[pysc2.lib.actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])])
        player_relative = obs[0].observation["screen"][self._PLAYER_RELATIVE]
        screen = player_relative + path_memory
        player_y, player_x = (player_relative == self._PLAYER_FRIENDLY).nonzero()
        player = [int(player_x.mean()), int(player_y.mean())]
        if(player[0] > 32):
            screen = shift(self.LEFT, player[0]-32, screen)
        elif(player[0] < 32):
            screen = shift(self.RIGHT, 32 - player[0], screen)
        if(player[1] > 32):
            screen = shift(self.UP, player[1]-32, screen)
        elif(player[1] < 32):
            screen = shift(self.DOWN, 32 - player[1], screen)
        reset = True
        with tempfile.TemporaryDirectory() as td:
            model_saved = False
            model_file = os.path.join(td, "model")
            for t in range(max_timesteps):
                if callback is not None:
                    if callback(locals(), globals()):
                        break
                # take action and update exploration to the newest value
                kwargs = {}
                if not param_noise:
                    update_eps = exploration.value(t)
                    update_param_noise_threshold = 0.0
                else:
                    update_eps = 0.0
                    if param_noise_threshold >= 0.0:
                        update_param_noise_threshold = param_noise_threshold
                    else:
                        # compute the threshold such that the KL divergence
                        # between perturbed and non-perturbed
                        # policy is comparable to eps-greedy exploration
                        # with eps = exploration.value(t).
                        # See Appendix C.1 in Parameter Space Noise for Exploration,
                        # Plappert et al., 2017 for detailed explanation.
                        update_param_noise_threshold = \
                            -numpy.log(
                                1.0 - exploration.value(t) +
                                exploration.value(t) / float(num_actions))
                kwargs['reset'] = reset
                kwargs['update_param_noise_threshold'] = update_param_noise_threshold
                kwargs['update_param_noise_scale'] = True
            action = act(numpy.array(screen)[None], update_eps=update_eps, **kwargs)[0]
            reset = False
            coord = [player[0], player[1]]
            rew = 0
            path_memory_ = numpy.array(path_memory, copy=True)
            if(action == 0): # UP
                if(player[1] >= 16):
                    coord = [player[0], player[1] - 16]
                    path_memory_[player[1] - 16 : player[1], player[0]] = -1
                elif(player[1] > 0):
                    coord = [player[0], 0]
                    path_memory_[0 : player[1], player[0]] = -1
                #else:
                #    rew -= 1
            elif(action == 1): # DOWN
                if(player[1] <= 47):
                    coord = [player[0], player[1] + 16]
                    path_memory_[player[1] : player[1] + 16, player[0]] = -1
                elif(player[1] > 47):
                    coord = [player[0], 63]
                    path_memory_[player[1] : 63, player[0]] = -1
                #else:
                #    rew -= 1
            elif(action == 2): # LEFT
                if(player[0] >= 16):
                    coord = [player[0] - 16, player[1]]
                    path_memory_[player[1], player[0] - 16 : player[0]] = -1
                elif(player[0] < 16):
                    coord = [0, player[1]]
                    path_memory_[player[1], 0 : player[0]] = -1
                #else:
                #    rew -= 1
            elif(action == 3): # RIGHT
                if(player[0] <= 47):
                    coord = [player[0] + 16, player[1]]
                    path_memory_[player[1], player[0] : player[0] + 16] = -1
                elif(player[0] > 47):
                    coord = [63, player[1]]
                    path_memory_[player[1], player[0] : 63] = -1
                #else:
                #    rew -= 1
            #else:
            #    # cannot move, give minus reward
            #    rew -= 1
            #if(path_memory[coord[1],coord[0]] != 0):
            #    rew -= 0.5
            path_memory = numpy.array(path_memory_)
            #print("action : %s Coord : %s" % (action, coord))
            if self._MOVE_SCREEN not in obs[0].observation["available_actions"]:
                obs = env.step(actions=[pysc2.lib.actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])])
            new_action = [pysc2.lib.actions.FunctionCall(self._MOVE_SCREEN, [self._NOT_QUEUED, coord])]
            #else:
            #    new_action = [pysc2.lib.actions.FunctionCall(self._NO_OP, [])]
            obs = env.step(actions=new_action)
            player_relative = obs[0].observation["screen"][self._PLAYER_RELATIVE]
            new_screen = player_relative + path_memory
            player_y, player_x = (player_relative == self._PLAYER_FRIENDLY).nonzero()
            player = [int(player_x.mean()), int(player_y.mean())]
            if(player[0] > 32):
                new_screen = shift(self.LEFT, player[0]-32, new_screen)
            elif(player[0] < 32):
                new_screen = shift(self.RIGHT, 32 - player[0], new_screen)
            if(player[1] > 32):
                new_screen = shift(self.UP, player[1]-32, new_screen)
            elif(player[1] < 32):
                new_screen = shift(self.DOWN, 32 - player[1], new_screen)
            rew = obs[0].reward
            done = obs[0].step_type == pysc2.env.environment.StepType.LAST
            # store transition in the replay buffer
            replay_buffer.add(screen, action, rew, new_screen, float(done))
            screen = new_screen
            episode_rewards[-1] += rew
            #episode_minerals[-1] += obs[0].reward
            if done:
                obs = env.reset()
                player_relative = obs[0].observation["screen"][self._PLAYER_RELATIVE]
                screen = player_relative + path_memory
                player_y, player_x = (player_relative == self._PLAYER_FRIENDLY).nonzero()
                player = [int(player_x.mean()), int(player_y.mean())]
                if(player[0] > 32):
                    screen = shift(self.LEFT, player[0]-32, screen)
                elif(player[0] < 32):
                    screen = shift(self.RIGHT, 32 - player[0], screen)
                if(player[1] > 32):
                    screen = shift(self.UP, player[1]-32, screen)
                elif(player[1] < 32):
                    screen = shift(self.DOWN, 32 - player[1], screen)
                # select all marines first
                env.step(actions=[pysc2.lib.actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])])
                episode_rewards.append(0.0)
                #episode_minerals.append(0.0)
                path_memory = numpy.zeros((64, 64))
                reset = True
            if t > learning_starts and t % train_freq == 0:
            # minimize the error in Bellman's equation on a batch sampled from replay buffer
                if prioritized_replay:
                    experience = replay_buffer.sample(batch_size, beta=beta_schedule.value(t))
                    (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
                else:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
                    weights, batch_idxes = numpy.ones_like(rewards), None
                td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)
                if prioritized_replay:
                    new_priorities = numpy.abs(td_errors) + prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)
            if t > learning_starts and t % target_network_update_freq == 0:
                # update target network periodically
                update_target()
            mean_100ep_reward = round(numpy.mean(episode_rewards[-101:-1]), 1)
            #mean_100ep_mineral = round(numpy.mean(episode_minerals[-101:-1]), 1)
            num_episodes = len(episode_rewards)
            if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
                baselines.logger.record_tabular("steps", t)
                baselines.logger.record_tabular("episodes", num_episodes)
                baselines.logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                #baselines.logger.record_tabular("mean 100 episode mineral", mean_100ep_mineral)
                baselines.logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                baselines.logger.dump_tabular()
            if (checkpoint_freq is not None and t > learning_starts and 
                num_episodes > 100 and t % checkpoint_freq == 0):
                if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                    if print_freq is not None:
                        baselines.logger.log(
                            "Saving model due to mean reward increase: {} -> {}".format(
                                saved_mean_reward, mean_100ep_reward))
                    baselines.common.tf_util.save_state(model_file)
                    model_saved = True
                    saved_mean_reward = mean_100ep_reward
            if model_saved:
                if print_freq is not None:
                    baselines.logger.log(
                        "Restored model with mean reward: {}".format(saved_mean_reward))
                baselines.common.tf_util.load_state(model_file)
        return ActWrapper(act)

    def intToCoordinate(num, size=64):
        """
        DOCSTRING
        """
        if size!=64:
            num = num * size * size // 4096
        y = num // size
        x = num - size * y
        return [x, y]

    def shift(direction, number, matrix):
        """
        shift given 2D matrix in-place the given number of rows or columns
        in the specified (UP, DOWN, LEFT, RIGHT) direction and return it
        """
        if direction in (self.UP):
            matrix = numpy.roll(matrix, -number, axis=0)
            matrix[number:,:] = -2
            return matrix
        elif direction in (self.DOWN):
            matrix = numpy.roll(matrix, number, axis=0)
            matrix[:number,:] = -2
            return matrix
        elif direction in (self.LEFT):
            matrix = numpy.roll(matrix, -number, axis=1)
            matrix[:,number:] = -2
            return matrix
        elif direction in (self.RIGHT):
            matrix = numpy.roll(matrix, number, axis=1)
            matrix[:,:number] = -2
            return matrix
        else:
            return matrix

class EnjoyMineralShards:
    """
    DOCSTRING
    """
    def __init__(self):
        self._PLAYER_RELATIVE = pysc2.lib.features.SCREEN_FEATURES.player_relative.index
        self._PLAYER_FRIENDLY = 1
        self._MOVE_SCREEN = pysc2.lib.actions.FUNCTIONS.Move_screen.id
        self._SELECT_ARMY = pysc2.libactions.FUNCTIONS.select_army.id
        self._NOT_QUEUED = [0]
        self._SELECT_ALL = [0]
        self.step_mul = 16
        self.steps = 400
        self.UP, self.DOWN, self.LEFT, self.RIGHT = 'up', 'down', 'left', 'right'

    def __call__():
        FLAGS(sys.argv)
        with pysc2.env.sc2_env.SC2Env(
            "CollectMineralShards", step_mul=step_mul, visualize=True,
            game_steps_per_episode=steps * step_mul) as env:
            model = baselines.deepq.models.cnn_to_mlp(
                convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], dueling=True)
            def make_obs_ph(name):
                return baselines.common.tf_util.BatchInput((64, 64), name=name)
            act_params = {
                'make_obs_ph': make_obs_ph, 'q_func': model, 'num_actions': 4}
            act = deepq_mineral_shards.load("mineral_shards.pkl", act_params=act_params)
            while True:
                obs = env.reset()
                episode_rew = 0
                done = False
                step_result = env.step(
                    actions=[pysc2.lib.actions.FunctionCall(self._SELECT_ARMY, [self._SELECT_ALL])])
                while not done:
                    player_relative = step_result[0].observation["screen"][self._PLAYER_RELATIVE]
                    obs = player_relative
                    player_y, player_x = (player_relative == self._PLAYER_FRIENDLY).nonzero()
                    player = [int(player_x.mean()), int(player_y.mean())]
                    if(player[0] > 32):
                        obs = shift(self.LEFT, player[0]-32, obs)
                    elif(player[0] > 32):
                        obs = shift(self.RIGHT, 32 - player[0], obs)
                    if(player[1] > 32):
                        obs = shift(self.UP, player[1]-32, obs)
                    elif(player[1] < 32):
                        obs = shift(self.DOWN, 32 - player[1], obs)
                    action = act(obs[None])[0]
                    coord = [player[0], player[1]]
                    if(action == 0): # UP
                        if(player[1] >= 16):
                            coord = [player[0], player[1] - 16]
                        elif(player[1] > 0):
                            coord = [player[0], 0]
                    elif(action == 1): # DOWN
                        if(player[1] <= 47):
                            coord = [player[0], player[1] + 16]
                        elif(player[1] > 47):
                            coord = [player[0], 63]
                    elif(action == 2): # LEFT
                        if(player[0] >= 16):
                            coord = [player[0] - 16, player[1]]
                        elif(player[0] < 16):
                            coord = [0, player[1]]
                    elif(action == 3): # RIGHT
                        if(player[0] <= 47):
                            coord = [player[0] + 16, player[1]]
                        elif(player[0] > 47):
                            coord = [63, player[1]]
                    new_action = [pysc2.lib.actions.FunctionCall(
                        self._MOVE_SCREEN, [self._NOT_QUEUED, coord])]
                    step_result = env.step(actions=new_action)
                    rew = step_result[0].reward
                    done = step_result[0].step_type == pysc2.env.environment.StepType.LAST
                    episode_rew += rew
                print("Episode reward", episode_rew)

    def shift(direction, number, matrix):
      """
      Shift given 2D matrix in-place the given number of rows or columns,
      in the specified (UP, DOWN, LEFT, RIGHT) direction and return.
      """
      if direction in (self.UP):
        matrix = numpy.roll(matrix, -number, axis=0)
        matrix[number:,:] = -2
        return matrix
      elif direction in (self.DOWN):
        matrix = numpy.roll(matrix, number, axis=0)
        matrix[:number,:] = -2
        return matrix
      elif direction in (self.LEFT):
        matrix = numpy.roll(matrix, -number, axis=1)
        matrix[:,number:] = -2
        return matrix
      elif direction in (self.RIGHT):
        matrix = numpy.roll(matrix, number, axis=1)
        matrix[:,:number] = -2
        return matrix
      else:
        return matrix

class TrainMineralShards:
    """
    DOCSTRING
    """
    def __init__(self):
        self.step_mul = 8

    def __call__():
        FLAGS(sys.argv)
        with pysc2.env.sc2_env.SC2Env(
            "CollectMineralShards", step_mul=self.step_mul, visualize=True) as env:
            model = baselines.deepq.models.cnn_to_mlp(
                convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)], hiddens=[256], dueling=True)
            act = deepq_mineral_shards.learn(
                env,
                q_func=model,
                num_actions=4,
                lr=1e-5,
                max_timesteps=2000000,
                buffer_size=100000,
                exploration_fraction=0.5,
                exploration_final_eps=0.01,
                train_freq=4,
                learning_starts=100000,
                target_network_update_freq=1000,
                gamma=0.99,
                prioritized_replay=True)
            act.save("mineral_shards.pkl")

if __name__ == '__main__':
    enjoy_mineral_shards = EnjoyMineralShards()
    enjoy_mineral_shards()
