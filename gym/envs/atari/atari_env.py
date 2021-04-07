import numpy as np
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import matplotlib
import matplotlib.pyplot as plt
import math
import sys

try:
    import atari_py
except ImportError as e:
    raise error.DependencyNotInstalled(
            "{}. (HINT: you can install Atari dependencies by running "
            "'pip install gym[atari]'.)".format(e))


def to_ram(ale):
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size), dtype=np.uint8)
    ale.getRAM(ram)
    return ram

#global variable to store the previous frames RGB
prevRGB=0

#function to find coordinates of Aliens and Agent
def coordinates(diffmat,bval):
  wherer = np.equal(diffmat[:,:,2], bval)

  indices = np.where(wherer) 

  listx=[]
  listy=[]
  for i in range(0,len(indices[0])):
    tempy= indices[0][i]
    tempx= indices[1][i]

    for j in range(0,len(indices[0])):

      if tempy == indices[0][j] and abs(tempx - indices[1][j]) == 1:
        listx=np.append(listx, tempx)
        listy=np.append(listy, tempy)

  return np.average(listx), np.average(listy)
 
#function to locate position of eggs 
def cooregg(d1) :
  wherer = np.not_equal(d1[:,:,2], 0)
  indices = np.where(wherer) 

  listx=[]
  listy=[]
  
  for i in range(0,len(indices[0])):
    tempy= indices[0][i]
    tempx= indices[1][i]
    if tempy < 169:
      if d1[tempy+1,tempx,2] != 0  and  d1[tempy-1,tempx,2] == 0 and  d1[tempy,tempx-1,2] == 0 and  d1[tempy,tempx+1,2] == 0 and  d1[tempy-1,tempx-1,2]==0 and  d1[tempy-1,tempx+1,2] == 0 and  d1[tempy+1,tempx-1,2] == 0 and  d1[tempy+1,tempx+1,2] == 0 and  d1[tempy+2,tempx,2] != 188 and  d1[tempy+3,tempx,2] != 188 and  d1[tempy+2,tempx,2] != 28 and  d1[tempy+3,tempx,2] != 28:
        listx=np.append(listx, tempx)
        listy=np.append(listy, tempy) 
  return listx, listy

#function to determine distance between objects using pythagoras
def distance(x1,x2,y1,y2):
  distance= math.sqrt((x1-x2)**2+(y1-y2)**2)
  return distance

class AtariEnv(gym.Env, utils.EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
            self,
            game='pong',
            mode=None,
            difficulty=None,
            obs_type='ram',
            frameskip=5,
            repeat_action_probability=0.,
            full_action_space=False):
        """Frameskip should be either a tuple (indicating a random range to
        choose from, with the top value exclude), or an int."""

        utils.EzPickle.__init__(
                self,
                game,
                mode,
                difficulty,
                obs_type,
                frameskip,
                repeat_action_probability)
        assert obs_type in ('ram', 'image')

        self.game = game
        self.game_path = atari_py.get_game_path(game)
        self.game_mode = mode
        self.game_difficulty = difficulty

        if not os.path.exists(self.game_path):
            msg = 'You asked for game %s but path %s does not exist'
            raise IOError(msg % (game, self.game_path))
        self._obs_type = obs_type
        self.frameskip = frameskip
        self.ale = atari_py.ALEInterface()
        self.viewer = None

        # Tune (or disable) ALE's action repeat:
        # https://github.com/openai/gym/issues/349
        assert isinstance(repeat_action_probability, (float, int)), \
                "Invalid repeat_action_probability: {!r}".format(repeat_action_probability)
        self.ale.setFloat(
                'repeat_action_probability'.encode('utf-8'),
                repeat_action_probability)

        self.seed()

        self._action_set = (self.ale.getLegalActionSet() if full_action_space
                            else self.ale.getMinimalActionSet())
        self.action_space = spaces.Discrete(len(self._action_set))

        (screen_width, screen_height) = self.ale.getScreenDims()
        if self._obs_type == 'ram':
            self.observation_space = spaces.Box(low=0, high=255, dtype=np.uint8, shape=(128,))
        elif self._obs_type == 'image':
            self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        # Empirically, we need to seed before loading the ROM.
        self.ale.setInt(b'random_seed', seed2)
        self.ale.loadROM(self.game_path)

        if self.game_mode is not None:
            modes = self.ale.getAvailableModes()

            assert self.game_mode in modes, (
                "Invalid game mode \"{}\" for game {}.\nAvailable modes are: {}"
            ).format(self.game_mode, self.game, modes)
            self.ale.setMode(self.game_mode)

        if self.game_difficulty is not None:
            difficulties = self.ale.getAvailableDifficulties()

            assert self.game_difficulty in difficulties, (
                "Invalid game difficulty \"{}\" for game {}.\nAvailable difficulties are: {}"
            ).format(self.game_difficulty, self.game, difficulties)
            self.ale.setDifficulty(self.game_difficulty)

        return [seed1, seed2]

    def step( self, a ):
        reward = 0.0

        #creating new variable to store value of original reward function
        gamescore=0.0
       
        global prevRGB
        action = self._action_set[a]

        if isinstance(self.frameskip, int):
            num_steps = self.frameskip
        else:
            num_steps = self.np_random.randint(self.frameskip[0], self.frameskip[1])
        
        for _ in range(num_steps):
            gamescore += self.ale.act(action)

        ob = self._get_obs()

        
        #Finding the difference in RGB for the very first frame and the current frame

        x= self.x[0:175,:]

        y=ob[0:175,:]
        d= x-y

        #Determining distance between agent and aliens

        [xcoragent,ycoragent]= coordinates(d,188)
        [xcorAlienA,ycorAlienA]= coordinates(d,216)
       
        distA= distance(xcoragent,xcorAlienA,ycoragent,ycorAlienA)
        
        [xcorAlienB,ycorAlienB]= coordinates(d,100)
        
        distB= distance(xcoragent,xcorAlienB,ycoragent,ycorAlienB)
        
        [xcorAlienC,ycorAlienC]= coordinates(d,228)
        distC= distance(xcoragent,xcorAlienC,ycoragent,ycorAlienC)
        
        
        #Finding difference between previous RGB frame and current frame
        d1= prevRGB - y
        prevRGB= y

        #Finding location of each egg and determining which is closest to agent
        [xcorEgg,ycorEgg]= cooregg(d1)

        #print(len(xcorEgg))

        mindist = 1000
        for i in range(0,len(xcorEgg)):
          tempdist= math.sqrt((xcoragent-xcorEgg[i])**2+(ycoragent-ycorEgg[i])**2)
          if tempdist < mindist :
            mindist=tempdist

        #print('Nearest Egg:',mindist)

        
        #determining the distance of the closest alien

        minAlien= min(distA,distB,distC)

        if math.isnan(distA) and math.isnan(distB) and math.isnan(distC):
          minAlien=1000

        #print('Nearest Alien:',minAlien)

        #Setting up conditions for new reward function
        
        if mindist < 20 and minAlien > 40  :
          reward=reward + 10

        if minAlien <= 40:
          reward= reward - 10

        #print('reward:', reward)

        return minAlien, mindist, ob, gamescore, reward, self.ale.game_over(), {"ale.lives": self.ale.lives()}

    def _get_image(self):
        return self.ale.getScreenRGB2()

    def _get_ram(self):
        return to_ram(self.ale)

    @property
    def _n_actions(self):
        return len(self._action_set)

    def _get_obs(self):
        if self._obs_type == 'ram':
            return self._get_ram()
        elif self._obs_type == 'image':
            img = self._get_image()
        return img

    # return: (states, observations)
    def reset(self):

        self.ale.reset_game()
        self.x = self._get_obs()
        global prevRGB
        prevRGB= self._get_obs()[0:175,:]
        
        return self._get_obs()

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action

    def clone_state(self):
        """Clone emulator state w/o system state. Restoring this state will
        *not* give an identical environment. For complete cloning and restoring
        of the full state, see `{clone,restore}_full_state()`."""
        state_ref = self.ale.cloneState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_state(self, state):
        """Restore emulator state w/o system state."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreState(state_ref)
        self.ale.deleteState(state_ref)

    def clone_full_state(self):
        """Clone emulator state w/ system state including pseudorandomness.
        Restoring this state will give an identical environment."""
        state_ref = self.ale.cloneSystemState()
        state = self.ale.encodeState(state_ref)
        self.ale.deleteState(state_ref)
        return state

    def restore_full_state(self, state):
        """Restore emulator state w/ system state including pseudorandomness."""
        state_ref = self.ale.decodeState(state)
        self.ale.restoreSystemState(state_ref)
        self.ale.deleteState(state_ref)


ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}
