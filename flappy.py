import pygame
import sys
import random
import os
import time
import shutil

import argparse
import numpy as np

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import A2C
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


_DEBUG = False


class TextOutput(object):
	fonts = {}

	@classmethod
	def load_font(cls, size):
		if not size in cls.fonts:
			cls.fonts[size] = pygame.font.SysFont(None, size)

	@classmethod
	def setup(cls):
		cls.load_font(24)

	@classmethod
	def draw_text(cls, screen, text, pos, size = 24, color = (100, 100, 100)):
		cls.load_font(size)
		img = cls.fonts[size].render(text, True, color)
		rect = img.get_rect()
		screen.blit(img, pos)
		return rect


class SpriteManager(object):
	spr = {}


	@classmethod
	def load(cls, name):
		if not name in cls.spr:
			img = None
			img = pygame.image.load(f"sprites/{name}")
			img.set_colorkey((255,0,255))
			img = img.convert_alpha()
			if img == None:
				print(f"Cannot load image: 'sprites/{name}'")
				return 
			cls.spr[name] = img

	@classmethod
	def draw(cls, screen, name, pos = (0, 0)):
		cls.load(name)
		screen.blit(cls.spr[name], pos)

	@classmethod
	def draw_scaled(cls, screen, name, pos = (0, 0), scale = (1, 1)):
		cls.load(name)
		tmp = pygame.transform.scale(cls.spr[name], scale)
		screen.blit(tmp, pos)

	@classmethod
	def draw_scaled_rot(cls, screen, name, pos = (0, 0), scale = (1, 1), angle = 0):
		cls.load(name)
		tmp = pygame.transform.scale(cls.spr[name], scale)

		loc = tmp.get_rect().center
		tmp = pygame.transform.rotate(tmp, angle)
		tmp.get_rect().center = loc

		screen.blit(tmp, pos)




class Pipe():
	POS_START = 500.0
	POS_CAP = 450.0
	SPEED_START = 3.0
	SPEED_CAP = 8.0
	GAP_START = 400
	GAP_CAP = 250


	def __init__(self, difficulty, window_height, offset = 0):
		self.window_height = window_height

		self.start_pos = 0

		self.difficulty = difficulty
		self.offset = offset
		self.x = 0
		self.width = 48
		self.speed = 0
		self.gap_size = 0
		self.gap_pos = 0


		self._calc_difficulty()
		self._calc_gap_pos()


	def _calc_gap_pos(self):
		rang = self.window_height - self.gap_size
		self.gap_pos = random.randint(0, rang)


	def _calc_difficulty(self):
		self.start_pos = self.POS_START - self.difficulty * 8

		if self.start_pos <= self.POS_CAP:
			self.start_pos = self.POS_CAP

		self.x = self.start_pos
		self.x += self.offset 

		self.speed = self.SPEED_START + (self.difficulty / 6)
		if self.speed >= self.SPEED_CAP:
			self.speed = self.SPEED_CAP

		self.gap_size = self.GAP_START - (self.difficulty * 8)
		if self.gap_size <= self.GAP_CAP:
			self.gap_size = self.GAP_CAP

	def step(self, action = 0):
		self.x -= self.speed


	def render(self, window):
		pygame.draw.rect(window, (0, 0, 0), (self.x, 0, self.width, self.gap_pos))
		pygame.draw.rect(window, (0, 0, 0), (self.x, self.gap_pos + self.gap_size, self.width, self.window_height))

		SpriteManager.draw_scaled(window, "pipe_up.png", (self.x - 8, self.gap_pos- 1200), (64, 1200))
		SpriteManager.draw_scaled(window, "pipe_down.png", (self.x - 8, self.gap_pos + self.gap_size), (64, 1200))
		
class Bird():
	def __init__(self, x, y, size, window_height):
		self.x = x
		self.y = y
		self.width = size
		self.height = size
		self.window_height = window_height

		self.dead = False

		self.GRAVITY = -8.0
		self.velocity = 0.0


	def kill(self):
		self.dead = True

	def flap(self, force = 25.0):
		self.velocity = force


	def _calc_movement(self):
		self.y -= self.GRAVITY
		self.y -= self.velocity
		if self.velocity <= 0.2:
			self.velocity = 0.0
		else:
			self.velocity -= self.velocity / 10


	def step(self, action = 0):
		if not self.dead:
			self._calc_movement()
			


	def render(self, window):
		#pygame.draw.rect(window, (0, 0, 0), (self.x, self.y, self.width, self.height))

		SpriteManager.draw_scaled_rot(window, "bird.png", (self.x-4, self.y-4), (self.width+8, self.height+8), -50  + (self.velocity * 5))

def calc_distances(bird, pipe):
	ret = dict()
	ret['position'] = bird.y
	ret['horizontal'] = pipe.x - bird.x + bird.width
	ret['up'] = bird.y - pipe.gap_pos
	ret['bottom'] = pipe.gap_pos + pipe.gap_size - bird.y - bird.height
	return ret

class FlappyEnv(gym.Env):

	metadata = {"render_modes": ["human"], "render_fps": 60}

	def __init__(self, render_mode=None, human_control = False):
		super().__init__()

		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Box(low=-800.0, high=800.0, shape=(4,), dtype=np.float32)

		self.window_width = 800
		self.window_height = 600
		self.window = None
		self.clock = None
		self.human_control = human_control
		self.render_mode = render_mode

		if human_control and render_mode == None:
			self.human_control = False
			print("To use human control set render_mode to 'human'")

		if not render_mode == None:
			self._init_pygame()
			TextOutput.setup()

		self.score = 0
		self.reward = 0


		self.bird = None
		self.pipe = None
		self.pipe_phantom = None
		self.pipe_queue = []

		self.bg_pos = 0

		self.observation = np.array([],dtype="float32")

		self.reset()
	
	def _collided_pipe(self):
		if self.bird.x + self.bird.width >= self.pipe.x and self.bird.x <= self.pipe.x + self.pipe.width:
			if self.bird.y <= self.pipe.gap_pos or self.bird.y + self.bird.height >= self.pipe.gap_pos + self.pipe.gap_size:
				return True
		return False

	def _collided_screen(self):
		if self.bird.y >= self.window_height - self.bird.height:
			self.bird.y = self.window_height - self.bird.height
			return True
		if self.bird.y <= 0:
			self.bird.y = 0
			return True

	def _handle_pygame_events(self):
		global _DEBUG
		
		if self.human_control == True and _DEBUG == True:
			keys=pygame.key.get_pressed()
			if keys[pygame.K_UP]:
				self.bird.y -= 4
			if keys[pygame.K_DOWN]:
				self.bird.y += 4
			if keys[pygame.K_LEFT]:
				self.bird.x -= 4
			if keys[pygame.K_RIGHT]:
				self.bird.x += 4


		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.quit()
				sys.exit()
			if event.type == pygame.KEYDOWN and self.human_control == True:
				if event.key == pygame.K_SPACE:
					self.bird.flap()
				if event.key == pygame.K_r:
					self.reset()



	def _render_debug_info(self, window):
		TextOutput.draw_text(window, f"Bird: [x:{self.bird.x:.1f}, y:{self.bird.y:.1f}, velo:{self.bird.velocity:.1f}, dead:{self.bird.dead}]", (10, 10), 22, (255, 0, 0))
		TextOutput.draw_text(window, f"Params: [score:{self.score:.1f}, reward:{self.reward:.1f}]", (10, 25), 22, (255, 0, 0))
		TextOutput.draw_text(window, f"Observation: [position:{self.observation[0]:.1f}, horizontal:{self.observation[1]:.1f}, up:{self.observation[2]:.1f}, bottom:{self.observation[3]:.1f}]", (10, 40), 22, (255, 0, 0))
		TextOutput.draw_text(window, f"Collide: {self._collided_pipe()}", (10, 55), 22, (255, 0, 0))
		TextOutput.draw_text(window, f"Difficulty: [speed:{self.pipe.speed:.1f}, gap_size:{self.pipe.gap_size:.1f}, start_pos:{self.pipe.start_pos:.1f}]", (10, 70), 22, (255, 0, 0))


	def _calc_score(self):
		self.score += 0.1

		if self.pipe.x <= self.bird.x - self.bird.width:
			self.score += 20 + self.pipe.difficulty
			self.reward += 500 + self.pipe.difficulty*50

			self.pipe_phantom = self.pipe
			self.pipe = self.pipe_queue.pop(0)
			self.pipe_queue.append(Pipe(self.pipe_queue[-1].difficulty + 1, self.window_height, self.pipe_queue[-1].x))


	def step(self, action):
		self.reward = 1

		if self.render_mode == "human":
			self._handle_pygame_events()

		if not self.human_control and action == 1:
			self.bird.flap()


		if not self.bird.dead:
			self.bird.step()
			self.pipe_phantom.step()
			self.pipe.step()
			for p in self.pipe_queue:
				p.step()
			self._calc_score()

		if self._collided_pipe() or self._collided_screen():
			self.bird.kill()
			self.reward = -1000

		


		
		dist = calc_distances(self.bird, self.pipe)
		self.observation = np.array([dist['position'], dist['horizontal'], self.pipe.gap_pos, self.pipe.gap_pos + self.pipe.gap_size - self.bird.height],dtype="float32")


		truncated = False
		if self.score >= 5000:
			truncated = True
			self.reward = 10000000

		observation = self.observation
		reward = self.reward
		terminated = self.bird.dead
		
		info = {}



		return observation, reward, terminated, truncated, info

	def _reset_pipes(self):
		self.pipe = Pipe(1, self.window_height)
		self.pipe_phantom = Pipe(1, self.window_height, -1000)
		self.pipe_queue = []
		self.pipe_queue.append(Pipe(2, self.window_height, self.pipe.x))
		for i in range(3, 5):
			self.pipe_queue.append(Pipe(i, self.window_height, self.pipe_queue[-1].x))

	def reset(self, seed=None, options=None):
		self.bird = Bird(150, self.window_height/2-32, 48, self.window_height)
		self._reset_pipes()

		self.score = 0
		self.reward = 0

		self.bg_pos = 0


		dist = calc_distances(self.bird, self.pipe)
		self.observation = np.array([dist['position'], dist['horizontal'], self.pipe.gap_pos, self.pipe.gap_pos + self.pipe.gap_size - self.bird.height],dtype="float32")

		observation = self.observation
		info = {}

		return observation, info

	def _render_pipes(self):
		self.pipe_phantom.render(self.window)
		self.pipe.render(self.window)
		for p in self.pipe_queue:
			p.render(self.window)


	def _render_background(self):
		wh = self.window_width * 2

		self.bg_pos -= self.pipe.speed;
		if self.bg_pos <= -wh:
			self.bg_pos = 0


		SpriteManager.draw_scaled(self.window, "back.png", (self.bg_pos, 0), (wh, self.window_height))
		SpriteManager.draw_scaled(self.window, "back.png", (self.bg_pos + wh, 0), (wh, self.window_height))
		SpriteManager.draw_scaled(self.window, "back.png", (self.bg_pos + 2*wh, 0), (wh, self.window_height))

	def _render_foreground(self):
		for i in range(3):
			SpriteManager.draw(self.window, "bottom.png", (i*406, self.window_height - 11))


	def _render_frame(self):
		self._render_background()

		self.bird.render(self.window)
		self._render_pipes()


		global _DEBUG
		if _DEBUG:
			self._render_debug_info(self.window)

		self._render_foreground()

	def render(self):
		if self.window is None and self.render_mode == "human":
			self._init_pygame()
		if self.clock is None and self.render_mode == "human":
			self.clock = pygame.time.Clock()


		if self.render_mode == "human":
			self.window.fill((255, 255, 255))	

			self._render_frame()

			pygame.display.flip()
			self.clock.tick(self.metadata["render_fps"])


	def close(self):
		...

	def _init_pygame(self):
		pygame.init()
		pygame.display.init()
		pygame.font.init()
		pygame.display.set_caption("FlappyDML")
		self.window = pygame.display.set_mode((self.window_width, self.window_height), flags=pygame.SCALED, vsync=0)




def model_save_exist(name):
	models_dir = f"models/{name}/"
	logs_dir = f"logs/{name}/"
	if os.path.exists(models_dir) or os.path.exists(logs_dir):
		print(f"Name {name} already taken!")
		return True
	return False

def create_model_save(name):
	models_dir = f"models/{name}/"
	logs_dir = f"logs/{name}/"

	if model_save_exist(name):		
		name = f"{name}-{int(time.time())}"
		models_dir = f"models/{name}"
		logs_dir = f"logs/{name}/"

	os.makedirs(models_dir)
	os.makedirs(logs_dir)
	return f"{name}"

def remove_model_save(name):
	models_dir = f"models/{name}/"
	logs_dir = f"logs/{name}/"
	try:
		shutil.rmtree(models_dir)
		shutil.rmtree(logs_dir)
	except:
		print("Directory already removed!")



def learn(name = "default", alg = "PPO", episodes = 10, timestamps = 1000, verbose = 1):
	env = FlappyEnv()
	env.reset()

	name = create_model_save(name)
	models_dir = f"models/{name}/"
	logs_dir = f"logs/{name}/"

	'''import torch as th
	policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[256, 256], vf=[256, 256]))'''

	model = None
	if alg == "PPO":
		model = PPO("MlpPolicy", env, verbose=verbose, tensorboard_log=logs_dir, device="cuda")
	elif alg == "A2C":
		model = A2C("MlpPolicy", env, verbose=verbose, tensorboard_log=logs_dir, device="cuda")
	elif alg == "DQN":
		model = DQN("MlpPolicy", env, verbose=verbose, tensorboard_log=logs_dir, device="cuda")
	else:
		print("Learning algorithm must be 'PPO', 'A2C', 'DQN'! Using 'PPO'")
		model = PPO("MlpPolicy", env, verbose=verbose, tensorboard_log=logs_dir, device="cuda")
		return

	print(model.policy)

	for i in range(1, episodes):
		model.learn(total_timesteps=timestamps, reset_num_timesteps=False, tb_log_name=alg)
		model.save(f"{models_dir}/{timestamps*i}")
		model.save(f"{models_dir}/latest")

	env.close()
	print(f"Learning: '{name}' finished.")

def display(name = "default", timestamp = "latest", eps = 10):

	models_dir = f"models/{name}/"
	logs_dir = f"logs/{name}/"
	model_path = f"{models_dir}/{timestamp}.zip"


	if not os.path.exists(models_dir) or not os.path.exists(logs_dir):
		print(f"Cannot open '{name}'! File not exist!")
		return 

	model = None
	env = FlappyEnv("human", False)
	env.reset()


	if os.path.exists(f"{logs_dir}/PPO_0"):
		model = PPO.load(model_path, env=env)
	elif os.path.exists(f"{logs_dir}/A2C_0"):
		model = A2C.load(model_path, env=env)
	elif os.path.exists(f"{logs_dir}/DQN_0"):
		model = DQN.load(model_path, env=env)
	else:
		print(f"Cannot open '{name}'! Cannot find algorithm!")
		return 

	if model == None:
		print(f"Cannot open '{name}'! Error while loading model!")
		return

	for ep in range(eps):
		observation, _ = env.reset()
		done = False
		while not done:
			action, _ = model.predict(observation)
			observation, reward, terminated, truncated, info = env.step(action)
			env.render()
			done = truncated or terminated 


def play():
	remove_model_save("test")
	flappy_env = FlappyEnv("human", True)
	flappy_env.reset()
	done = False
	while not done:
		observation, reward, terminated, truncated, info = flappy_env.step(0)
		flappy_env.render()
		done = truncated or terminated 

def test():
	env = FlappyEnv()
	check_env(env, skip_render_check = False)
	learn(name = "test", alg = "PPO", episodes = 2, timestamps = 800, verbose = 1)
	display(name = "test", timestamp = "latest", eps = 5)
	remove_model_save("test")
	


def handle_args():
	parser = argparse.ArgumentParser(description="A simple script teaching bird to fly")
	subparsers = parser.add_subparsers(dest="command", help="Available commands")

	# Subparser for the "learn" command
	parser_learn = subparsers.add_parser("learn", help="Train a model")
	parser_learn.add_argument("--name", default="default", help="Model name")
	parser_learn.add_argument("--alg", default="PPO", help="Model algorithm")
	parser_learn.add_argument("--eps", type=int, default=10, help="Number of episodes")
	parser_learn.add_argument("--timestamps", type=int, default=1000, help="timestamp per episode")
	parser_learn.add_argument("--verbose", type=int, default=1, help="Verbose level")
	parser_learn.add_argument("--debug", action="store_true", help="Enable debug mode")

	# Subparser for the "display" command
	parser_display = subparsers.add_parser("display", help="Display results")
	parser_display.add_argument("--name", default="default", help="Model name")
	parser_display.add_argument("--timestamp", default="latest", help="Timestamp parameter")
	parser_display.add_argument("--eps", type=int, default=10, help="Number of episodes")
	parser_display.add_argument("--debug", action="store_true", help="Enable debug mode")

	# Subparser for the "play" command
	parser_play = subparsers.add_parser("play", help="Play a game")
	parser_play.add_argument("--debug", action="store_true", help="Enable debug mode")
		
	# Subparser for the "test" command
	parser_test = subparsers.add_parser("test", help="Tests integration")

	args = parser.parse_args()

	global _DEBUG
	try:
		_DEBUG = args.debug
	except AttributeError:
		_DEBUG = False

	if args.command == "learn":
		learn(args.name, args.alg, args.eps, args.timestamps, args.verbose)
	elif args.command == "display":
		display(args.name, args.timestamp, args.eps)
	elif args.command == "play":
		play()
	elif args.command == "test":
		test()

if __name__ == "__main__":
	handle_args()


# TODO: Provide support for hardware accelerators (TPU, CPU, GPU/CUDA) for training.
# TODO: Rename function from 'learn' to 'train' for clarity and consistency.
# TODO: Implement functionality to resume training from saved checkpoints.
# TODO: Refactor observation calculation to improve performance (e.g., replace dictionaries with tuples).
# TODO: Add a function to clean all saved model files for better disk space management.
# TODO: Allow customization of parameter size.
# TODO: Add more training algorithm options.
# TODO: Perform code cleanup and organization.
# TODO: Enhance observation process by add observation two pipes.
# TODO: Address the bug related to the rotating bird behavior within the game.
# TODO: Display distance metrics at key points such as after displaying results and human play.
# TODO: Enable simultaneous gameplay for humans and AI agents.
# TODO: Save mean reward information along with models for analysis and comparison.
# TODO: Create a graphical user interface (GUI) for improved interaction and visualization.
# TODO: Enhance SpriteManager to offer greater versatility and functionality.
# TODO: Add support for custom neural network policies to allow user-defined architectures or policies.
# TODO: Package the project for distribution via pip, including dependencies and installation instructions.
# TODO: Implement graceful closing functionality during training sessions.
# TODO: Improve text alignment in TextOutput for better readability.
# TODO: Allow simultaneous display of multiple model instances for analysis.
