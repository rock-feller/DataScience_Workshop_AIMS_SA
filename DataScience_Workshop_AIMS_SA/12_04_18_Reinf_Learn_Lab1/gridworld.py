
import numpy as np

def oneD2twoD(idx, shape):
	return (idx / shape[1], np.mod(idx,shape[1]))

def twoD2oneD(r, c, shape):
	return r * shape[1] + c

class GridWorld(object):
	
	action_dict = {'D':0, 'U':1, 'R':2, 'L':3, 'J':4}

	def __init__(self, shape, start, terminals, obstacles, rewards, jumps):
		"""
		Args:
			shape (tuple): defines the shape of the gridworld
			start (tuple): defines the starting position of the agent
			terminals (tuple or list): defines terminal states (end of episodes)
			obstacles (tuple or list): defines obstacle squares (cannot be visited)
			rewards (dictionary): states to reward values for EXITING those states
			jumps (dictionary): non-neighbor state to state transitions
		"""
		self.shape = shape
		self.nstates = shape[0]*shape[1]
		self.start = twoD2oneD(start[0], start[1], shape)
		self.state = 0
		if isinstance(terminals, tuple):
			self.terminal2D = [terminals]
			self.terminal = [twoD2oneD(terminals[0],terminals[1],shape)]
		else:
			self.terminal2D = terminals
			self.terminal = [twoD2oneD(r,c,shape) for r,c in terminals]
		if isinstance(obstacles, tuple):
			self.obstacle2D = [obstacles]
			self.obstacle = [twoD2oneD(obstacles[0],obstacles[1],shape)]
		else:
			self.obstacle2D = obstacles
			self.obstacle = [twoD2oneD(r,c,shape) for r,c in obstacles]
		self.jump = jumps
		self.jump_from = [twoD2oneD(x,y,shape) for x,y in jumps.keys()]
		self.jump_to = [twoD2oneD(x,y,shape) for x,y in jumps.values()]
		self.buildTransitionMatrices()

		self.R = np.zeros((self.nstates,))  # rewards received upon leaving state
		for r,c in rewards.keys():
			self.R[twoD2oneD(r,c,self.shape)] = rewards[(r,c)]

	def buildTransitionMatrices(self):

		# initialize
		self.P = np.zeros((5, self.nstates, self.nstates))  # down, up, right, left, jump
		# add neighbor connections and jumps, remove for endlines
		self.P[0,range(0,self.nstates-self.shape[1]),range(self.shape[1],self.nstates)] = 1;  	# down
		self.P[1,range(self.shape[1],self.nstates),range(0,self.nstates-self.shape[1])] = 1;  	# up
		self.P[2,range(0,self.nstates-1),range(1,self.nstates)] = 1  							# right
		self.P[3,range(1,self.nstates),range(0,self.nstates-1)] = 1  							# left
		self.P[4,self.jump_from,self.jump_to] = 1												# jump
		# NOTE: above we automatically remove the top and bottom endlines, whereas this is done manually for the sides below. 
		# remove select states
		endlines = range(self.shape[1]-1,self.nstates-self.shape[1],self.shape[1])
		endlines2 = [x+1 for x in endlines]
		self.P[2,endlines,endlines2] = 0	# remove transitions at the end of the grid
		self.P[3,endlines2,endlines] = 0
		for i in range(4):
			self.P[i,:,self.obstacle] = 0  	# remove transitions into obstacles
			self.P[i,self.obstacle,:] = 0  	# remove transitions from obstacles
			self.P[i,self.terminal,:] = 0  	# remove transitions from terminal states
			self.P[i,self.jump_from,:] = 0 	# remove neighbor transitions from jump states 

	def init(self):
		self.state = self.start

	def get_state(self):
		return self.state

	def set_state(self, state):
		self.state = state

	def get_actions(self):
		return np.any(self.P[:,self.state,:],axis=1)

	def move(self, action):
		"""
		Args:
			move (str): one of ['D','U','R','L','J'] for down, up, right, left, and jump, respectively.
		Returns:
			tuple (a,b,c): a is the new state, b is the reward value, and c is a bool signifying terminal state
		"""
		
		# check if move is valid, and then move
		if not self.get_actions()[self.action_dict[action]]:
			raise Exception('Agent has tried an invalid action!')
		reward = self.R[self.state]
		self.state = np.nonzero(self.P[self.action_dict[action],self.state,:])[0][0]  # update to new state

		# check if this is a terminal state
		is_terminal = True if self.state in self.terminal else False

		return (self.state, reward, is_terminal)


class GridWorldExample1(GridWorld):
	def __init__(self):
		shape = (4,5)
		start = (0,0)
		terminals = (3,4)
		obstacles = [(3,2),(2,2),(1,2)]
		rewards = {(3,4):1}
		jumps = {(0,4):(3,0)}
		super(GridWorldExample1, self).__init__(shape, start, terminals, obstacles, rewards, jumps)

class GridWorldExample2(GridWorld):
	def __init__(self):
		shape = (5,7)
		start = (4,0)
		terminals = (4,6)
		obstacles = [(1,1),(2,1),(3,1),(4,1),(0,3),(1,3),(2,3),(3,3),(2,5),(3,5),(4,5)]
		rewards = {(3,6):1, (0,5):15}
		jumps = {(0,5):(4,0)}
		super(GridWorldExample2, self).__init__(shape, start, terminals, obstacles, rewards, jumps)

class GridWorldExample3(GridWorld):
	def __init__(self):
		shape = (6,7)
		start = (0,0)
		terminals = (0,6)
		obstacles = [(0,1),(1,1),(0,4),(1,4),(3,1),(3,0),(3,5),(3,6),(5,6)]
		rewards = {(5,0):1, (0,2):1, (4,6):1, (0,6):10}#, (0,0):1}#, 
           # (1,0):-1, (2,0):-1, (2,1):-1, (1,2):-1, (2,2):-1, (0,3):-1, (1,3):-1, (2,3):-1, (2,4):-1, 
           # (0,5):-1, (1,5):-1, (2,5):-1, (1,6):-1, (2,6):-1, (3,2):-1, (3,3):-1, (3,4):-1, (4,0):-1, 
           # (4,1):-1, (4,2):-1, (4,3):-1, (4,4):-1, (4,5):-1, (5,1):-1, (5,2):-1, (5,3):-1,(5,4):-1, (5,5):-1}
		jumps = {}
		super(GridWorldExample3, self).__init__(shape, start, terminals, obstacles, rewards, jumps)

class CliffWorld(GridWorld):
	def __init__(self):
		shape = (3,7)
		start = (2,0)
		terminals = (2,6)
		obstacles = []
		rewards = {}
		jumps = {}
		for i in range(14):
			rewards[oneD2twoD(i,shape)] = -1
		rewards[(2,0)]=-1
		for i in range(15,21):
			rewards[oneD2twoD(i,shape)] = -10
			jumps[oneD2twoD(i,shape)] = (2,0)
		super(CliffWorld, self).__init__(shape, start, terminals, obstacles, rewards, jumps)

class DreasdenGrid(GridWorld):
	def __init__(self):
		shape = (7,8)
		start = (0,0)
		terminals = (6,0)
		obstacles = [(0,1),(1,1),(0,4),(1,4),(3,1),(3,2),(3,5),(3,6),(5,6)]
		rewards = {(5,0):1, (0,2):1, (4,6):1}
		jumps = {}
		super(DreasdenGrid, self).__init__(shape, start, terminals, obstacles, rewards, jumps)