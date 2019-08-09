
import numpy as np
from gridworld import twoD2oneD, oneD2twoD
from plotFunctions import plotStateActionValue

class RLAgent(object):

	actionlist = np.array(['D','U','R','L','J'])
	action_dict = {'D':0, 'U':1, 'R':2, 'L':3, 'J':4}

	def __init__(self, world):
		self.world = world
		self.v = np.zeros((self.world.nstates,))
		self.q = np.zeros((self.world.nstates,5))  # one column for each action
		self.policy = self.randPolicy

	def reset(self):
		self.world.init()
		self.state = self.world.get_state()

	def choose_action(self):
		state = self.world.get_state()
		actions = self.world.get_actions()
		self.action = self.policy(state, actions)
		return self.action

	def take_action(self, action):
		(self.state, self.reward, terminal) = self.world.move(action)
		return terminal

	def take_action_(self, action):
		(state, reward, terminal) = self.world.move(action)
		return state
    
	def run_episode(self):
		print("Running episode...")
		is_terminal = False
		self.reset()
		c = 0
		while (is_terminal == False):
			c += 1
			print("Current position:", oneD2twoD(self.state,self.world.shape))
			action = self.choose_action()
			is_terminal = self.take_action(action)
			print("Action number", c, ": move to", oneD2twoD(self.state,self.world.shape))
			print("Reward:", self.reward)
		print("Terminated.")

	def randPolicy(self, state, actions):
		available_actions = self.actionlist[actions]
		return available_actions[np.random.randint(len(available_actions))]
    
	def detPolicy(self, state, actions):
		prev_state = self.state
		available_actions = self.actionlist[actions]
		for i in available_actions:
			next_state = np.nonzero(self.world.P[self.action_dict[i], self.world.get_state(), :])[0][0]
			print (prev_state, next_state)
			if self.P_pi[prev_state][next_state] == 1:
				return i
            
	def greedyQPolicy(self, state, actions):
		idx = np.arange(5)[actions]
		return self.actionlist[idx[np.argmax(self.q[state,actions])]]

	def epsilongreedyQPolicy(self, state, actions, epsilon=0.1):
		idx = np.arange(5)[actions]
		greedy_action = self.actionlist[idx[np.argmax(self.q[state,actions])]]
		nongreedy_actions = np.delete(self.actionlist[self.world.get_actions()],np.argwhere(self.actionlist[self.world.get_actions()]==greedy_action))
		r = np.random.rand()
		for c in range(len(nongreedy_actions)):
			if (r<((c+1)*epsilon/len(nongreedy_actions))):
				return nongreedy_actions[c]
		return greedy_action

class RLExampleAgent(RLAgent):
	def __init__(self, world):
		super(RLExampleAgent, self).__init__(world)
		self.v = np.zeros(world.nstates)
		self.q = np.zeros((world.nstates,5))
		self.P_pi = np.zeros((world.nstates,world.nstates))
		rows = np.array([0,7,14,15,16,17,18,19,20,13])
		cols = np.array([7,14,15,16,17,18,19,20,13,6])
		self.P_pi[rows,cols] = 1
		self.policy = self.detPolicy # change this to random policy to ignore the transition probabilities defined above

class DP_Agent(RLAgent):
	def initDetPolicy(self):
		self.policy = self.detPolicy

# this random policy assigns equal probability to each action
	def initRandomPolicy(self):
		Psum = self.world.P.sum(axis=0)
		Pnorm = Psum.sum(axis=1)
		zero_idxs = Pnorm==0.0
		Pnorm[zero_idxs] = 1.0
		self.P_pi = (Psum.T / Pnorm).T

	def evaluatePolicy(self, gamma):
		delta = 1
		while delta > 0.001:
			v_new = self.world.R + gamma*self.P_pi.dot(self.v)
			delta = np.max(np.abs(v_new - self.v))
			self.v = v_new
            
# this was my original attempt, note that the function above simplifies the operation into a vector

	def evaluatePolicy_beta(self, gamma):
		delta = 1
		self.v = np.zeros((self.world.nstates))
		while delta > 0.001:
			delta = 0
			for s in range(self.world.nstates):
				value_s = self.world.R[s] + gamma*np.dot(self.P_pi[s],self.v)
				delta = max(delta, abs(self.v[s] - value_s))
				self.v[s] = value_s               
            
	def improvePolicy(self):
		for i in range(self.world.nstates):
			if i in self.world.obstacle or i in self.world.terminal:
				pass
			else:
				next_states = np.nonzero(self.P_pi[i, :])[0]
				greedy = next_states[np.argmax(self.v[next_states])]
				self.P_pi[i,:] = 0 
				self.P_pi[i,greedy] = 1
    
	def policyIteration(self, gamma):
		stable = False
		while stable == False:
			b = self.P_pi
			self.evaluatePolicy(gamma)
			self.improvePolicy
			if b.all() == self.P_pi.all():
				stable = True
    
class TDSarsa_Agent(RLAgent):

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
		trial = 1
		while trial < ntrials: 
			is_terminal = False
			self.world.set_state(0)
			action = self.choose_action()
			while is_terminal == False:
				action1 = self.action_dict[action]
				state1 = self.world.get_state()
				is_terminal = self.take_action(action)
				if is_terminal == True:
					break
				action = self.choose_action()
				state = self.world.get_state()
				self.q[state1][action1] = self.q[state1][action1] + alpha*(self.world.R[state1] + gamma*self.q[state][self.action_dict[action]]-self.q[state1][action1])
				trial = trial + 1

	def policyIteration(self, gamma, alpha, ntrials):
		maxitrs = 1000
		stable = False
		#self.policy = self.epsilongreedyQPolicy
		itr = 0
		while (stable == False and itr < maxitrs):
			itr += 1
			b = self.q
			self.evaluatePolicyQ(gamma, alpha, ntrials)
			if b.all() == self.q.all():
				stable = True 

class TDQ_Agent(TDSarsa_Agent):
	
	def __init__(self, world):
		super(TDQ_Agent, self).__init__(world)
		self.offpolicy = self.greedyQPolicy

	def choose_offpolicy_action(self):
		state = self.world.get_state()
		actions = self.world.get_actions()
		self.action = self.offpolicy(state, actions)
		return self.action

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
		pass


class TDSarsaLambda_Agent(TDSarsa_Agent):
	
	def __init__(self, world, lamb):
		super(TDSarsaLambda_Agent, self).__init__(world)
		self.lamb = lamb

	def evaluatePolicyQ(self, gamma, alpha, ntrials):
		pass