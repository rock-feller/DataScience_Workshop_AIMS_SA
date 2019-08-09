# gridworld_setup.py

import numpy as np
import matplotlib.pyplot as plt
from plotFunctions import plotWorld, plotStateValue, plotStateActionValue, plotPolicyPi
from gridworld import GridWorldExample1
from rlagents import RLAgent

# build and plot world
world = GridWorldExample1()
plotWorld(world,plotNow=False)

# initialize an agent and run an episode (see program printout)
agent = RLExampleAgent(world)
agent.run_episode()

# plot value functions and policy
plotStateValue(agent.v, world, plotNow=False)
plotStateActionValue(agent.q, world, plotNow=False)
plotPolicyPi(agent.Ppi, world, plotNow=False)

plt.show()