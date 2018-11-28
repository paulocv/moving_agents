"""
Moving agents models for flat surfaces.

"""

import numpy as np
import random as rnd
import math
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace
from toolbox.rand_tools import random_function_generator_from_dict
# from toolbox.surfaces import FlatSpace


class ContinuousRandomWalker(Agent):
    """Base class for a random walk agent that holds an epidemic
    spreading process. Moves in a continuous space.

    This agent is meant to be used along with a mesa.ContinuousSpace object.
    Therefore, it must be instantiated in a mesa.Model which contains an
    attribute called 'grid', and it must be a mesa.ContinuousSpace instance.

    """

    def __init__(self, unique_id, model, state=None,
                 step_size=None, pj=None, delta=None):
        """Creates a random walk agent.

        Inputs
        ------
        unique_id : any hashable
            Identifier for the agent.
        model : FlatRandomWalk
            Model inside which the agent is placed.
        state : str
            State of the agent for the model.
        """
        super().__init__(unique_id, model)
        # Position in grid
        self.pos = None
        self.next_pos = None
        # State (for dynamical processes)
        self.state = state
        self.next_state = state
        # Size of the spatial step
        if step_size is None:
            self.step_size = self.model.step_size
        else:
            self.step_size = step_size
        # Probability of a random repositioning
        if pj is None:
            self.pj = self.model.pj
        else:
            self.pj = self.model.pj
        # Radius of interaction
        if delta is None:
            self.delta = self.model.delta
        else:
            self.delta = delta

    def move_to(self, pos):
        """Moves the agent to position pos."""
        self.model.grid.move_agent(self, pos)

    # def get_polar_step_pos(self, d, theta):
    #     """Gets the final position of the agent after a polar
    #       step"""
    #     pass

    def polar_step_move(self, d, theta):
        """Displaces the agent by d(cos(theta), sin(theta))"""
        dx = d*np.cos(theta)
        dy = d*np.sin(theta)
        new_pos = (self.pos[0]+dx, self.pos[1]+dy)
        self.move_to(new_pos)

    def random_step_move(self, d):
        """Gives a random step of length d in space, to a random angle."""

        # Draws the uniformly distributed angle, moves the agent by it.
        theta = np.pi*(2.*rnd.random()-1.)
        self.polar_step_move(d, theta)

    def random_jump_move(self):
        """Moves the agent to a random position in space.
        Assumes that the grid has a method called random_position().
        """
        self.model.grid.move_agent(self, self.model.grid.random_position())

    def get_neighbors(self, radius):
        """Calls the grid method 'get_neighbors' to find the neighboring agents
        of self inside a circle with 'radius'.

        Inputs
        ------
        radius : float
            Radius of the circular neighborhood space around the self agent.
        """
        return self.model.grid.get_neighbors(self.pos, radius, include_center=False)

    def set_state(self, state):
        self.state = state

    def set_next_state(self, state):
        self.next_state = state

    def copy_state_to_next(self):
        self.next_state = self.state

    def update_state_from_next(self):
        self.state = self.next_state

    def step(self):
        """Agent iteration step"""
        pass

    def num_neighbors_in_state(self, state, radius):
        """Counts the number of neighboring agents in state."""
        return self.model.grid.num_neighbors_in_state(
            self.pos, radius, state
        )

    def num_neighbors_in_statelist(self, statelist, radius):
        """Counts the number of neighboring agents in state."""
        return self.model.grid.num_neighbors_in_statelist(
            self.pos, radius, statelist
        )

    def advance(self):
        """Consolidation of the iteration changes.
        Includes a random move.
        """
        # Updates epidemic state
        self.update_state_from_next()

        # Gives a random step or a random jump.
        if rnd.random() < self.pj:
            self.random_jump_move()
        else:
            self.random_step_move(self.step_size)


class FlatSpace(ContinuousSpace):

    def __init__(self, x_max, y_max, torus, x_min=0, y_min=0,
                 grid_width=100, grid_height=100):
        """ Create a new continuous space.

        Args:
            x_max, y_max: Maximum x and y coordinates for the space.
            torus: Boolean for whether the edges loop around.
            x_min, y_min: (default 0) If provided, set the minimum x and y
                          coordinates for the space. Below them, values loop to
                          the other edge (if torus=True) or raise an exception.
            grid_width, _height: (default 100) Determine the size of the
                                 internal storage grid. More cells will slow
                                 down movement, but speed up neighbor lookup.
                                 Probably only fiddle with this if one or the
                                 other is impacting your model's performance.

        """
        super().__init__(x_max, y_max, torus, x_min, y_min,
                         grid_width, grid_height)

        # This calculation is performed only once, to optimize 'num_neighbors_in_state'
        self.scale = max(self.cell_width, self.cell_height)

    def random_position(self):
        """Chooses a valid random position at the space."""
        x = self.x_min + rnd.random()*self.width
        y = self.y_min + rnd.random()*self.height
        return x, y

    def get_neighbors_in_state(self, pos, radius, state, include_center=False):
        """ Get all objects within a certain radius (modified) which are in a
        certain epidemic state.

        Args:
            pos: (x,y) coordinate tuple to center the search at.
            radius: Get all the objects within this distance of the center.
            state : string of the desired neighbor state.
            include_center: If True, include an object at the *exact* provided
                            coordinates. i.e. if you are searching for the
                            neighbors of a given agent, True will include that
                            agent in the results.

        """
        # Get candidate objects
        scale = max(self.cell_width, self.cell_height)
        cell_radius = math.ceil(radius / scale)
        cell_pos = self._point_to_cell(pos)
        possible_objs = self._grid.get_neighbors(cell_pos,
                                                 True, True, cell_radius)
        neighbors = []
        # Iterate over candidates and check actual distance.
        for obj in possible_objs:
            dist = self.get_distance(pos, obj.pos)
            if dist <= radius and (include_center or dist > 0) and obj.state == state:
                neighbors.append(obj)
        return neighbors

    def num_neighbors_in_state_old(self, pos, radius, state, include_center=False):
        """ Get the number of objects within a certain radius (modified)
        which are in a certain epidemic state.

        Args:
            pos: (x,y) coordinate tuple to center the search at.
            radius: Get all the objects within this distance of the center.
            state : string of the desired neighbor state.
            include_center: If True, include an object at the *exact* provided
                            coordinates. i.e. if you are searching for the
                            neighbors of a given agent, True will include that
                            agent in the results.

        Returns:
            neighbors: (int) number of neighbors in state.
        """
        # Get candidate objects
        scale = max(self.cell_width, self.cell_height)
        cell_radius = math.ceil(radius / scale)
        cell_pos = self._point_to_cell(pos)
        possible_objs = self._grid.get_neighbors(cell_pos,
                                                 True, True, cell_radius)
        neighbors = 0
        # Iterate over candidates and check actual distance.
        for obj in possible_objs:
            dist = self.get_distance(pos, obj.pos)
            if dist <= radius and (include_center or dist > 0) and obj.state == state:
                neighbors += 1
        return neighbors

    def num_neighbors_in_state(self, pos, radius, state):
        """ Get the number of objects within a certain radius (modified)
        which are in a certain epidemic state.

        Args:
            pos: (x,y) coordinate tuple to center the search at.
            radius: Get all the objects within this distance of the center.
            state : string of the desired neighbor state.
            # include_center: (ALWAYS FALSE FOR THIS IMPLEMENTATION)
                            If True, include an object at the *exact* provided
                            coordinates. i.e. if you are searching for the
                            neighbors of a given agent, True will include that
                            agent in the results.

        Returns:
            neighbors: (int) number of neighbors in state.
        """
        # Get candidate objects
        cell_radius = math.ceil(radius / self.scale)
        cell_pos = self._point_to_cell(pos)
        possible_objs = self._grid.get_neighbors(cell_pos,
                                                 True, True, cell_radius)
        neighbors = 0
        # Iterate over candidates and check actual distance.
        for obj in possible_objs:
            dist = self.get_distance(pos, obj.pos)
            if dist <= radius and (dist > 0.) and obj.state == state:
                neighbors += 1
        return neighbors

    def num_neighbors_in_statelist(self, pos, radius, statelist):
        """ Get the number of objects within a certain radius (modified)
        which are in a certain list of epidemic states.

        Args:
            pos: (x,y) coordinate tuple to center the search at.
            radius: Get all the objects within this distance of the center.
            statelist : list of the desired neighbor states.
            # include_center: (ALWAYS FALSE FOR THIS IMPLEMENTATION)
                            If True, include an object at the *exact* provided
                            coordinates. i.e. if you are searching for the
                            neighbors of a given agent, True will include that
                            agent in the results.

        Returns:
            neighbors: (int) number of neighbors in state.
        """
        # Get candidate objects
        cell_radius = math.ceil(radius / self.scale)
        cell_pos = self._point_to_cell(pos)
        possible_objs = self._grid.get_neighbors(cell_pos,
                                                 True, True, cell_radius)
        neighbors = 0
        # Iterate over candidates and check actual distance.
        for obj in possible_objs:
            dist = self.get_distance(pos, obj.pos)
            if dist <= radius and (dist > 0.) and obj.state in statelist:
                neighbors += 1
        return neighbors


def flat_random_walk_get_parameters(input_dict):
    """Translates the input dictionary into a set of parameters to
    initialize a FlatRandomWalk instance

    Parameters
    ----------
    input_dict : dict
        Dictionary of inputs.
    """
    n_agents = int(input_dict["num_agents"])
    x_max = float(input_dict["grid_width"])
    x_min = float(input_dict["grid_height"])
    step_size = float(input_dict["step_size"])
    pj = float(input_dict["pj"])
    delta = float(input_dict["interaction_radius"])

    return [n_agents, x_max, x_min, step_size, pj, delta]


class FlatRandomWalk(Model):

    grid = None

    def __init__(self, n_agents, x_max, y_max, step_size, jump_probab=0.,
                 interaction_radius=0.,  # Interaction radius
                 model_reporters=None, agent_reporters=None, seed=None):
        super().__init__(seed)
        # Defines agent activation scheduler and grid (space) type.
        self.schedule = SimultaneousActivation(self)
        self.grid = FlatSpace(x_max, y_max, True)

        # If reporters are not passed, they are translated to empty dicts.
        if model_reporters is None:
            model_reporters = {}
        if agent_reporters is None:
            agent_reporters = {}

        # Initializes the data collector.
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,  # Collects each agent's position
            tables={}
        )

        # Global parameters of this model
        self.step_size = step_size
        self.pj = jump_probab
        self.delta = interaction_radius

        self.place_agents_at_random(n_agents)

        # Collects the "zero-time" condition
        self.datacollector.collect(self)

    def place_agents_at_random(self, n_agents):

        # id's of the agents are set as integers.
        for i in range(n_agents):
            # Generates agent
            ag = ContinuousRandomWalker(i, self)
            # Puts agent in scheduler
            self.schedule.add(ag)
            # Puts agent in random position at the grid
            self.grid.place_agent(ag, self.grid.random_position())

    def reposition_agents_at_random(self):
        """For all the present agents, reposition them in uniformelly
        random positions at the space.
        Important: assumes that agents are already positioned at the
        grid AND the schedule.
        """
        for ag in self.agents():
            ag.random_jump_move()

    def agents(self):
        """
        Returns
        -------
        agents : list
            List of agents that are placed in the model's schedule.
        """
        return self.schedule.agents

    def num_agents(self):
        return self.schedule.get_agent_count()

    def update_all_states_from_next(self):
        """Makes state <-- next_state to all agents.

        ag : ContinuousRandomWalker
        """
        for ag in self.agents():
            ag.update_state_from_next()

    def copy_all_states_to_next(self):
        """Makes next_state <-- state to all agents.

        ag : ContinuousRandomWalker
        """
        for ag in self.agents():
            ag.copy_state_to_next()

    def num_state(self, state):
        """Counts the number of agents in a given state
        """
        count = 0
        for ag in self.agents():
            if ag.state == state:
                count += 1
        return count

    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)

    def agent_unique_ids(self):
        """Gets the list of unique_ids of the agents."""
        return [ag.unique_id for ag in self.agents()]


class SisFlatRandomWalker(ContinuousRandomWalker):

    def __init__(self, unique_id, model, state=None,
                 step_size=None, pj=None, delta=None
                 ):
        """Initializes the SisFlatRandomWalker.

        Parameters
        ----------
        unique_id : any hashable
            Identifier for the agent.
        model : FlatRandomWalk
            Model inside which the agent is placed.
        state : str
            State of the agent for the model.
        """
        super().__init__(unique_id, model, state,
                         step_size, pj, delta)
        self.beta = self.model.beta  # Infection probability
        self.mu = self.model.mu  # Healing probabililty

    def to_infect(self):
        self.set_next_state("I")

    def to_heal(self):
        self.set_next_state("S")

    def infect_now(self):
        self.set_state("I")

    def heal_now(self):
        self.set_state("S")

    def num_infected_neighbors(self):
        """Returns the number of neighbors (inside radius delta)
        that are in 'I' state."""
        return self.num_neighbors_in_state("I", self.delta)

    def infection_probab(self):
        """Returns the current probability that the agent becomes
        infected.

        $P = 1 - (1-\beta)^{n_i}$

        Where n_i is the number of infected neighbors.
        """
        n_i = self.num_infected_neighbors()
        return 1. - (1.-self.beta)**n_i

    def sis_step_reinfection(self):
        """Promotes an SIS iteration for the agent.
        Reinfection is considered as possible.
        Modifies only the "flag" (next_state), not the state.
        """
        # Infection of susceptible:
        if self.state == "S":
            if rnd.random() < self.infection_probab():
                self.to_infect()
        # Healing (and possible reinfection)
        else:
            # If it gets healed AND escapes from being infected again.
            # (nested ifs for better performance, infection_probab is expensive).
            if rnd.random() < self.mu:
                if rnd.random() > self.infection_probab():
                    self.to_heal()

    def sis_step(self):
        """Promotes an SIS iteration for the agent.
        Modifies only the "flag" (next_state), not the state.
        """
        # Infection of susceptible:
        if self.state == "S":
            if rnd.random() < self.infection_probab():
                self.to_infect()
        # Healing (and possible reinfection)
        else:
            # If it gets healed.
            if rnd.random() < self.mu:
                # if rnd.random() > self.infection_probab():
                self.to_heal()

    def step(self):
        self.sis_step()

    # def advance(self):
    #     """METHOD ADVANCE IS ALREADY WELL DEFINED IN PARENT CLASS.
    #     REDEFINE IT HERE IF THAT CHANGES"""


def sis_flat_random_walk_get_parameters(input_dict):
    """Translates the input dictionary into a set of parameters to
    initialize a SisFlatRandomWalk instance

    Parameters
    ----------
    input_dict : dict
        Dictionary of inputs.
    """
    n_agents = int(input_dict["num_agents"])
    x_max = float(input_dict["grid_width"])
    x_min = float(input_dict["grid_height"])
    step_size = float(input_dict["step_size"])
    pj = float(input_dict["pj"])
    delta = float(input_dict["interaction_radius"])
    beta = float(input_dict["beta"])
    mu = float(input_dict["mu"])

    return [n_agents, x_max, x_min, step_size, pj, delta, beta, mu]


class SisFlatRandomWalk(FlatRandomWalk):

    def __init__(self, n_agents, x_max, y_max, step_size, jump_probab=0.,
                 interaction_radius=0.,  # Interaction radius
                 beta=0.0, mu=1.0,  # Epidemic parameters
                 model_reporters=None, agent_reporters=None, seed=None):
        """Initializes a SisFlatRandomWalk class.

        The reporters (model_reporters, agent_reporters) must be functions
        of one parameter (an agent for agent reporters and a model for
        model reporters). The functions must be listed in a dictionary.

        Parameters:
            n_agents : int
                Number of agents.
            x_max : float
                Width of the rectangular space.
            y_max : float
                Height of the rectangular space.
            step_size : float
                Length of the regular spatial step taken by each agent, at each
                time step.
            jump_probab : float
                Probability that, when activated, the agent takes a jump to a
                random position at the space.
            interaction_radius : float
                Maximum distance at which the agents interact (disease
                transmission).
            beta : float
                Disease transmission probability.
            mu : float
                Disease healing probability.
            model_reporters : dict
                Dictionary of model reporters to collect general model information
                (see mesa documentation)
            agent_reporters : dict
                Dictionary of agent reporters to collect specific information
                from each agent (see mesa documentation).
            seed : int
                Seed for the random numbers.
        """
        self.beta = beta
        self.mu = mu

        super().__init__(n_agents, x_max, y_max, step_size, jump_probab,
                         interaction_radius,  # Interaction radius
                         model_reporters, agent_reporters, seed)

    def place_agents_at_random(self, n_agents):
        """Generates and places the agents in grid and schedule.

        Parameters
        ----------
        n_agents : int
            Number of agents to generate.
        """
        # id's of the agents are set as integers.
        for i in range(n_agents):
            # Generates agent
            ag = SisFlatRandomWalker(i, self)
            # Puts agent in scheduler
            self.schedule.add(ag)
            # Puts agent in random position at the grid
            self.grid.place_agent(ag, self.grid.random_position())

    def reinitialize_epidemic_states(self, infected_fraction):
        """Sets a fraction of the agents as I and the others as S."""
        for ag in self.agents():
            ag.state = "S"

        # Randomly chooses a fraction of agents to infect.
        num_i = int(infected_fraction * self.schedule.get_agent_count())
        for ag in rnd.sample(self.agents(), num_i):
            ag.state = "I"

        # Sets also the next_state flags.
        self.copy_all_states_to_next()

    def num_infected(self):
        return self.num_state("I")

    def num_susceptible(self):
        return self.num_state("S")

    def simulate_simple(self, num_steps):
        """Simulates the system for a number of steps, or until the
        number of infected agents is zero.

        Performance note: this function does not assume that the
        self.num_infected collector is used by the model. Therefore,
        it calculates the number of infected nodes at each time step.
        If num_infected is present, it means that the calculation is
        performed twice.

        Parameters
        ----------
        num_steps : int
            Number of iterations.

        Returns
        -------
        num_infected_array : np.Array
            Time series of the number of infected agents.
            The size of the array is t_max + 1, because it also stores
            the zero-time count.
        """

        # Initializes the array of infected count. Its first element is
        # read for t = 0.
        num_infected_array = np.zeros(num_steps+1, dtype=int)
        num_infected_array[0] = self.num_infected()

        for t in range(num_steps):
            # First verify if the number of infected is not zero.
            if num_infected_array[t] == 0:
                break
            # Performs one step
            self.step()
            # Calculates the number of infected nodes and stores
            num_infected_array[t+1] = self.num_infected()

        return num_infected_array

    def simulate_persistent(self, num_steps, max_trials=10):
        """Simulates the system for a number of steps.
        If the epidemic dies out, the system is reinitialized and
        simulation is re-performed. If after max_trials attempts the
        epidemics dies, the prevalence of infected agents is
        considered zero."""
        pass

# ------------------------------------------------------------
# ------------------------------------------------------------
# RUMOR MODEL(S)
# ------------------------------------------------------------
# ------------------------------------------------------------


class MTrumorFlatRandomWalker(ContinuousRandomWalker):

    def __init__(self, unique_id, model, state=None,
                 step_size=None, pj=None, delta=None
                 ):
        """Initializes the MTrumorFlatRandomWalker.

        Parameters
        ----------
        unique_id : any hashable
            Identifier for the agent.
        model : FlatRandomWalk
            Model inside which the agent is placed.
        state : str
            State of the agent for the model.
        """
        super().__init__(unique_id, model, state,
                         step_size, pj, delta
                         )
        self.beta = self.model.beta  # Infection probability
        self.mu = self.model.mu  # Healing probabililty

    def to_infect(self):
        self.set_next_state("I")

    def to_heal(self):
        self.set_next_state("R")

    def infect_now(self):
        self.set_state("I")

    def heal_now(self):
        self.set_state("R")

    def num_infected_neighbors(self):
        """Returns the number of neighbors (inside radius delta)
        that are in 'I' state."""
        return self.num_neighbors_in_state("I", self.delta)

    def num_infected_and_stifler_neighbors(self):
        return self.num_neighbors_in_statelist(["I", "R"], self.delta)

    def infection_probab(self):
        """Returns the current probability that the agent becomes
        infected.

        $P = 1 - (1-\beta)^{n_i}$

        Where n_i is the number of infected neighbors.
        """
        n_i = self.num_infected_neighbors()
        return 1. - (1.-self.beta)**n_i

    def stiflering_probab(self):
        """Returns the current probability that the agent becomes
        stifler, considering MT.

        $P = 1 - (1-\mu)^{n_i}$

        Where n_i is the number of infected or stifler neighbors.
        """
        n_i = self.num_infected_and_stifler_neighbors()
        return 1. - (1.-self.mu)**n_i

    def sir_step(self):
        """Promotes an SIR epidemic iteration for the agent.
        Modifies only the "flag" (next_state), not the state.
        """
        # Infection of susceptible:
        if self.state == "S":
            if rnd.random() < self.infection_probab():
                self.to_infect()
        # Healing
        elif self.state == "I":
            # If it gets healed.
            if rnd.random() < self.mu:
                self.to_heal()

    def mtrumor_step(self):
        """Promotes a Maki-Thomson rumor iteration for the agent.
        Modifies only the "flag" (next_state), not the state.
        """
        # Infection of susceptible:
        if self.state == "S":
            if rnd.random() < self.infection_probab():
                self.to_infect()
        # Healing
        elif self.state == "I":
            # If it gets healed.
            if rnd.random() < self.stiflering_probab():
                self.to_heal()

    def step(self):
        self.mtrumor_step()
        # self.sir_step()

    # def advance(self):
    #     """METHOD ADVANCE IS ALREADY WELL DEFINED IN PARENT CLASS.
    #     REDEFINE IT HERE IF THAT CHANGES"""


def mtrumor_flat_random_walk_get_parameters(input_dict):
    """Translates the input dictionary into a set of parameters to
    initialize an MTrumorFlatRandomWalk instance

    Parameters
    ----------
    input_dict : dict
        Dictionary of inputs.
    """
    n_agents = int(input_dict["num_agents"])
    x_max = float(input_dict["grid_width"])
    x_min = float(input_dict["grid_height"])
    step_size = float(input_dict["step_size"])
    pj = float(input_dict["pj"])
    delta = float(input_dict["interaction_radius"])
    beta = float(input_dict["beta"])
    mu = float(input_dict["mu"])

    return [n_agents, x_max, x_min, step_size, pj, delta, beta, mu]


class MTrumorFlatRandomWalk(FlatRandomWalk):

    def __init__(self, n_agents, x_max, y_max,
                 step_size=0., jump_probab=0.,
                 interaction_radius=0.,  # Interaction radius
                 beta=0.0, mu=1.0,  # Epidemic parameters
                 model_reporters=None, agent_reporters=None, seed=None):
        """Initializes an MTrumorFlatRandomWalk class.

        The reporters (model_reporters, agent_reporters) must be functions
        of one parameter (an agent for agent reporters and a model for
        model reporters). The functions must be listed in a dictionary.

        Parameters:
            n_agents : int
                Number of agents.
            x_max : float
                Width of the rectangular space.
            y_max : float
                Height of the rectangular space.
            step_size : float
                Length of the regular spatial step taken by each agent, at each
                time step.
            jump_probab : float
                Probability that, when activated, the agent takes a jump to a
                random position at the space.
            interaction_radius : float
                Maximum distance at which the agents interact (disease
                transmission).
            beta : float
                Disease/rumor transmission probability.
            mu : float
                Disease/rumor healing probability.
            model_reporters : dict
                Dictionary of model reporters to collect general model information
                (see mesa documentation)
            agent_reporters : dict
                Dictionary of agent reporters to collect specific information
                from each agent (see mesa documentation).
            seed : int
                Seed for the random numbers.
        """
        self.beta = beta
        self.mu = mu

        super().__init__(n_agents, x_max, y_max, step_size, jump_probab,
                         interaction_radius,  # Interaction radius
                         model_reporters, agent_reporters, seed)

    def place_agents_at_random(self, n_agents):
        """Generates and places the agents in grid and schedule.

        Parameters
        ----------
        n_agents : int
            Number of agents to generate.
        """
        # id's of the agents are set as integers.
        for i in range(n_agents):
            # Generates agent
            ag = MTrumorFlatRandomWalker(i, self)
            # Puts agent in scheduler
            self.schedule.add(ag)
            # Puts agent in random position at the grid
            self.grid.place_agent(ag, self.grid.random_position())

    def reinitialize_epidemic_states(self, infected_fraction):
        """Sets a fraction of the agents as I and the others as S."""
        for ag in self.agents():
            ag.state = "S"

        # Randomly chooses a fraction of agents to infect.
        num_i = int(infected_fraction * self.schedule.get_agent_count())
        for ag in rnd.sample(self.agents(), num_i):
            ag.state = "I"

        # Sets also the next_state flags.
        self.copy_all_states_to_next()

    def num_infected(self):
        return self.num_state("I")

    def num_susceptible(self):
        return self.num_state("S")

    def num_stifler(self):
        return self.num_state("R")

    def simulate_simple(self, num_steps):
        """
        !!!!!!!!!!!!!!NEEED TO TEST THIS FUNCTION!!!!!!!!!!!!!!!!!!!!
        Simulates the system for a number of steps, or until the
        number of infected agents is zero.

        Performance note: this function does not assume that the
        self.num_infected collector is used by the model. Therefore,
        it calculates the number of infected nodes at each time step.
        If num_infected is present, it means that the calculation is
        performed twice.

        Parameters
        ----------
        num_steps : int
            Number of iterations.

        Returns
        -------
        num_infected_array : np.Array
            Time series of the number of infected agents.
            The size of the array is t_max + 1, because it also stores
            the zero-time count.
        """

        # Initializes the array of infected count. Its first element is
        # read for t = 0.
        num_infected_array = np.zeros(num_steps+1, dtype=int)
        num_infected_array[0] = self.num_infected()

        # Array of recovered/stifler counts
        num_stifler_array = np.zeros(num_steps+1, dtype=int)
        num_stifler_array[0] = self.num_stifler()

        for t in range(num_steps):
            # First verify if the number of infected is not zero.
            if num_infected_array[t] == 0:
                # If the simulation is over, fixes the number of stiflers until the end.
                num_stifler = self.num_stifler()
                for u in range(t, num_steps):
                    num_stifler_array[t+1] = num_stifler

                break

            # Performs one step
            self.step()
            # Calculates the number of infected and stilfer nodes and stores
            num_infected_array[t+1] = self.num_infected()
            num_stifler_array[t+1] = self.num_stifler()

        return num_infected_array, num_stifler_array

    def simulate_until_dies(self):
        """Simulates the system until the number of infected agents
        is zero.

        Returns only the number of healed/stifler agents at the last step.

        Returns
        -------
        num_infected_array : np.Array
            Time series of the number of infected agents.
            The size of the array is t_max + 1, because it also stores
            the zero-time count.
        """
        num_infected = self.num_infected()

        while num_infected > 0:
            self.step()
            num_infected = self.num_infected()

        return self.num_stifler()

    def simulate_and_retry(self, infected_fraction, max_t=1000, max_trials=5,
                           return_numsteps=False):
        """Tries to simulate the model. If it does not stop after max_t
        iterations, resets the epidemic and positional state of the agents
        and tries again. Stops and returns the number of infected plus
        recovered/stifler agents after max_trials resets.

        If parameter return_numsteps is True, returns also the number of
        steps and trials that were executed.
        """

        # Avoids dumb warning by the PyCharm interpreter...
        num_infected = self.num_infected()
        num_stifler = self.num_stifler()

        for n in range(max_trials):
            for t in range(max_t):
                # Runs one simulation step and calculates the infected number.
                self.step()
                num_infected = self.num_infected()
                num_stifler = self.num_stifler()
                # Checks for 'death' of the process:
                if num_infected == 0:
                    if return_numsteps:
                        return self.num_stifler(), t+1, n+1
                    else:
                        return self.num_stifler()

            # After max_t is reached, reinitializes the simulation.
            self.reinitialize_epidemic_states(infected_fraction)
            self.reposition_agents_at_random()

        # If max_trials simulations are attempted, returns the number of
        # infected + stiflers for the last time step.
        print("[Process was not over after {} rounds of {} steps]"
              "".format(max_trials, max_t))
        if return_numsteps:
            return num_infected + num_stifler, max_t, max_trials
        else:
            return num_infected + num_stifler


# ------------------------------------------------------------
# ------------------------------------------------------------
# SIR MODEL(S)
# ------------------------------------------------------------
# ------------------------------------------------------------


class SirFlatRandomWalker(MTrumorFlatRandomWalker):

    def step(self):
        # self.mtrumor_step()
        self.sir_step()


def sir_flat_random_walk_get_parameters(input_dict):
    """Translates the input dictionary into a set of parameters to
    initialize an MTrumorFlatRandomWalk instance

    Parameters
    ----------
    input_dict : dict
        Dictionary of inputs.
    """
    n_agents = int(input_dict["num_agents"])
    x_max = float(input_dict["grid_width"])
    x_min = float(input_dict["grid_height"])
    step_size = float(input_dict["step_size"])
    pj = float(input_dict["pj"])
    delta = float(input_dict["interaction_radius"])
    beta = float(input_dict["beta"])
    mu = float(input_dict["mu"])

    return [n_agents, x_max, x_min, step_size, pj, delta, beta, mu]


class SirFlatRandomWalk(MTrumorFlatRandomWalk):

    def place_agents_at_random(self, n_agents):
        """Generates and places the agents in grid and schedule.

        Parameters
        ----------
        n_agents : int
            Number of agents to generate.
        """
        # id's of the agents are set as integers.
        for i in range(n_agents):
            # Generates agent
            ag = SirFlatRandomWalker(i, self)
            # Puts agent in scheduler
            self.schedule.add(ag)
            # Puts agent in random position at the grid
            self.grid.place_agent(ag, self.grid.random_position())

    def num_recovered(self):
        return self.num_stifler()

# ------------------------------------------------------------
# ------------------------------------------------------------
# RUMOR MODELS WITH INHOMOGENEOUS PARAMETERS
# ------------------------------------------------------------
# ------------------------------------------------------------


def mtrumor_flat_radvar_random_walk_get_parameters(input_dict):
    """Translates the input dictionary into a set of parameters to
    initialize an MTrumorFlatRandomWalk instance

    Parameters
    ----------
    input_dict : dict
        Dictionary of inputs.
    """
    n_agents = int(input_dict["num_agents"])
    x_max = float(input_dict["grid_width"])
    x_min = float(input_dict["grid_height"])
    step_size = float(input_dict["step_size"])
    pj = float(input_dict["pj"])
    # delta = float(input_dict["interaction_radius"])  # Replaced by distrib
    beta = float(input_dict["beta"])
    mu = float(input_dict["mu"])

    # Generates the distribution function.
    distrib = random_function_generator_from_dict(
        input_dict,
        prefix="radius_"
    )

    return [n_agents, x_max, x_min, distrib, step_size, pj, beta, mu]


# Variable radius of interaction (delta).
class MTrumorFlatRadvarRandomWalk(MTrumorFlatRandomWalk):
    """Maki-Thompson rumor model with inhomogeneous radius of interaction."""

    def __init__(self, n_agents, x_max, y_max, radius_distrib,
                 step_size=0., jump_probab=0.,
                 # interaction_radius=0.,  # Interaction radius - Replaced by distrib
                 beta=0.0, mu=1.0,
                 model_reporters=None, agent_reporters=None, seed=None):

        # Calls the initializator of the mesa.Model class
        Model.__init__(self, seed)

        # Defines agent activation scheduler and grid (space) type.
        self.schedule = SimultaneousActivation(self)
        self.grid = FlatSpace(x_max, y_max, True)

        # If reporters are not passed, they are translated to empty dicts.
        if model_reporters is None:
            model_reporters = {}
        if agent_reporters is None:
            agent_reporters = {}

        # Initializes the data collector.
        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters,  # Collects each agent's position
            tables={}
        )

        # Global parameters of this model
        self.step_size = step_size
        self.pj = jump_probab
        self.beta = beta
        self.mu = mu

        # Distribution function of the interaction_radius (delta)
        self.radius_distrib = radius_distrib
        # Generates the sequence of values of delta, indexed the
        # same way as 'agents'
        self.delta_array = radius_distrib(n_agents)

        # Places the agents at the space.
        self.place_agents_at_random(n_agents)

        # Collects the "zero-time" condition
        self.datacollector.collect(self)

    def place_agents_at_random(self, n_agents):
        """Generates and places the agents in grid and schedule.

        Parameters
        ----------
        n_agents : int
            Number of agents to generate.
        """
        # Generates the sequence of

        # id's of the agents are set as integers.
        for i in range(n_agents):
            # Generates agent (with random radius of interaction)
            ag = MTrumorFlatRandomWalker(i, self,
                                         delta=self.delta_array[i])
            # Puts agent in scheduler
            self.schedule.add(ag)
            # Puts agent in random position at the grid
            self.grid.place_agent(ag, self.grid.random_position())

    def regenerate_all_deltas(self):
        """ Draws again all the values of the interaction radius, for each agent."""
        # Takes n_agents random numbers from the distribution function.
        self.delta_array = self.radius_distrib(self.num_agents())

        # Assigns the drawn values to the agents' radii of iteration.
        for i, ag in enumerate(self.agents()):
            ag.delta = self.delta_array[i]
