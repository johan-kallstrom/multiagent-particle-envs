import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 2
        num_adversaries = 1
        num_landmarks = 5
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            if i < num_adversaries:
                agent.adversary = True
                agent.color = np.array([0.75, 0.25, 0.25])
            else:
                agent.adversary = False
                agent.color = np.array([0.25, 0.25, 0.75])
        # add landmarks for goal posts and puck
        goal_posts = [[-0.25, -1.0],
                      [-0.25, 1.0],
                      [0.25, -1.0],
                      [0.25, 1.0]]
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            if i > 0:
                landmark.collide = True
                landmark.movable = False
                landmark.state.p_pos = np.array(goal_posts[i-1])
                landmark.state.p_vel = np.zeros(world.dim_p)
            else:
                landmark.collide = True
                landmark.movable = True
        # add landmarks for rink boundary
        #world.landmarks += self.set_boundaries(world)
        # make initial conditions
        self.reset_world(world)
        return world

    def set_boundaries(self, world):
        boundary_list = []
        landmark_size = 1
        edge = 1 + landmark_size
        num_landmarks = int(edge * 2 / landmark_size)
        for x_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([x_pos, -1 + i * landmark_size])
                boundary_list.append(l)

        for y_pos in [-edge, edge]:
            for i in range(num_landmarks):
                l = Landmark()
                l.state.p_pos = np.array([-1 + i * landmark_size, y_pos])
                boundary_list.append(l)

        for i, l in enumerate(boundary_list):
            l.name = 'boundary %d' % i
            l.collide = True
            l.movable = False
            l.boundary = True
            l.color = np.array([0.75, 0.75, 0.75])
            l.size = landmark_size
            l.state.p_vel = np.zeros(world.dim_p)

        return boundary_list

    def reset_world(self, world):
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            if i > 0:
                landmark.color = np.array([0.7, 0.7, 0.7])
            else:
                landmark.color = np.array([0.1, 0.1, 0.1])
            landmark.index = i
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        world.landmarks[0].state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

    # return all agents of the blue team
    def blue_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all agents of the red team
    def red_agents(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on team they belong to
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # reward for blue team agent
        return 0.0

    def adversary_reward(self, agent, world):
        # reward for red team agent
        return 0.0
               
    def observation(self, agent, world):
        # get positions/vel of all entities in this agent's reference frame
        entity_pos = []
        entity_vel = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
            if entity.movable:
                entity_vel.append(entity.state.p_vel)
        # get positions/vel of all other agents in this agent's reference frame
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + entity_pos + entity_vel + other_pos + other_vel)
