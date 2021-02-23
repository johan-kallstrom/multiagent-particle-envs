import numpy as np
from multiagent.core import World, Agent, AccelerationAction, FireAction, Landmark, Sensor, Rwr
from multiagent.scenario import BaseScenario

from gym import spaces

class LandMarkObservation:

    def __init__(self, n_landmarks, agent, world):
        self.n_landmarks = n_landmarks
        self.observation_space = spaces.Box(low=-100.0, high=+100.0, shape=(n_landmarks*2,), dtype=np.float32)

    def observation(self, agent, world):
        entity_pos = np.zeros(shape=(self.n_landmarks*2,))
        for i, detection in enumerate(agent.sensor.detections):
            pos = ((detection.state.p_pos - agent.state.p_pos) / (world.position_scale))
            entity_pos[i*2] = pos[0]
            entity_pos[i*2+1] = pos[1]
        return entity_pos

class Scenario(BaseScenario):
    def make_world(self):
        world = World(is_dynamic=False, position_scale=200000.0)
        world.discrete_action_space = True
        world.dt = 1.0
        n_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(1)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.fusioned_sa = LandMarkObservation(n_landmarks, agent, world)
            agent.platform_action = AccelerationAction()
            agent.fire_action = FireAction(2)
            agent.max_speed = 700.0
            agent.min_speed = 0.25 * agent.max_speed
            agent.accel = [1.0*9.81, 8]
            agent.sensor = Sensor([2 * np.pi / 3], [100000.0], [2.5e-5])
        # add landmarks
        world.landmarks = [Landmark() for i in range(n_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
            # landmark.rwr = Rwr(max_range=200000.0, min_range=2.5e-5)
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
        world.landmarks[0].color = np.array([0.75,0.25,0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1*world.position_scale ,+1*world.position_scale , world.dim_p)
            agent.state.p_vel = np.ones(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.missiles = agent.state.missiles_loaded
            agent.state.missiles_in_flight = []
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1*world.position_scale,+1*world.position_scale, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        # reset steps
        world.steps = 0

    def reward(self, agent, world):
        rew = -0.1
        if len(agent.sensor.detections) > 0:
            rew += 10
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append((entity.state.p_pos - agent.state.p_pos) / (2*world.position_scale))
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def done(self, agent, world):
        return (len(agent.sensor.detections)) or (world.steps > 300)
