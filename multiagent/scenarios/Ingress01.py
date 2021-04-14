import numpy as np
from multiagent.core import World, Agent, Entity, HeadingAction, RouteAction, FireAction, Landmark, Sensor, Rwr
from multiagent.scenario import BaseScenario

from gym import spaces

class LandMarkObservation:

    def __init__(self, n_landmarks, agent, world):
        self.n_landmarks = n_landmarks
        self.observation_space = spaces.Box(low=-100.0, high=+100.0, shape=(n_landmarks*2 + 2,), dtype=np.float32)

    def observation(self, agent, world):
        obs = np.zeros(shape=(self.n_landmarks*2 + 2,))
        for i, entity in enumerate(world.landmarks):
            entity_pos = ((entity.state.p_pos - agent.state.p_pos) / (world.position_scale))
            obs[i*2] = entity_pos[0]
            obs[i*2+1] = entity_pos[1]
        # vel_norm = agent.state.p_vel / np.linalg.norm(agent.state.p_vel)
        vel_norm = agent.state.p_vel / 350.0
        obs[-2] = vel_norm[0]
        obs[-1] = vel_norm[1]
        
        return obs

class Scenario(BaseScenario):
    def make_world(self):
        world = World(is_dynamic=False, position_scale=50000.0)
        world.discrete_action_space = True
        world.dt = 1.0
        n_landmarks = 1
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.fusioned_sa = LandMarkObservation(n_landmarks, agent, world)
            agent.max_speed = 350.0 # 700.0
            agent.min_speed = 0.8 * agent.max_speed
            agent.accel = [1.0*9.81, 8.0]
            agent.sensor = Sensor([1 * np.pi / 3], [20000.0], [2.5e-5])
            agent.fire_action = FireAction(2)
            if i == 0:
                route = [Landmark() for i in range(2)]
                route[0].state.p_pos = np.array([-10000.0, 0])
                route[0].state.p_vel = np.zeros(world.dim_p)
                route[1].state.p_pos = np.array([10000.0, 0])
                route[1].state.p_vel = np.zeros(world.dim_p)
                agent.platform_action = RouteAction(route=route)
            else:
                agent.platform_action = HeadingAction()
        # add landmarks
        world.landmarks = [Landmark()]
        world.landmarks[0].name = 'target_landmark'
        world.landmarks[0].collide = False
        world.landmarks[0].movable = False
        world.landmarks[0].state.p_pos = np.array([0.0, 0.0])
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            rgb = np.array([0.75,0.75,0.75])
            agent.color = np.array([0.0,0.0,0.0])
            agent.color[np.minimum(i,1)] = 1.0
            agent.color = agent.color * rgb
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
        # world.landmarks[0].color = np.array([0.75,0.25,0.25])
        # set random initial states
        for i, agent in enumerate(world.agents):
            if i == 0:
                pos_x = np.random.uniform(-5000.0, 5000.0)
                vel_x = np.random.choice([-agent.max_speed, agent.max_speed])
                agent.state.p_pos = np.array([pos_x, 0.0])
                agent.state.p_vel = np.array([vel_x, 0.0])
            else:
                agent.state.p_pos = np.array([50000.0, 0.0])
                agent.state.p_vel = np.array([-agent.max_speed, 0.0])
            agent.state.c = np.zeros(world.dim_c)
            agent.state.missiles = agent.state.missiles_loaded
            agent.state.missiles_in_flight = []
        # reset steps
        world.steps = 0

    def reward(self, agent, world):
        rew = 0.0
        # rew = -0.1
        # if len(agent.sensor.detections) > 0:
        #     rew += 10
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos)))/(world.position_scale) for a in world.agents]
            # dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        # print(rew)
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append((entity.state.p_pos - agent.state.p_pos) / (2*world.position_scale))
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def done(self, agent, world):
        # return (len(agent.sensor.detections)) or (world.steps > 300)
        return (world.steps > 300)
