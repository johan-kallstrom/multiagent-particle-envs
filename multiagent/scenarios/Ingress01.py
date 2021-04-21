import numpy as np
from multiagent.core import World, Agent, Entity, HeadingAction, RouteAction, FireAction, Landmark, Sensor, Rwr
from multiagent.scenario import BaseScenario

from gym import spaces

class IngressObservation:

    def __init__(self, agent, world):
        n_agents = len(world.agents)
        self.observation_space = spaces.Box(low=-100.0, 
                                            high=+100.0, 
                                            shape=(n_agents*4 + 2,), # pos and vel for agents, pos for target
                                            dtype=np.float32)

    def observation(self, agent, world):
        # get positions of target in this agent's reference frame
        tgt_pos = (world.landmarks[0].state.p_pos - agent.state.p_pos) / world.position_scale
        # get positions and velocities of other agents in this agent's reference frame
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append((other.state.p_pos - agent.state.p_pos) / world.position_scale)
            other_vel.append(other.state.p_vel / other.max_speed)
        return np.concatenate([agent.state.p_pos / world.position_scale] + [agent.state.p_vel / agent.max_speed] + [tgt_pos] + other_pos + other_vel)

class Scenario(BaseScenario):
    def make_world(self, random_starts=False, done_on_detection=False):
        world = World(is_dynamic=False, position_scale=50000.0)
        world.discrete_action_space = True
        world.dt = 1.0
        self.random_starts = random_starts
        self.done_on_detection = done_on_detection
        # add agents
        world.agents = [Agent() for i in range(2)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.fusioned_sa = IngressObservation(agent, world)
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
        # add landmark for target
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
                if self.random_starts:
                    heading = np.random.random()
                    agent.state.p_pos = 50000.0 * np.array([np.sin(heading), np.cos(heading)])
                    agent.state.p_vel = -agent.max_speed * np.array([np.sin(heading), np.cos(heading)])
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
        if (not agent is world.agents[0]) and self.done(agent, world):
            distance_to_target = np.linalg.norm(world.landmarks[0].state.p_pos - agent.state.p_pos)
            start_distance = np.linalg.norm(world.landmarks[0].state.p_pos - np.array([50000.0, 0.0]))
            rew = (start_distance - distance_to_target) / world.position_scale

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append((entity.state.p_pos - agent.state.p_pos) / (2*world.position_scale))
        return np.concatenate([agent.state.p_vel] + entity_pos)

    def done(self, agent, world):
        # return (len(agent.sensor.detections)) or (world.steps > 300)
        done = (world.steps >= 300)
        if self.done_on_detection:
            detected = len(agent.sensor.detections)
            done = done or detected

        return done
