import numpy as np
from gym import spaces

# physical/external base state of all entites
class EntityState(object):
    def __init__(self,
                 missiles_loaded=6,
                 is_dynamic=True):
        self.is_dynamic = is_dynamic
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        # indication of being observed
        self.observed = False
        # missiles
        self.missiles_loaded = missiles_loaded
        self.missiles = missiles_loaded
        self.missiles_in_flight = []

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# action of the agent
class VelocityAction(object):
    def __init__(self):
        # physical action (speed and heading)
        self.u = None
        # action space: Speed [min to max] and Heading relative north (y axis) [between -pi and pi]
        self.action_space = spaces.Box(low=np.array([0.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

    def set_action(self, action):
        self.u = action

    def update_entity_state(self, agent, world):
        if not agent.movable: return
        dt = world.dt
        speed_delta = (agent.max_speed - agent.min_speed) / 2
        speed = agent.platform_action.u[0] * speed_delta + (agent.min_speed + speed_delta)
        heading = agent.platform_action.u[1] * np.pi
        agent.state.p_vel[0] = speed * np.sin(heading)
        agent.state.p_vel[1] = speed * np.cos(heading)
        agent.state.p_pos += agent.state.p_vel * dt     

# thrust and turn acceleration command
class AccelerationAction(object):
    def __init__(self):
        # physical action (acceleration)
        self.u = np.array([0.0, 0.0])
        self.north = np.array([0.0, 1.0])
        # action space: Acceleration for thrust and load factor [min to max]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

    def set_action(self, action):
        self.u = action

    def update_entity_state(self, agent, world):
        if not agent.movable: return
        dt = world.dt
        T = self.u[0] * agent.accel[0] # commanded thrust TODO: Remove symmetry forward/backward
        if self.u[1] > 0:              # commanded load factor
            n = 1.0 + self.u[1] * agent.accel[1]
        else:
            n = -1.0 + self.u[1] * agent.accel[1]

        # effect of throttle action
        speed = np.linalg.norm(agent.state.p_vel) + T * dt
        speed = np.minimum(np.maximum(speed, agent.min_speed), agent.max_speed)

        # effect of turn action
        w = np.sign(n) * 9.81 * np.sqrt(n**2 - 1) / speed # turn rate
        cos_heading = np.dot(agent.state.p_vel, self.north) / np.linalg.norm(agent.state.p_vel)
        heading = np.sign(agent.state.p_vel[0]) * np.arccos(cos_heading) + w * dt
        if heading > np.pi:
            heading = heading - 2 * np.pi
        elif heading < -np.pi:
            heading = heading  + 2 * np.pi

        # update velocity and position
        agent.state.p_vel[0] = speed * np.sin(heading)
        agent.state.p_vel[1] = speed * np.cos(heading)
        agent.state.p_pos += agent.state.p_vel * dt

# thrust and heading command
class HeadingAction(object):
    def __init__(self):
        self.u = None
        self.target_dist = 100000.0
        self.north = np.array([0.0, 1.0])
        # action space: Command thrust [min to max] and heading [-pi and +pi]
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)

        # use acceleration action for control
        self.Kp = 1 / np.pi
        self.acc_action = AccelerationAction()

    def set_action(self, action):
        self.u = action
        
    def update_entity_state(self, agent, world):
        # set turn agent based on deviation from desired heading
        tgt_heading = self.u[1] * np.pi
        tgt_vector = np.array([np.sin(tgt_heading), np.cos(tgt_heading)], dtype=np.float32) # unit vector

        # angle error according to dot product between tgt direction and current direction
        cos_angle_error = np.dot(tgt_vector, agent.state.p_vel) / (np.linalg.norm(tgt_vector) + np.linalg.norm(agent.state.p_vel))
        angle_error = np.arccos(cos_angle_error)

        # turn direction according to determinant between target direction and current direction
        turn_direction = np.sign(tgt_vector[0] * agent.state.p_vel[1] - tgt_vector[1] * agent.state.p_vel[0])

        # calculate turn action using P controller
        turn_action = self.Kp * turn_direction * angle_error

        self.acc_action.set_action(np.array([self.u[0], turn_action]))
        self.acc_action.update_entity_state(agent, world)

# missile PN guidance
class PnGuidanceAction(object):
    def __init__(self,
                 entity,
                 target,
                 max_acc,
                 N=5):
        self.target = target
        self.max_acc = max_acc
        self.N = N
        self.north = np.array([0.0, 1.0])      
        self.tgt_last_los = target.state.p_pos - entity.state.p_pos

    def update_entity_state(self, entity, world):
        # if not entity.movable: return
        dt = world.dt
        
        cur_speed = np.linalg.norm(entity.state.p_vel)
        tgt_cur_los = self.target.state.p_pos - entity.state.p_pos

        # check line-of-sight rotation relative north
        cos_lambda_last = np.dot(self.tgt_last_los, self.north) / np.linalg.norm(self.tgt_last_los)
        lambda_last = np.sign(self.tgt_last_los[0]) * np.arccos(cos_lambda_last)
        cos_lambda_cur = np.dot(tgt_cur_los, self.north) / np.linalg.norm(tgt_cur_los)
        lambda_cur = np.sign(tgt_cur_los[0]) * np.arccos(cos_lambda_cur)

        lambda_delta = lambda_cur - lambda_last
        if lambda_delta > np.pi:
            lambda_delta = lambda_delta - 2 * np.pi
        elif lambda_delta < -np.pi:
            lambda_delta = lambda_delta  + 2 * np.pi

        lambda_dot = lambda_delta / dt

        # check closing speed
        Vc = (np.linalg.norm(tgt_cur_los) - np.linalg.norm(self.tgt_last_los)) / dt
        speed_c = np.linalg.norm(Vc)

        # calculate acceleration normal to current velocity
        a_n = np.minimum(self.N * lambda_dot * speed_c, self.max_acc)

        # rotate acceleration vector ([a_n, 0] for heading = 0) based on current heading
        cos_heading = np.dot(entity.state.p_vel, self.north) / np.linalg.norm(entity.state.p_vel)
        heading = np.sign(entity.state.p_vel[0]) * np.arccos(cos_heading)

        a_n_rot = np.array([a_n * np.cos(-heading), a_n * np.sin(-heading)])

        # update velocity and position
        entity.state.p_vel += a_n_rot * dt
        entity.state.p_vel = entity.state.p_vel / np.linalg.norm(entity.state.p_vel) * cur_speed # keep speed constant
        entity.state.p_pos += entity.state.p_vel * dt

        # store values for next update
        self.tgt_last_los = tgt_cur_los

# TODO: Weapon, Sensor and Electronic Warfare control actions

# action of the agent
class FireAction(object):
    def __init__(self, 
                 n_targets): # number of targets in the environment
        # fire action (fire (u > 0) or not (u = 0))
        self.u = None
        # action space
        self.action_space = spaces.Discrete(n_targets+1)

    def set_action(self, action):
        self.u = action

    def update_entity_state(self, entity, world):
        if entity.state.missiles > 0 and self.u > 0 and entity.sensor is not None and len(entity.sensor.detections) >= self.u:
            missile = Missile(entity.sensor.detections[self.u-1], init_pos=entity.state.p_pos.copy(), init_vel=entity.state.p_vel.copy(), world=world) # fire missile
            entity.state.missiles_in_flight.append(missile)
            entity.state.missiles -= 1
        live_missiles = []
        for missile in entity.state.missiles_in_flight:
            missile.update_state(world)
            if not missile.destroyed:
                live_missiles.append(missile)
        entity.state.missiles_in_flight = live_missiles

# sensor object for agents wit limited vision
class Sensor(object):
    def __init__(self, sensor_fovs, max_sensor_ranges, min_sensor_ranges, default_mode=0, sensor_heading=None):
        self.sensor_fovs = sensor_fovs
        self.max_sensor_ranges = max_sensor_ranges
        self.min_sensor_ranges = min_sensor_ranges
        self.fov = sensor_fovs[default_mode]
        self.max_range = max_sensor_ranges[default_mode]
        self.min_range = min_sensor_ranges[default_mode]
        self.heading = sensor_heading
        self.detections = []
        self.set_mode(default_mode)

    def set_mode(self, mode):
        self.fov = self.sensor_fovs[mode]
        self.max_range = self.max_sensor_ranges[mode]
        self.min_range = self.min_sensor_ranges[mode]

    def check_detection(self, entity, other):
        if self.other_in_range(entity, other) and self.other_in_fov(entity, other):
            self.detections.append(other)
            # other.state.observed = True # TODO: make this less of a strange side effect

    def other_in_range(self, entity, other):
        distance = np.sqrt(np.sum(np.square(entity.state.p_pos - other.state.p_pos)))
        return (distance <= self.max_range) and (distance >= self.min_range)

    def other_in_fov(self, entity, other):
        e_to_ot_vector = other.state.p_pos-entity.state.p_pos
        dot_product = np.dot(entity.state.p_vel, e_to_ot_vector)
        norms_product = np.linalg.norm(entity.state.p_vel) * np.linalg.norm(e_to_ot_vector)
        angle = np.arccos(dot_product / norms_product)
        return angle < self.fov / 2

# radar warner
class Rwr(object):
    def __init__(self, max_range, min_range):
        self.max_range = max_range
        self.min_range = min_range
        self.observers = []

    def check_observed(self, entity, other):
        if self.im_in_range(entity, other) and self.im_in_fov(entity, other):
            self.observers.append(other)

    def im_in_range(self, entity, other):
        distance = np.sqrt(np.sum(np.square(entity.state.p_pos - other.state.p_pos)))
        return (distance <= self.max_range) and (distance >= self.min_range)

    def im_in_fov(self, entity, other):
        ot_to_e_vector = entity.state.p_pos - other.state.p_pos
        dot_product = np.dot(ot_to_e_vector, other.state.p_vel)
        norms_product = np.linalg.norm(other.state.p_vel) * np.linalg.norm(ot_to_e_vector)
        angle = np.arccos(dot_product / norms_product)
        return angle < other.sensor.fov / 2

# properties and state of physical world entity
class Entity(object):
    def __init__(self, is_dynamic=True):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color and rendering
        self.color = None
        self.onetime_render = False
        # max/min speed and accel
        self.max_speed = None
        self.min_speed = None
        self.accel = None
        # state
        self.state = EntityState(is_dynamic=is_dynamic)
        # mass
        self.initial_mass = 1.0
        # sensor
        self.sensor = None
        # radar warner
        self.rwr = None

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of missile entities,
# scenario should define sensor and its properties
class Missile(Entity):
    def __init__(self, target, init_pos, init_vel, world):
        super(Missile, self).__init__()
        self.target = target
        self.lethal_range = self.size * world.position_scale
        self.color = np.array([0.25,0.25,0.25])
        self.onetime_render = True
        self.size = 0.25 * self.size
        self.destroyed = False
        self.state.p_pos = init_pos
        self.state.p_vel = 2 * init_vel # TODO: COnfigure speed
        self.guidance = PnGuidanceAction(self,
                                         target,
                                         max_acc=9*9.82)

    def update_state(self, world):
        self.guidance.update_entity_state(self, world)
        m_to_tgt = self.target.state.p_pos - self.state.p_pos
        if np.linalg.norm(m_to_tgt) < self.lethal_range:
            self.destroyed = True

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # actions
        # ------------------------------------------------------------------------------------------
        self.action = Action()          # standard particle envs force and communication actions
        # ------------------------------------------------------------------------------------------
        self.platform_action = None     # Tuple action space: Action for platform control
        self.com_action = None          # Tuple action space: Action for communication among agents
        self.sensor_action = None       # Tuple action space: Action for sensor control
        self.ew_action = None           # Tuple action space: Action for EW control
        self.fire_action = None         # Tuple action space: Action for weapon fire control
        # ------------------------------------------------------------------------------------------
        # situational awareness
        # ------------------------------------------------------------------------------------------
        self.fusioned_sa = None
        # ------------------------------------------------------------------------------------------
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self, is_dynamic=True, position_scale=1.0):
        # determines if forces are used to update world state
        self.is_dynamic = is_dynamic
        self.position_scale = position_scale
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.missiles = []
        # communication channel dimensionality
        self.dim_c = 0
        # physical control dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # episode step
        self.steps = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        if self.is_dynamic:
            # gather forces applied to entities
            p_force = [None] * len(self.entities)
            # apply agent physical controls
            p_force = self.apply_action_force(p_force)
            # apply environment forces
            p_force = self.apply_environment_force(p_force)
            # integrate physical state
            self.integrate_state(p_force)
        else:            
            for agent in self.agents:
                # update agent's physical states based on commanded action
                agent.platform_action.update_entity_state(agent, self)
                # update agent's missile states
                agent.fire_action.update_entity_state(agent, self)
            # self.integrate_state_from_vel_action()
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        # update sensor detections
        for entity in self.entities: # TODO: make this more efficient
            if entity.rwr is None: continue
            entity.rwr.observers = []
        for entity in self.entities:
            if entity.sensor is None: continue
            entity.sensor.detections = []
            for other in self.entities:
                if entity == other: continue
                entity.sensor.check_detection(entity, other)
                if other.rwr is None: continue
                other.rwr.check_observed(other, entity)
        # update world steps
        self.steps += 1

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    # integrate physical state when action is commanded velocity
    def integrate_state_from_vel_action(self):
        for agent in self.agents:
            if not agent.movable: continue
            speed_delta = (agent.max_speed - agent.min_speed) / 2
            speed = agent.platform_action.u[0] * speed_delta + (agent.min_speed + speed_delta)
            heading = agent.platform_action.u[1] * np.pi
            agent.state.p_vel[0] = speed * np.cos(heading)
            agent.state.p_vel[1] = speed * np.sin(heading)
            agent.state.p_pos += agent.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise   

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
