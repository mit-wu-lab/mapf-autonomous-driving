import argparse
from collections import deque, Counter, defaultdict
import math
from time import time
import cvxpy as cp
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib import animation
from util import format_yaml, Namespace, Path, Logger
from mcts.tree.search import MCTS
from mcts.tree.nodes import MCTSNode
from mcts.scenario.common import MCTSState
from simulator.intersection.envs import MultiAgentIntersectionEnv

### Setup Traffic Network / Model
class Lane(Namespace):
    def __repr__(self):
        return f'Lane(id={self.id})'

class Subzone(Namespace):
    def __repr__(self):
        return f'Subzone(id={self.id})'

class Route(Namespace):
    def __repr__(self):
        return f'Route(id={self.id}, length={self.length}, segments={self.segments})'

class Vehicle(Namespace):
    def __repr__(self):
        return f'Vehicle(id={self.id}, route={self.route.id}, segment={self.segment.id}, route_position={self.route_position}, segment_position={self.segment_position}, speed={self.speed})'

    def get_polygon(self, route_position=None):
        """
        Get vehicle polygon along the vehicle's route for rendering
        """
        route = self.route
        route_position = self.route_position if route_position is None else route_position
        index = np.argmin(route_position >= route.shape_distances)
        a, b = route.shape[index - 1], route.shape[index]
        distance_from_a = route_position - route.shape_distances[index - 1]
        tangent = (b - a) / np.linalg.norm(b - a)
        front_center = a + distance_from_a * tangent
        perp = rotate(tangent, 1)
        front = [-perp * (car_width / 2), perp * (car_width / 2)]
        return front_center + np.array([*front[::-1], *(front - tangent * car_length)])

def rotate(xy, rot90, center=[0, 0]):
    # Rotate xy coordinate(s) by multiples of 90 degrees around center
    theta = rot90 * np.pi / 2
    cos, sin = np.cos(theta), np.sin(theta)
    mat = np.array([[cos, -sin], [sin, cos]])
    return (mat @ (np.asarray(xy) - center).T).T + center

def init_traffic_network():
    zone_indices = np.arange(4)
    directions = ['U', 'L', 'D', 'R']
    # Subconflict zones: each zone's id is defined as 'ab' where a in [0, 1, 2, 3] indicates the [lower right, upper right, upper left, lower left] square quadrant and b in [0, 1, 2, 3] indicates the [lower, right, upper, left] triangle within quadrant a.
    # E.g. id='03' indicates the lower right square quadrant's left triangle subzone
    subzone_infos = []
    for x in zone_indices:
        for y in zone_indices:
            subzone_infos.append(f'{x}{y}')
    subzones = {id: Subzone(id=id, position={}, next={}, prev={}, vehicle=None, reservations=[], curr_reservation_index=0) for id in subzone_infos}

    # Incoming and outgoing lanes. Each lane's id is defined as 'ab' where a in [U, L, D, R] indicates [upward, leftward, downward, rightward] direction and b in [0, 1] indicates the [inward, outward] from the intersection.
    # E.g. id='D1' indicates the downward lane going out from the intersection.
    
    center = np.zeros(2) # ignor center
    lane_infos = [(
        f'{d}{j}',
        center
    ) for i, d in enumerate(directions) for j in range(2)]
    
    lanes = {id: Lane(id=id, position={}, next={}, prev={}, vehicles=deque(), arrivals=[], shape=shape) for id, shape in lane_infos}

    # Each route consists of a sequence of segments (subzones or lanes). Each route's id is defined as 'ab' where both a and b are in [U, L, D, R]. a indicates the incoming direction to the intersection and b indicates the outgoing direction from the intersection.
    # E.g. id='DR' indicates a route where vehicles go downward at first, then turn left at the intersection to go rightward.
    # route.segments is a list of tuples: [(arriving_lane, lane_start_pos, lane_end_pos, None), (subzone1, subzone_start_pos, subzone_end_pos, subzone_speed), ..., (departing_lane, ...)]
    routes = {}
    for i, from_d in enumerate(directions):
        ego_directions = np.roll(directions, -i)
        ego_zone_indices = np.roll(zone_indices, -i)
        ego_subzone = lambda a, b: subzones[''.join(map(str, ego_zone_indices[[a, b]]))]
        for j, to_d in enumerate(ego_directions):
            segments = [(lanes[f'{from_d}0'], 0, lane_length - car_length / 2, None)]
            def extend_segments(*args, speed=None):
                offset = segments[-1][2]
                segments.extend([(z, offset + a, offset + b, speed) for z, a, b in args])
            half_intersection_len = 2.5 * lane_width
            if j == 0: # straight
                half_len = half_intersection_len
                first_end = half_intersection_len / math.sqrt(2)
                second_start = 1.5 * lane_width
                second_end = 2 * lane_width
                extend_segments(
                    (ego_subzone(0, 0), 0, first_end),
                    (ego_subzone(0, 3), 0, half_len),
                    (ego_subzone(0, 1), second_start, second_end),
                    (ego_subzone(0, 2), first_end, half_len),
                    (ego_subzone(1, 0), half_len, 2 * half_len - first_end),
                    (ego_subzone(1, 3), half_len, 2 * half_len),
                    (ego_subzone(1, 1), 2 * half_len - second_end, 2 * half_len - second_start),
                    (ego_subzone(1, 2), 2 * half_len - first_end, 2 * half_len),
                    speed=max_speed,
                )
            elif j == 1: # left turn
                radius = 3 * lane_width
                half_len = (np.pi / 4) * radius
                # 3.5w sin θ = 2.5w sin φ and 3.5w cos θ + 2.5w cos φ = 5w ----> θ = 2 arctan(sqrt(2/33))
                first_end = 2 * math.atan(math.sqrt(2 / 33)) * radius
                extend_segments(
                    (ego_subzone(0, 0), 0, first_end),
                    (ego_subzone(0, 3), 0, half_len),
                    (ego_subzone(3, 1), 0, half_len),
                    (ego_subzone(3, 2), half_len, 2 * half_len),
                    (ego_subzone(2, 0), half_len, 2 * half_len),
                    (ego_subzone(2, 3), 2 * half_len - first_end, 2 * half_len),
                    speed=left_speed,
                )
            elif j == 2: # u-turn
                y_center2 = 1.62467 * lane_width
                y_center2_from_outer = half_intersection_len - y_center2
                r2 = y_center2 / np.sqrt(2) - car_width
                r13 = 1.5 * lane_width
                switch_angle = np.arctan(y_center2_from_outer / half_intersection_len)
                half_len = r2 * (np.pi / 2 + switch_angle) + r13 * switch_angle
                extend_segments(
                    (ego_subzone(0, 0), 0, half_len),
                    (ego_subzone(0, 3), 0, 2 * half_len),
                    (ego_subzone(3, 1), 0, 2 * half_len),
                    (ego_subzone(3, 0), half_len, 2 * half_len),
                    speed=u_speed,
                )
            elif j == 3: # right turn
                radius = 2 * lane_width
                half_len = (np.pi / 4) * radius
                extend_segments(
                    (ego_subzone(0, 0), 0, half_len),
                    (ego_subzone(0, 1), half_len, 2 * half_len),
                    speed=right_speed,
                )
            extend_segments((lanes[f'{to_d}1'], 0, lane_length), speed=None)
            # shape = rotate(shape, i)
            shape = np.zeros(2)
            # shape_distances = np.cumsum([0, *np.sqrt(((shape[1:] - shape[:-1]) ** 2).sum(axis=1))])
            shape_distances = 0
            routes[from_d + to_d] = route = Route(id=f'{from_d}{to_d}', segments=segments, subzones=[x for x in segments if isinstance(x[0], Subzone)], shape=shape, shape_distances=shape_distances)
            for z, s, e, _ in segments:
                z.position[route.id] = (s, e)
            for (z1, _, _, _), (z2, _, _, _) in zip(segments, segments[1:]):
                z1.next[route.id] = z2
                z2.prev[route.id] = z1

            route.length = route.segments[-1][2]
            route.max_crossing_speed = segments[1][3]
            route.min_travel_time = sum([
                compute_min_approach_time(segments[0][2] - segments[0][1], init_speed, route.max_crossing_speed),
                (segments[-2][2] - segments[1][1]) / route.max_crossing_speed,
                compute_min_approach_time(segments[-1][2] - segments[-1][1], route.max_crossing_speed, max_speed),
            ])
            # assert shape_distances[-1] == route.length
            route.vehicles = deque()
            route.id_count = 0

    return lanes, subzones, routes

### Planning

def compute_min_approach_time(distance, init_speed, final_speed):
    if init_speed < final_speed:
        min_time = (final_speed - init_speed) / max_accel
    else:
        min_time = (init_speed - final_speed) / max_decel
    min_distance = min_time * (final_speed + init_speed) / 2
    if min_distance > distance:
        return None # infeasible
    if min_distance == distance:
        return min_time

    # try accelerate to max then decelerate
    init_accel_time = (max_speed - init_speed) / max_accel
    init_accel_distance = init_accel_time * (max_speed + init_speed) / 2
    final_decel_time = (max_speed - final_speed) / max_decel
    final_decel_distance = final_decel_time * (max_speed + final_speed) / 2
    const_speed_distance = distance - init_accel_distance - final_decel_distance
    if const_speed_distance >= 0: # Accelerate, constant init_speed, then decelerate
        return init_accel_time + (const_speed_distance / max_speed) + final_decel_time
    # Accelerate at max_accel then decelerate at -max_decel. Need to solve a quadratic equation
    a = max_accel * (max_accel + max_decel)
    b = 2 * init_speed * (max_accel + max_decel)
    c = init_speed * init_speed - final_speed * final_speed - 2 * max_decel * distance
    init_accel_time = (-b + math.sqrt(b * b - 4 * a * c)) / 2 / a
    final_decel_time = (init_speed - final_speed + max_accel * init_accel_time) / max_decel
    return init_accel_time + final_decel_time

def optimize_speed(veh, H_p, max_positions, min_positions=None, target_speeds=None, max_step_speed=None, max_position_speed=None):
    if max_position_speed:
        pos, pos_max_speed = max_position_speed

        from gurobipy import Model, GRB
        m = Model(f'{step}_{veh.id}')
        m.setParam("OutputFlag", 0)
        m.setParam(GRB.Param.IntFeasTol, 1e-2)
        m.setParam(GRB.Param.MIPGap, 1e-2)
        m.setParam(GRB.Param.Threads, 10)
        x = m.addMVar(1 + H_p, lb=0)
        v = m.addMVar(1 + H_p, lb=0, ub=max_speed)
        m.update()
        a = (v[1:] - v[:-1]) / dt

        m.addConstr(x[0] == veh.route_position)
        m.addConstr(v[0] == veh.speed)
        # m.addConstr(x[1:] - x[:-1] == (v[:-1] + v[1:]) / 2 * dt)
        m.addConstr(x[1:] - x[:-1] == v[:-1] * dt)
        m.addConstr(a >= -max_decel)
        m.addConstr(a <= max_accel)
        m.addConstr(x <= max_positions)
        m.addConstr(x >= min_positions)
        
        if pos_max_speed < max_speed:
            v_max = m.addMVar(1 + H_p, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            v_max_left = m.addMVar(1 + H_p, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            v_max_right = m.addMVar(1 + H_p, lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.update()
            m.addConstr(v_max_left == pos_max_speed + max_accel * (x - pos) / pos_max_speed)
            m.addConstr(v_max_right == pos_max_speed + max_decel * (pos - x) / pos_max_speed)
            for i in range(1 + H_p):
                m.addGenConstrMax(v_max[i], [v_max_left[i], v_max_right[i]])
            m.addConstr(v <= v_max)
# TODO try:
#             radius = (max_speed * dt) / 2 + 1e-4
#             m.addConstr(v <= max_speed + (pos_max_speed - max_speed) * and_(-radius <= x - pos, x - pos <= radius))
        m.setObjective(v.sum(), GRB.MAXIMIZE)
        m.optimize()
        if m.status == GRB.Status.INFEASIBLE:
            return None, None, None
        ret = x.X, v.X, a.getValue()
        m.dispose()
        return ret

    x = cp.Variable(1 + H_p)
    v = cp.Variable(1 + H_p)
    objective = cp.Maximize(cp.sum(v))
    a = (v[1:] - v[:-1]) / dt
    constraints = [
        x[0] == veh.route_position,
        v[0] == veh.speed,
        v >= 0,
        v <= max_speed,
        # x[1:] - x[:-1] == (v[:-1] + v[1:]) / 2 * dt,
        x[1:] - x[:-1] == v[:-1] * dt,
        a >= -max_decel,
        a <= max_accel,
    ]
    max_positions_mask = ~np.isinf(max_positions)
    max_position_constraints = max_positions[max_positions_mask]
    if len(max_position_constraints):
        constraints.append(x[:len(max_positions)][max_positions_mask] <= max_position_constraints)

    if min_positions is not None:
        min_positions_mask = ~np.isinf(min_positions)
        min_position_constraints = min_positions[min_positions_mask]
        if len(min_position_constraints):
            constraints.append(x[:len(min_positions)][min_positions_mask] >= min_position_constraints)

    if target_speeds is not None:
        target_speed_mask = ~np.isnan(target_speeds)
        target_speed_constraints = target_speeds[target_speed_mask]
        if len(target_speed_constraints):
            constraints.append(v[:len(target_speeds)][target_speed_mask] == target_speed_constraints)

    if max_step_speed:
        target_speed_step, max_step_speed = max_step_speed
        constraints.append(v[target_speed_step] <= max_step_speed)

    problem = cp.Problem(objective, constraints)
    problem.solve()
    if problem.status == cp.INFEASIBLE:
        return None, None, None
    return x.value, v.value, a.value

def compute_vehicle_step_info(veh):
    # if veh.get('last_info_step') == step:
    #     return True
    route = veh.route
    from_lane, _, veh.from_lane_pos_end, _ = route.segments[0]
    veh.approach_distance = veh.from_lane_pos_end - veh.segment_position
    veh.max_crossing_speed = veh.crossing_speed = route.max_crossing_speed
    veh.min_approach_time = compute_min_approach_time(veh.approach_distance, veh.speed, veh.crossing_speed)
    if veh.min_approach_time is None:
        if veh.speed > veh.max_crossing_speed:
            return False # infeasible to reduce veh's speed from veh.speed to veh.max_crossing_speed
        else: # Try to increase vehicle speed as much as possible
            veh.crossing_speed = math.sqrt(veh.speed * veh.speed + 2 * max_accel * veh.approach_distance)
            veh.min_approach_time = (veh.crossing_speed - veh.speed) / max_decel
    veh.min_stop_distance = (veh.speed / 2) * (veh.speed / max_decel)
    veh.max_approach_time = np.inf
    if veh.min_stop_distance > veh.approach_distance: # Cannot stop fully before approach_distance, so instead calculate the approach_time if max_decel is used
        veh.max_approach_time = (veh.speed - math.sqrt(veh.speed * veh.speed - 2 * max_decel * veh.approach_distance)) / max_decel
    assert veh.max_approach_time >= veh.min_approach_time - 1e-4
    veh.last_info_step = step
    return True

def plan_vehicle(veh, use_leaders=False, use_arrival_times=True, reserve_actual=False, allow_infeasible=False,
    get_last_reservation=lambda subzone: subzone.reservations[-1] if subzone.reservations else (None, step, step),
    add_reservation=lambda subzone, reservation: subzone.reservations.append(reservation),
):
    assert compute_vehicle_step_info(veh)

    route = veh.route
    from_lane, _, from_lane_pos_end, _ = route.segments[0]
    to_lane, to_lane_pos_start, to_lane_pos_end, _ = route.segments[-1]

    # Compute necessary information for the arrival and departure times at each subzone
    other_end_steps, subzone_start_offsets, subzone_end_offsets = [], [], []
    for subzone, pos_start, pos_end, _ in route.subzones:
        other_end_steps.append(get_last_reservation(subzone)[-1] - step)
        subzone_start_offsets.append((pos_start - from_lane_pos_end) / veh.crossing_speed / dt)
        subzone_end_offsets.append((pos_end - from_lane_pos_end + car_length) / veh.crossing_speed / dt)
    other_end_steps, subzone_start_offsets, subzone_end_offsets = np.array(other_end_steps), np.array(subzone_start_offsets), np.array(subzone_end_offsets)

    veh.cross_available_step = (other_end_steps - subzone_start_offsets).max()
    veh.cross_start_step = cross_start_step = math.ceil(max(veh.min_approach_time / dt, veh.cross_available_step)) # Take the max arrival time at the intersection

    reserve_start_steps = np.maximum(other_end_steps, np.floor(cross_start_step + subzone_start_offsets).astype(int))
    reserve_end_steps = np.ceil(reserve_start_steps + 1 - subzone_start_offsets + subzone_end_offsets).astype(int)

    if use_leaders: # Low level computations
        target_speed_step = round((reserve_start_steps[0] + reserve_end_steps[-1]) / 2)
        H_p = math.ceil(reserve_end_steps[-1] + compute_min_approach_time(to_lane_pos_end - to_lane_pos_start, veh.crossing_speed, max_speed)) + H_p_padding

        leader_constraints = {}
        if from_lane.arrivals:
            from_leader = from_lane.arrivals[-1]
            from_leader_rear_pos = from_leader.route_positions[step - from_leader.start_step:] - car_length
            if len(from_leader_rear_pos):
                assert from_leader_rear_pos[-1] > from_lane_pos_end
                from_leader_end_step = np.argmax(from_leader_rear_pos > from_lane_pos_end)
                leader_constraints[from_leader.id, 0, from_leader_end_step] = from_leader_rear_pos[:from_leader_end_step]
        if to_lane.arrivals:
            to_leader = to_lane.arrivals[-1]
            to_leader_to_lane_pos_start, _ = to_lane.position[to_leader.route.id]
            to_leader_rear_pos = to_leader.route_positions[step - to_leader.start_step:] - car_length
            if len(to_leader_rear_pos):
                assert to_leader_rear_pos[-1] >= to_leader_to_lane_pos_start
                leader_start_step = np.argmax(to_leader_rear_pos >= to_leader_to_lane_pos_start)
                min_len = min(1 + H_p, len(to_leader_rear_pos))
                leader_constraints[to_leader.id, leader_start_step, min_len] = to_leader_rear_pos[leader_start_step: min_len] - to_leader_to_lane_pos_start + to_lane_pos_start

        max_positions, min_positions = np.empty(1 + H_p), np.empty(1 + H_p)
        additional_delay_steps = iter([1, 1] if use_arrival_times is False else [])
        while True: # Adjust until feasible
            max_positions[:] = np.inf
            min_positions[:] = 0
            for (_, start_step, end_step), positions in leader_constraints.items():
                np.minimum(max_positions[start_step: end_step], positions, max_positions[start_step: end_step])
            if use_arrival_times is False:
                for (subzone, pos_start, _, _), other_end_step in zip(route.subzones, other_end_steps):
                    if other_end_step > 0:
                        np.minimum(max_positions[:other_end_step + 1], pos_start, max_positions[:other_end_step + 1])
                optimize_kwargs = dict(max_position_speed=(route.length / 2, veh.max_crossing_speed))
            else:
                if use_arrival_times == 'partial':
                    np.minimum(max_positions[:reserve_start_steps[0] + 1], from_lane_pos_end, max_positions[:reserve_start_steps[0] + 1])
                    np.maximum(min_positions[reserve_end_steps[-1]:], to_lane_pos_start, min_positions[reserve_end_steps[-1]:])
                    for (subzone, pos_start, _, _), other_end_step in zip(route.subzones, other_end_steps):
                        if other_end_step > 0:
                            np.minimum(max_positions[:other_end_step + 1], pos_start, max_positions[:other_end_step + 1])
                else:
                    for (subzone, pos_start, pos_end, _), start_step, end_step in zip(route.subzones, reserve_start_steps, reserve_end_steps):
                        np.minimum(max_positions[:start_step + 1], pos_start, max_positions[:start_step + 1]) # Must not enter the subzone before start_step
                        np.maximum(min_positions[end_step:], pos_end + car_length, min_positions[end_step:]) # Must leave the subzone before end_step
                optimize_kwargs = dict(max_step_speed=(target_speed_step, veh.max_crossing_speed))
            route_positions, speeds, accels = optimize_speed(veh, H_p, max_positions, min_positions=min_positions, **optimize_kwargs)
            if route_positions is None:
                try:
                    next_delay_steps = next(additional_delay_steps)
                except StopIteration:
                    if allow_infeasible: return False
                    # veh.update({k: v for k, v in locals().items() if k not in ['route_positions', 'speeds', 'accels', 'start_step']}) # For debug
                    raise RuntimeError(veh, 'infeasible')
#                 print(veh, f'infeasible, add {next_delay_steps} delay steps')
                reserve_end_steps += next_delay_steps
                target_speed_step = round((reserve_start_steps[0] + reserve_end_steps[-1]) / 2)
                continue

            if use_arrival_times in [False, 'partial'] or reserve_actual:
                pos_starts, pos_ends = zip(*((pos_start, pos_end) for _, pos_start, pos_end, _ in route.subzones))
                reserve_start_steps = np.argmax(route_positions[:, None] - 1e-6 > pos_starts, axis=0) - 1
                reserve_end_steps_ = np.argmax(route_positions[:, None] - car_length >= pos_ends, axis=0)
                if use_arrival_times and not (reserve_start_steps[0] <= target_speed_step < reserve_end_steps_[-1]):
                    new_target_speed_step = round((target_speed_step + reserve_start_steps[0] + reserve_end_steps_[-1]) / 3)
                    print(f'{veh}. target_speed_step={target_speed_step} outside of crossing steps [{reserve_start_steps[0]}, {reserve_end_steps_[-1]}]. Try again with target_speed_step={new_target_speed_step}')
                    target_speed_step = new_target_speed_step
                    continue
                reserve_end_steps = reserve_end_steps_
            break
        veh.route_positions = np.concatenate([veh.route_positions[:step - veh.start_step], route_positions])
        veh.speeds = np.concatenate([veh.speeds[:step - veh.start_step], speeds])
        veh.accels = np.concatenate([veh.accels[:step - veh.start_step], accels])

        from_lane.arrivals.append(veh)
        to_lane.arrivals.append(veh)
    assert (reserve_start_steps >= other_end_steps).all()
    # Make reservations with arrival and departure times
    for (subzone, _, _, _), start_step, end_step in zip(route.subzones, step + reserve_start_steps, step + reserve_end_steps):
        add_reservation(subzone, (veh, start_step, end_step))
    return True

def plan_vehicles(order=None, **kwargs):
    start_time = time()
    success = True
    for veh in (order or [veh for veh in vehicles.values() if len(veh.accels) == 0]):
        if not plan_vehicle(veh, use_leaders=True, use_arrival_times=use_arrival_times, reserve_actual=reserve_actual, **kwargs):
            success = False
            break
    return success, dict(async_plan_time=time() - start_time)

def try_plan_vehicles_ordered(order, order_id_set=None, **kwargs):
    order_id_set = order_id_set or {veh.id for veh in order}
    # Save in case of revert
    for lane in lanes.values():
        lane.arrivals_ = []
        while lane.arrivals and lane.arrivals[-1].id in order_id_set:
            lane.arrivals_.append(lane.arrivals.pop())
    for subzone in subzones.values():
        subzone.reservations_ = []
        while subzone.reservations and subzone.reservations[-1][0].id in order_id_set:
            subzone.reservations_.append(subzone.reservations.pop())
    for veh in order:
        # for k in ['route_positions', 'speeds', 'accels']:
        #     veh[k + '_'] = veh[k]

        veh.route_positions_ = veh.route_positions
        veh.speeds_ = veh.speeds
        veh.accels_ = veh.accels
    
    
    # Low level replan
    success, stats = plan_vehicles(order=order, allow_infeasible=True, **kwargs)
    stats['sync_plan_time_low'] = stats.pop('async_plan_time')
    if not success: # Failed, revert states
        for veh in order:
            # for k in ['route_positions', 'speeds', 'accels']:
            #     veh[k] = veh[k + '_']
            veh.route_positions = veh.route_positions_
            veh.speeds = veh.speeds_
            veh.accels = veh.accels_

        for lane in lanes.values():
            while lane.arrivals and lane.arrivals[-1].id in order_id_set:
                lane.arrivals.pop()
            lane.arrivals.extend(reversed(lane.arrivals_))
        for subzone in subzones.values():
            while subzone.reservations and subzone.reservations[-1][0].id in order_id_set:
                subzone.reservations.pop()
            subzone.reservations.extend(reversed(subzone.reservations_))
    return success, stats

def setup_high_level():
    in_lanes = [lane for lane in lanes.values() if lane.id.endswith('0')]
    order_map = {veh.id: i for i, veh in enumerate(curr_order)}
    replan_vehicles = {veh.id: veh for lane in in_lanes for veh in lane.vehicles if veh.id in order_map}
    last_infeasible = -1
    for veh_id, veh in replan_vehicles.items():
        if not compute_vehicle_step_info(veh):
            last_infeasible = max(last_infeasible, order_map[veh_id])
    if last_infeasible >= 0:
        replan_vehicles = {veh_id: veh for veh_id, veh in replan_vehicles.items() if order_map[veh_id] > last_infeasible}
    for subzone in subzones.values():
        for veh, start_step, _ in reversed(subzone.reservations):
            if start_step >= step + min_steps_to_reservation:
                continue
            if start_step < step:
                break
            replan_vehicles.pop(veh.id, None)

    for veh in replan_vehicles.values():
        route = veh.route
        veh.subzone_approach_times = veh.min_approach_time + (np.array([subzone.position.get(route.id, (np.nan,))[0] for subzone in subzones.values()]) - veh.from_lane_pos_end) / veh.crossing_speed

    replan_vehs_fifo = sorted(replan_vehicles.values(), key=lambda veh: int(veh.id.split('_')[0]))
    replan_lane_vehs = [[veh for veh in lane.vehicles if veh.id in replan_vehicles] for lane in in_lanes]

    for subzone in subzones.values():
        subzone.last_reservation_ = (None, step, step) # Store this for now
        for veh, start_step, end_step in reversed(subzone.reservations):
            if veh.id not in replan_vehicles:
                subzone.last_reservation_ = (veh, start_step, end_step)
                break
    return replan_vehs_fifo, replan_vehicles, replan_lane_vehs, dict(
        get_last_reservation=lambda subzone: subzone.last_reservation,
        add_reservation=lambda subzone, reservation: subzone.update(last_reservation=reservation),
    )

def get_nondominated_min_subzone_approach(vehs):
    approach_times = np.array([veh.subzone_approach_times for veh in vehs])
    no_approach = np.isnan(approach_times)
    closer_to_subzone = no_approach[:, None, :] | no_approach[None, :, :] | (approach_times[:, None, :] <= approach_times[None, :, :]) # (num_queues, num_queues, num_subzones)
    closest = closer_to_subzone.all(axis=(1, 2))
    action_mask = closest if closest.any() else closer_to_subzone.all(axis=1).any(axis=1)
    return np.where(action_mask)[0]

def plan_order_approach_times(order, reservation_fns, approach_times=None, min_approach_times=None):
    for subzone in subzones.values():
        subzone.last_reservation = subzone.last_reservation_
    min_approach_times = np.zeros(len(order)) if min_approach_times is None else min_approach_times
    approach_times = np.zeros_like(min_approach_times) if approach_times is None else approach_times
    for j, veh in enumerate(order):
        plan_vehicle(veh, use_leaders=False, **reservation_fns)
        plan_vehicle.count += 1
        if veh.cross_available_step * dt > veh.max_approach_time: # Vehicle cannot stop in time for the current crossing order
            approach_times[j] = np.inf
            break
        approach_times[j] = veh.cross_start_step * dt
        min_approach_times[j] = veh.min_approach_time
    return approach_times, min_approach_times

def plan_vehicles_synchronous(get_order_fn):
    if step % H_c != 0:
        return True, plan_vehicles()[1]

    fifo_order, replan_vehicles, replan_lane_vehs, reservation_fns = setup_high_level()
    plan_vehicle.count = 0
    fifo_approach_times, fifo_min_approach_times = plan_order_approach_times(fifo_order, reservation_fns)
    fifo_delay = (fifo_approach_times - fifo_min_approach_times).mean()
    
    start_time = time()
    plan_vehicle.count = 0
    order, stats = get_order_fn(fifo_order, replan_vehicles, replan_lane_vehs, reservation_fns)
    stats.update(fifo_delay=fifo_delay, sync_plan_time_high=time() - start_time, plan_vehicle_count=plan_vehicle.count)
    if order is None:
        print('step', step, 'no order obtained, falling back to previous trajectories')
        return False, dict(**stats, **plan_vehicles()[1])
    approach_times, min_approach_times = plan_order_approach_times(order, reservation_fns)
    stats['delay'] = (approach_times - min_approach_times).mean()

    # actual_approach_times = np.array([np.argmax(np.array(veh.route_position_history[step - veh.start_step:]) >= veh.route.subzones[0][1]) * dt for veh in order])
    # stats['actual_delay'] = actual_approach_times.mean() - min_approach_times.mean()

    success, low_level_stats = try_plan_vehicles_ordered(order=order, order_id_set=replan_vehicles)
    stats.update(low_level_stats)
    if not success:
        print('step', step, 'order infeasible, falling back to previous trajectories')
        return False, {**stats, **plan_vehicles()[1]}
    curr_order[:] = order
    return True, stats

def prioritized_planning(fifo_order, replan_vehicles, replan_lane_vehs, reservation_fns, num_random_orders=0):
    orders = [fifo_order]
    for _ in range(num_random_orders):
        # To construct an order use MCTS paper's two heuristics based on lane ordering and subzone approach times
        vehicle_queues = [lane_vehs[::-1] for lane_vehs in replan_lane_vehs if len(lane_vehs)]
        orders.append(order := [])
        while len(vehicle_queues):
            use_constraint = use_approach_time_constraint and len(order) >= use_approach_time_min_depth
            actions = get_nondominated_min_subzone_approach([lane_vehs[-1] for lane_vehs in vehicle_queues]) if use_constraint else len(vehicle_queues)
            i = alg_state.choice(actions)
            order.append(vehicle_queues[i].pop())
            if len(vehicle_queues[i]) == 0:
                vehicle_queues.pop(i)
    unique_orders = {'_'.join(veh.id.split('_')[0] for veh in order): order for order in orders}
    orders = list(unique_orders.values())

    approach_times = np.zeros((len(orders), len(replan_vehicles)))
    min_approach_times = np.zeros_like(approach_times)
    for i, order in enumerate(orders):
        plan_order_approach_times(order, reservation_fns, approach_times[i], min_approach_times[i])
    delays = (approach_times - min_approach_times).mean(axis=1)
    stats = dict(num_orders=len(orders))
    if np.isinf(delays).all():
        print('step', step, f'all {len(orders)} orders infeasible')
        return None, stats
    return orders[np.argmin(delays)], stats

class PointMassMCTSState(MCTSState):
    def __init__(self, lane_vehs, next_lane_veh_idxs=None, order=[]):
        self.lane_vehs = lane_vehs
        self.next_lane_veh_idxs = next_lane_veh_idxs or [0] * len(lane_vehs)
        self.order = order

    def rollout_result(self):
        self.approach_times, self.min_approach_times = plan_order_approach_times(self.order, PointMassMCTSState.reservation_fns)
        self.delay = (self.approach_times - self.min_approach_times).mean()
        return -self.delay

    def get_legal_actions(self):
        vehs = {i: lane_vehs_i[idx] for i, (lane_vehs_i, idx) in enumerate(zip(self.lane_vehs, self.next_lane_veh_idxs)) if idx < len(lane_vehs_i)}
        use_constraint = use_approach_time_constraint and len(self.order) >= use_approach_time_min_depth
        return list(np.array(list(vehs.keys()))[get_nondominated_min_subzone_approach(list(vehs.values()))] if use_constraint else vehs.keys())

    def move(self, action):
        lane_vehs_a, idx_a = self.lane_vehs[action], self.next_lane_veh_idxs[action]
        new_order = self.order + [lane_vehs_a[idx_a]]
        new_next_lane_veh_idxs = self.next_lane_veh_idxs.copy()
        new_next_lane_veh_idxs[action] += 1
        return PointMassMCTSState(self.lane_vehs, new_next_lane_veh_idxs, new_order)

    def is_terminal(self):
        return all(idx == len(lane_vehs_i) for lane_vehs_i, idx in zip(self.lane_vehs, self.next_lane_veh_idxs))

class PointMassMCTSNode(MCTSNode):
    def rollout_policy(self, possible_moves):        
        return possible_moves[alg_state.randint(len(possible_moves))]


def mcts(fifo_order, replan_vehicles, replan_lane_vehs, reservation_fns):
    PointMassMCTSState.reservation_fns = reservation_fns
    state = PointMassMCTSState(replan_lane_vehs)
    while len(state.order) < len(replan_vehicles):
        root = PointMassMCTSNode(state)
        mcts = MCTS(root)
        best_node = mcts.best_action(num_simulations=num_mcts_simulations)
        state = best_node.state
    return state.order, dict()

def obs(fifo_order, replan_vehicles, replan_lane_vehs, reservation_fns):
    sentinel_veh = Namespace(lowers=[], cumulative_end_steps=np.array([subzone.last_reservation_[-1] for subzone in subzones.values()]))
    # Plan shortest paths (compute earliest reservations) for all vehicles conditioned on their preceding vehicle
    for lane_vehs in replan_lane_vehs:
        for subzone in subzones.values():
            subzone.last_reservation = subzone.last_reservation_
        
        prev_veh = sentinel_veh
        for veh in lane_vehs: # Plan each lane's vehicle independently of other lanes
            plan_vehicle(veh, use_leaders=False, **reservation_fns)
            plan_vehicle.count += 1
            veh.start_steps, veh.end_steps = np.array([(start, end) if veh_ is veh else (np.inf, step) for subzone in subzones.values() for veh_, start, end in (subzone.last_reservation,)]).T
            veh.cumulative_end_steps = np.maximum(prev_veh.cumulative_end_steps, veh.end_steps)
            veh.highers = [prev_veh] # Vehicles with immediately higher priority than veh
            prev_veh.lowers.append(veh)
            prev_veh = veh
            prev_veh.lowers = [] # Vehicles with immediately lower priority than veh
        
        next_veh = None
        for veh in reversed(lane_vehs):
            veh.cumulative_start_steps = veh.start_steps if next_veh is None else np.minimum(veh.start_steps, next_veh.cumulative_start_steps)
            next_veh = veh
                
    heads = sorted(sentinel_veh.lowers, key=lambda veh: veh.cross_start_step) # Sorting the heads by the estimated approach step helps a lot

    def replan(veh, restore, next_vehs=None):
        # Replan a vehicle veh. Replan vehicles with lower priority than veh
        if veh.id not in restore:
            # restore[veh.id] = {key: veh[key] for key in ['start_steps', 'end_steps', 'cumulative_start_steps', 'cumulative_end_steps', 'cross_start_step']}
            restore[veh.id] = {'start_steps': veh.start_steps,
                                'end_steps': veh.end_steps,
                                'cumulative_start_steps': veh.cumulative_start_steps,
                                'cumulative_end_steps': veh.cumulative_end_steps,
                                'cross_start_step': veh.cross_start_step}
        for subzone, end_step in zip(subzones.values(), np.maximum.reduce([veh_.cumulative_end_steps for veh_ in veh.highers])):
            subzone.last_reservation = (None, None, end_step)
        plan_vehicle(veh, use_leaders=False, **reservation_fns)
        plan_vehicle.count += 1
        veh.start_steps, veh.end_steps = np.array([(start, end) if veh_ is veh else (np.inf, step) for subzone in subzones.values() for veh_, start, end in (subzone.last_reservation,)]).T
        veh.cumulative_end_steps = np.array([subzone.last_reservation[2] for subzone in subzones.values()])
        next_vehs = next_vehs or {}
        next_vehs_candidates = veh.lowers.copy()
        while len(next_vehs_candidates):
            veh_ = next_vehs_candidates.pop()
            if (veh.cumulative_end_steps > veh_.cumulative_start_steps).any():
                next_vehs[veh_.id] = veh_
                next_vehs_candidates.extend(veh_.lowers)
        if next_vehs:
            veh_ = min(next_vehs.values(), key=lambda veh_: veh_.cross_start_step)
            next_vehs.pop(veh_.id)
            replan(veh_, restore, next_vehs=next_vehs)
        veh.cumulative_start_steps = np.minimum.reduce([veh.start_steps, *(veh_.cumulative_start_steps for veh_ in veh.lowers)])

    def search(heads, limit):
        while len(heads) > 0:
            cumulative_start_steps = np.array([veh.cumulative_start_steps for veh in heads]).reshape((len(heads), len(subzones)))
            cumulative_end_steps = np.array([veh.cumulative_end_steps for veh in heads]).reshape((len(heads), len(subzones)))
            higher_than_others = (cumulative_start_steps[None, :, :] >= cumulative_end_steps[:, None, :]).all(axis=2)
            num_higher_than_others = higher_than_others.sum(axis=1)
            index = np.argmax(num_higher_than_others)
            if num_higher_than_others[index] == len(heads) - 1:
                veh = heads.pop(index)
                heads.extend(veh.lowers)
                heads.sort(key=lambda veh: veh.cross_start_step) # Sorting the heads helps a lot
            else:
                veh_a = heads[index]
                a_higher_than_others = higher_than_others[index]
                num_higher_than_others[index] = -2
                num_higher_than_others[a_higher_than_others] = -1
                veh_b_index = np.argmax(num_higher_than_others)
                veh_b = heads[veh_b_index]
                break

        if len(heads) == 0:
            order = []
            for veh in replan_vehicles.values():
                veh.highers_ = veh.highers.copy()
            todo = sentinel_veh.lowers.copy()
            while todo:
                for veh in todo:
                    if all((veh.cumulative_end_steps <= veh_.cumulative_start_steps).all() for veh_ in todo if veh_ is not veh):
                        break
                else:
                    raise RuntimeError('This should not happen')
                todo.remove(veh)
                order.append(veh)
                for lower in veh.lowers:
                    lower.highers_.remove(veh)
                    if not lower.highers_:
                        todo.append(lower)
            
            delay = np.mean([veh.cross_start_step * dt - veh.min_approach_time for veh in order])
            assert len(order) == len(replan_vehicles)
            orders.append(order)
            delays.append(delay)
            return 1

        num_orders = 0
        child_limit = limit and math.ceil(limit / 2)
        for veh1, veh2 in [(veh_a, veh_b), (veh_b, veh_a)]:
            veh2.highers.append(veh1)
            veh1.lowers.append(veh2)
            restore = {veh1.id: dict(cumulative_start_steps=veh1.cumulative_start_steps)}
            
            replan(veh2, restore)
            veh1.cumulative_start_steps = np.minimum.reduce([veh1.start_steps, *(veh_.cumulative_start_steps for veh_ in veh1.lowers)])
            num_orders += search([head for head in heads if head is not veh2], child_limit) # veh2 is no longer a head since it is after veh1
            
            for veh_id, info in restore.items():
                for attribute in info.keys():
                # replan_vehicles[veh_id].update(**info)
                    setattr(replan_vehicles[veh_id], attribute, info[attribute])
            assert veh1 is veh2.highers.pop()
            assert veh2 is veh1.lowers.pop()

            child_limit = limit and (limit - num_orders)
            if num_orders >= obs_order_threshold or child_limit == 0:
                return num_orders
        return num_orders

    orders, delays = [], []
    num_orders = search(heads, num_obs_orders)
    return orders[np.argmin(delays)], dict(num_orders=num_orders)
    
### Simulator
def simulate():
    
    global lanes, subzones, routes, vehicles, step, curr_order
    from_d_map = dict(U=0, L=3, D=2, R=1) # Mapping "from direction" to HighwayEnv nodes
    to_d_map = dict(U=2, L=1, D=0, R=3) # Mapping "to direction" to HighwayEnv nodes

    spawn_state = np.random.RandomState(spawn_seed)
    lanes, subzones, routes = init_traffic_network()
    env = MultiAgentIntersectionEnv(config={"policy_frequency": 1/dt, "seed": seed}, subzones=subzones, render=render)
    import matplotlib
    vehicle_colors = [*map(tuple, np.array([*map(matplotlib.colors.to_rgb, matplotlib.pyplot.rcParams['axes.prop_cycle'].by_key()['color'])]) * 255)]
    next_color = 0

    queue = Counter()
    global_counter = 0
    direction_counter = Counter()
    vehicles = {}
    finished_vehicles = {}
    curr_order = []
    for step in range(H):
        start_time = time()
        t = step * dt

        # Spawn new vehicles
        for i, from_d in enumerate(directions):
            if spawn_state.random() < (lam[from_d] if isinstance(lam, dict) else lam) / 3600 * dt or step == 0:
                queue[from_d] += 1
            if queue[from_d] == 0:
                continue

            lane = lanes[f'{from_d}0']
            if lane.vehicles:
                leader = lane.vehicles[-1]
                leader_pos, leader_speed = leader.segment_position, leader.speed
                leader_min_stop_time = leader_speed / max_decel
                leader_min_stop_pos = leader_pos + (leader_speed / 2) * leader_min_stop_time
                veh_min_stop_pos = (init_speed / 2) * (init_speed / max_decel)
                leader_min_stop_times = np.arange(1, int(leader_min_stop_time / dt)) * dt
                leader_min_stop_traj = leader_pos + (leader_speed - max_decel * leader_min_stop_time / 2) * leader_min_stop_times
                veh_min_stop_traj = (init_speed - max_decel * leader_min_stop_times / 2) * leader_min_stop_times
                if leader_pos < car_length or (veh_min_stop_traj > leader_min_stop_traj - car_length).any() or veh_min_stop_pos > leader_min_stop_pos - car_length: # Cannot guarantee no collision with leader
                    continue

            queue[from_d] -= 1
            ego_directions = np.roll(directions, -i)
            to_d = spawn_state.choice(ego_directions, p=p)
            route = routes[f'{from_d}{to_d}']
            initial_lane, target_lane = from_d_map[from_d], to_d_map[to_d]
            veh_id = f'{global_counter}_{from_d}{direction_counter[from_d]}_{route.id}{route.id_count}'
            global_counter += 1
            direction_counter[from_d] += 1
            route.id_count += 1
            veh = env.make_vehicle(initial_lane, target_lane, init_speed,
                id=veh_id,
                route=route,
                segment=lane,
                speeds=[init_speed],
                speed_history=[init_speed],
                segment_position=0,
                start_step=step,
                route_position=0,
                accel=None,
                route_positions=[0],
                accels=[],
                route_position_history=[0],
                color=vehicle_colors[next_color],
            )
            next_color = (next_color + 1) % len(vehicle_colors)

            if veh is not None:
                vehicles[veh_id] = veh
                lane.vehicles.append(veh)
                route.vehicles.append(veh)
                curr_order.append(veh)

        init_time = time() - start_time
        success, stats = plan_vehicles_fn()
        start_time = time()
        # Update vehicle positions and segments
        new_segment_vehicles = []
        action_all = np.zeros(len(env.controlled_vehicles))
        for veh in vehicles.values():
            veh_step = step - veh.start_step
            # veh.route_positions[step - veh.start_step] is the position at the beginning of step
            # veh.accels[step - veh.start_step] is the accel command at step
            if veh_step < len(veh.accels) or veh.route_position < veh.route.segments[-1][1]:
                # assert abs(veh.route_position - veh.route_positions[veh_step]) < 1e-4
                veh.accel = veh.accels[veh_step]
            else:
                veh.accel = max_accel
            idx = env.controlled_vehicles.index(veh)
            action_all[idx] = veh.accel
            veh.route_position = pos = veh.route_position + veh.speed * dt
            veh.route_position_history.append(pos)

        for subzone in subzones.values():
            while len(subzone.reservations) > subzone.curr_reservation_index and step >= subzone.reservations[subzone.curr_reservation_index][2]:
                subzone.curr_reservation_index += 1
            if len(subzone.reservations) > subzone.curr_reservation_index and step >= subzone.reservations[subzone.curr_reservation_index][1]:
                subzone.reserving_vehicle = subzone.reservations[subzone.curr_reservation_index][0]
            else:
                subzone.reserving_vehicle = None
        env.step(tuple(action_all))

        for veh in vehicles.values():
            route = veh.route
            veh_step = step - veh.start_step
            new_speed = veh.speed
            veh.speed_history.append(new_speed)
            pos = veh.route_position

            segment = init_segment = veh.segment
            while True:
                next_segment = segment.next.get(route.id, None)
                if not next_segment:
                    break
                pos_start, pos_end = next_segment.position[route.id]
                if pos < pos_start:
                    break
                veh.segment = segment = next_segment
            if segment != init_segment:
                if isinstance(init_segment, Subzone):
                    init_segment.vehicle = None
                else:
                    init_segment.vehicles.remove(veh)
                new_segment_vehicles.append(veh)
            veh.segment_position = pos - segment.position[route.id][0]
        for veh in sorted(new_segment_vehicles, key=lambda x: x.segment_position):
            if isinstance(veh.segment, Subzone):
                assert veh.segment.vehicle is None, f'{veh} and {veh.segment.vehicle} in the same subzone!'
                veh.segment.vehicle = veh
            else:
                veh.segment.vehicles.append(veh)

        # Finished vehicles
        for route in routes.values():
            while route.vehicles and route.vehicles[0].route_position >= route.length:
                veh = route.vehicles.popleft()
                veh.finished = True
                finished_vehicles[veh.id] = vehicles.pop(veh.id)
                veh.segment.vehicles.remove(veh)

        stats['simulation_time'] = init_time + (time() - start_time)
        stats['global_cumulative_count'] = global_counter
        for direction, count in direction_counter.items():
            stats[f'direction_{direction}_cumulative_count'] = count
        for route in routes.values():
            stats[f'route_{route.id}_cumulative_count'] = route.id_count
        stats['curr_count'] = len(vehicles)
        for direction, count in queue.items():
            stats[f'curr_queue_{direction}_count'] = count
        for lane in lanes.values():
            stats[f'curr_lane_{lane.id}'] = len(lane.vehicles)
        logger.log(step, stats, stdout=method != 'fifo_async' and step % H_c == 0)
    finished_travel_time = np.mean([len(veh.route_position_history) for veh in finished_vehicles.values()]) * dt
    unfinished_travel_time = 2 * np.mean([len(veh.route_position_history) for veh in vehicles.values()]) * dt
    finished_delay = finished_travel_time - np.mean([veh.route.min_travel_time for veh in finished_vehicles.values()])
    unfinished_delay = unfinished_travel_time - np.mean([veh.route.min_travel_time for veh in vehicles.values()])
    return {**finished_vehicles, **vehicles}, subzones, dict(
        throughput=(len(finished_vehicles) + len(vehicles) / 2) / (H * dt / 3600),
        finished_count=len(finished_vehicles),
        finished_travel_time=finished_travel_time,
        finished_delay=finished_delay,
        unfinished_count=len(vehicles),
        unfinished_travel_time=unfinished_travel_time,
        unfinished_delay=unfinished_delay,
    )

render = False
render_image = False

# Defaults
seed = 0
lane_width = 4.5
car_width = 2
car_length = 5
lane_length = 250

max_speed = 13
right_speed = 4.5
left_speed = 6.5
u_speed = 3

directions = ['U', 'L', 'D', 'R']
lam = {'U': 1000, 'L': 1000, 'D': 2000, 'R': 1000} # veh/hr/lane
p = [0.6, 0.2, 0, 0.2] # straight, left, u, right
H = 1000 # Simulation horizon
dt = 0.1
init_speed = 5
max_accel = 2.6
max_decel = 4.5

H_p_padding = 50 # Constant should be fine
min_steps_to_reservation = 0 # Constant
H_c = 100 # Control horizon for synchronous methods (prioritized planning, MCTS, OBS)

use_arrival_times = True # [False, 'partial', True]
reserve_actual = False # [False, True]
method = 'fifo_async' # ['fifo_async', 'fifo', 'pp', 'mcts', 'obs']

use_approach_time_constraint = True
use_approach_time_min_depth = 0
num_mcts_simulations = 10
num_random_orders = 5
obs_order_threshold = 50
num_obs_orders = None

parser = argparse.ArgumentParser()
parser.add_argument('run_dir', type=Path)

if __name__ == '__main__':
    args = parser.parse_args()
    run_dir = args.run_dir
    config = (run_dir / 'config.yaml').load()
    print('Running with config')
    print(format_yaml(config))
    globals().update(config)
    spawn_seed = alg_seed = seed

    plan_vehicles_fn = dict(
        # Asynchronous, vehicles get assigned a trajectory when they enter the simulation
        fifo_async=plan_vehicles,

        # Synchronous, periodic replan every H_c steps
        fifo=lambda: plan_vehicles_synchronous(lambda *args: prioritized_planning(*args, num_random_orders=0)),
        pp=lambda: plan_vehicles_synchronous(lambda *args: prioritized_planning(*args, num_random_orders=num_random_orders)),
        mcts=lambda: plan_vehicles_synchronous(mcts),
        obs=lambda: plan_vehicles_synchronous(obs),
    )[method]
    alg_state = np.random.RandomState(alg_seed)

    logger = Logger(run_dir / 'stats.ftr')
    all_vehicles, subzones, final_stats = simulate()
    logger.save()

    (run_dir / 'final_stats.yaml').save(final_stats)
    # print(format_yaml(final_stats))
    np.savez_compressed(run_dir / 'vehicles.npz', **{veh.id: np.array(veh.route_position_history, dtype=np.float16) for veh in all_vehicles.values()})
    np.savez_compressed(run_dir / 'subzone_vehicles.npz', **{s_id: np.array([veh.id for veh, _, _ in subzone.reservations]) for s_id, subzone in subzones.items()})
    np.savez_compressed(run_dir / 'subzone_starts_ends.npz', **{s_id: np.array([[start, end] for _, start, end in subzone.reservations]) for s_id, subzone in subzones.items()})