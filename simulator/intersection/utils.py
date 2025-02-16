import numpy as np
import cvxpy as cp
import time
import math
from mixed_autonomy_mpc.util import Namespace
from collections import deque, Counter, defaultdict
from mixed_autonomy_mpc.mcts.tree.search import MCTS
from mixed_autonomy_mpc.mcts.tree.nodes import MCTSNode


from simulator.intersection.config import *

class Lane(Namespace):
    def __repr__(self):
        return f'Lane(id={self.id})'

class Subzone(Namespace):
    def __repr__(self):
        return f'Subzone(id={self.id})'

class Route(Namespace):
    def __repr__(self):
        return f'Route(id={self.id}, length={self.length}, segments={self.segments})'




##### uitil functions ##### 

def rotate(xy, rot90, center=[0, 0]):
    # Rotate xy coordinate(s) by multiples of 90 degrees around center
    theta = rot90 * np.pi / 2
    cos, sin = np.cos(theta), np.sin(theta)
    mat = np.array([[cos, -sin], [sin, cos]])
    return (mat @ (np.asarray(xy) - center).T).T + center

def init_traffic_network(lane_width=lane_width, lane_length=lane_length, left_speed=left_speed, right_speed=right_speed, max_speed=max_speed, u_speed=1):
    zone_indices = np.arange(4)
    directions = ['U', 'L', 'D', 'R']
    # Subconflict zones: each zone's id is defined as 'ab' where a in [0, 1, 2, 3] indicates the [lower right, upper right, upper left, lower left] square quadrant and b in [0, 1, 2, 3] indicates the [lower, right, upper, left] triangle within quadrant a.
    # E.g. id='03' indicates the lower right square quadrant's left triangle subzone
    subzone_infos = []
    for x in zone_indices:
        # center = rotate([lw2, -lw2], x)
        center = np.zeros(2)
        for y in zone_indices:
            # subzone_infos.append((f'{x}{y}', center + rotate([[0, 0], [-lw2, -lw2], [lw2, -lw2]], y)))
            subzone_infos.append((f'{x}{y}', center))
    subzones = {id: Subzone(id=id, position={}, next={}, prev={}, vehicle=None, reservations=[], shape=shape) for id, shape in subzone_infos}

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
            segments = [(lanes[f'{from_d}0'], 0, lane_length, None)]
            def extend_segments(*args, speed=None):
                offset = segments[-1][2]
                segments.extend([(z, offset + a, offset + b, speed) for z, a, b in args])
            if j == 0: # straight
                # shape = [(lw2, -lllw), (lw2, lllw)]
                for zone_index in [0, 1]:
                    extend_segments(
                        (ego_subzone(zone_index, 0), 0, 2.5*lane_width),
                        (ego_subzone(zone_index, 1), 0, 2.5*lane_width),
                        (ego_subzone(zone_index, 3), 0, 2.5*lane_width),
                        (ego_subzone(zone_index, 2), 0, 2.5*lane_width),
                        speed=max_speed,
                    )
            elif j == 1: # left turn
                # shape = [(lw2, -lllw), (lw2, -lane_width), (-lane_width, lw2), (-lllw, lw2)]
                left_zone_half_len = 2*np.pi*3*lane_width/8
                extend_segments(
                    (ego_subzone(0, 0), 0, left_zone_half_len),
                    (ego_subzone(0, 3), 0, left_zone_half_len),
                    (ego_subzone(3, 1), 0, left_zone_half_len),
                    (ego_subzone(3, 2), left_zone_half_len, 2*left_zone_half_len),
                    (ego_subzone(2, 0), left_zone_half_len, 2*left_zone_half_len),
                    (ego_subzone(2, 3), left_zone_half_len, 2*left_zone_half_len),
                    speed=left_speed,
                )

            elif j == 2: # u-turn (not used at moment)
                # shape = [(lw2, -lllw), (lw2, -lane_width), (0, -lw2), (-lw2, -lane_width), (-lw2, -lllw)]
                # extend_segments(
                #     (ego_subzone(0, 0), 0, diag),
                #     (ego_subzone(0, 3), diag, 2 * diag),
                #     (ego_subzone(3, 1), 2 * diag, 3 * diag),
                #     (ego_subzone(3, 0), 3 * diag, 4 * diag),
                #     speed=u_speed,
                # )
                continue

            elif j == 3: # right turn
                right_zone_half_len = 2*np.pi*2*lane_width/8
                # shape = [(lw2, -lllw), (lw2, -lane_width), (lane_width, -lw2), (lllw, -lw2)]
                extend_segments(
                    (ego_subzone(0, 0), 0, right_zone_half_len),
                    (ego_subzone(0, 1), right_zone_half_len, 2 * right_zone_half_len),
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
    

def optimize_speed(veh, H_p, max_positions, step, min_positions=None, target_speeds=None, max_step_speed=None, max_position_speed=None):
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
        x[1:] - x[:-1] == (v[:-1] + v[1:]) / 2 * dt,
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

def compute_vehicle_step_info(veh, step):
    # if veh.get('last_info_step') == step:
    #     return True
    route = veh.route
    from_lane, _, veh.from_lane_pos_end, _ = route.segments[0]
    veh.approach_distance = veh.from_lane_pos_end - veh.segment_position
    veh.max_crossing_speed = veh.crossing_speed = route.subzones[0][3]
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

def plan_vehicle(step, veh, use_leaders=False, use_arrival_times=True, reserve_actual=False, allow_infeasible=True,
    get_last_reservation=lambda subzone, step: subzone.reservations[-1] if subzone.reservations else (None, step, step),
    add_reservation=lambda subzone, reservation: subzone.reservations.append(reservation)):

    assert compute_vehicle_step_info(veh, step)
    
    route = veh.route
    from_lane, _, from_lane_pos_end, _ = route.segments[0]
    to_lane, to_lane_pos_start, to_lane_pos_end, _ = route.segments[-1]

    # Compute necessary information for the arrival and departure times at each subzone
    other_end_steps, subzone_start_offsets, subzone_end_offsets = [], [], []
    for subzone, pos_start, pos_end, _ in route.subzones:
        other_end_steps.append(get_last_reservation(subzone, step)[-1] - step)
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
                leader_constraints[0, from_leader_end_step] = from_leader_rear_pos[:from_leader_end_step]
        if to_lane.arrivals:
            to_leader = to_lane.arrivals[-1]
            to_leader_to_lane_pos_start, _ = to_lane.position[to_leader.route.id]
            to_leader_rear_pos = to_leader.route_positions[step - to_leader.start_step:] - car_length
            if len(to_leader_rear_pos):
                assert to_leader_rear_pos[-1] >= to_leader_to_lane_pos_start
                leader_start_step = np.argmax(to_leader_rear_pos >= to_leader_to_lane_pos_start)
                min_len = min(1 + H_p, len(to_leader_rear_pos))
                leader_constraints[leader_start_step, min_len] = to_leader_rear_pos[leader_start_step: min_len] - to_leader_to_lane_pos_start + to_lane_pos_start

        max_positions, min_positions = np.empty(1 + H_p), np.empty(1 + H_p)
#         additional_delay_steps = iter([1, 1, 2, 4, 8])
        additional_delay_steps = iter([1, 1] if use_arrival_times is False else [])
        while True: # Adjust until feasible
            max_positions[:] = np.inf
            min_positions[:] = 0
            for (start_step, end_step), positions in leader_constraints.items():
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
            route_positions, speeds, accels = optimize_speed(veh, H_p, max_positions, step, min_positions=min_positions, **optimize_kwargs)
            if route_positions is None:
                try:
                    next_delay_steps = next(additional_delay_steps)
                except StopIteration:
                    if allow_infeasible: return False
                    # veh.update({k: v for k, v in locals().items() ixf k not in ['route_positions', 'speeds', 'accels', 'start_step']})# For debug
                    raise RuntimeError(veh, 'infeasible')
                print(veh, f'infeasible, add {next_delay_steps} delay steps')
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

def plan_vehicles_ordered(step, vehicles, order=None, *args, **kwargs):
    for veh in (order or [veh for veh in vehicles.values() if len(veh.accels) == 0]):
        if not plan_vehicle(step, veh, use_leaders=True, use_arrival_times=use_arrival_times, reserve_actual=reserve_actual, **kwargs):
            return False
    return order, True

def try_plan_vehicles_ordered(step, lanes, subzones, order, vehicles, order_id_set=None, **kwargs):
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
    if plan_vehicles_ordered(step, vehicles, order=order, allow_infeasible=True, **kwargs): # Success
        return True
    # Failed, revert states
    for veh in order:
        # for k in ['route_positions', 'speeds', 'accels']:
            # veh[k] = veh[k + '_']
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
    return False

# def plan_vehicles(step, vehicles, order=None, *args, **kwargs):
#     start_time = time()
#     success = True
#     for veh in (order or [veh for veh in vehicles.values() if len(veh.accels) == 0]):
#         if not plan_vehicle(veh, use_leaders=True, use_arrival_times=use_arrival_times, reserve_actual=reserve_actual, **kwargs):
#             success = False
#             break
#     return success, dict(async_plan_time=time() - start_time)

# def try_plan_vehicles_ordered(step, lanes, subzones, order, vehicles, order_id_set=None, **kwargs):
#     order_id_set = order_id_set or {veh.id for veh in order}
#     # Save in case of revert
#     for lane in lanes.values():
#         lane.arrivals_ = []
#         while lane.arrivals and lane.arrivals[-1].id in order_id_set:
#             lane.arrivals_.append(lane.arrivals.pop())
#     for subzone in subzones.values():
#         subzone.reservations_ = []
#         while subzone.reservations and subzone.reservations[-1][0].id in order_id_set:
#             subzone.reservations_.append(subzone.reservations.pop())
#     for veh in order:
#         for k in ['route_positions', 'speeds', 'accels']:
#             veh[k + '_'] = veh[k]
    
#     # Low level replan
#     success, stats = plan_vehicles(step, vehicles, order=order, allow_infeasible=True, **kwargs)
#     stats['sync_plan_time_low'] = stats.pop('async_plan_time')
#     if not success: # Failed, revert states
#         for veh in order:
#             for k in ['route_positions', 'speeds', 'accels']:
#                 veh[k] = veh[k + '_']
#         for lane in lanes.values():
#             while lane.arrivals and lane.arrivals[-1].id in order_id_set:
#                 lane.arrivals.pop()
#             lane.arrivals.extend(reversed(lane.arrivals_))
#         for subzone in subzones.values():
#             while subzone.reservations and subzone.reservations[-1][0].id in order_id_set:
#                 subzone.reservations.pop()
#             subzone.reservations.extend(reversed(subzone.reservations_))
#     return success, stats


def setup_high_level(step, curr_order, lanes, subzones):
    in_lanes = [lane for lane in lanes.values() if lane.id.endswith('0')]
    order_map = {veh.id: i for i, veh in enumerate(curr_order)}
    replan_vehicles = {veh.id: veh for lane in in_lanes for veh in lane.vehicles if veh.id in order_map}
    last_infeasible = -1
    for veh_id, veh in replan_vehicles.items():
        if not compute_vehicle_step_info(veh, step):
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
        get_last_reservation=lambda subzone, step: subzone.last_reservation,
        add_reservation=lambda subzone, reservation: subzone.update(last_reservation=reservation),
    )

def get_nondominated_min_subzone_approach(vehs):
    approach_times = np.array([veh.subzone_approach_times for veh in vehs])
    no_approach = np.isnan(approach_times)
    closer_to_subzone = no_approach[:, None, :] | no_approach[None, :, :] | (approach_times[:, None, :] <= approach_times[None, :, :]) # (num_queues, num_queues, num_subzones)
    closest = closer_to_subzone.all(axis=(1, 2))
    action_mask = closest if closest.any() else closer_to_subzone.all(axis=1).any(axis=1)
    return np.where(action_mask)[0]

def plan_order_approach_times(subzones, order, reservation_fns, approach_times=None, min_approach_times=None):
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


# def plan_vehicles_synchronous(get_order_fn):
#     if step % H_c != 0:
#         return True, plan_vehicles()[1]

#     fifo_order, replan_vehicles, replan_lane_vehs, reservation_fns = setup_high_level()
#     plan_vehicle.count = 0
#     fifo_approach_times, fifo_min_approach_times = plan_order_approach_times(fifo_order, reservation_fns)
#     fifo_delay = (fifo_approach_times - fifo_min_approach_times).mean()
    
#     start_time = time()
#     plan_vehicle.count = 0
#     order, stats = get_order_fn(fifo_order, replan_vehicles, replan_lane_vehs, reservation_fns)
#     stats.update(fifo_delay=fifo_delay, sync_plan_time_high=time() - start_time, plan_vehicle_count=plan_vehicle.count)
#     if order is None:
#         print('step', step, 'no order obtained, falling back to previous trajectories')
#         return False, dict(**stats, **plan_vehicles()[1])
#     approach_times, min_approach_times = plan_order_approach_times(order, reservation_fns)
#     stats['delay'] = (approach_times - min_approach_times).mean()

#     actual_approach_times = np.array([np.argmax(veh.route_positions[step - veh.start_step:] >= veh.route.subzones[0][1]) * dt for veh in order])
#     stats['actual_delay'] = actual_approach_times.mean() - min_approach_times.mean()

#     success, low_level_stats = try_plan_vehicles_ordered(order=order, order_id_set=replan_vehicles)
#     stats.update(low_level_stats)
#     if not success:
#         print('step', step, 'order infeasible, falling back to previous trajectories')
#         return False, {**stats, **plan_vehicles()[1]}
#     curr_order[:] = order

#     return curr_order, stats

def plan_vehicles_mcts(step, vehicles, curr_order, lanes, subzones):
    if step % H_c != 0:
        return curr_order, plan_vehicles_ordered(step, vehicles)
    replan_vehs_fifo, replan_vehicles, replan_lane_vehs, MCTSState.reservation_fns = setup_high_level(step, curr_order, lanes, subzones)
    state = MCTSState(replan_lane_vehs, subzones, step)
    while len(state.order) < len(replan_vehicles):
        root = MCTSNode(state)
        mcts = MCTS(root)
        best_node = mcts.best_action(num_simulations=num_mcts_simulations)
        state = best_node.state
    if not try_plan_vehicles_ordered(step, lanes, subzones, state.order, vehicles, order_id_set=replan_vehicles):
        return curr_order, plan_vehicles_ordered(step, vehicles)
    curr_order[:] = state.order

    return curr_order, True

def plan_vehicles_prioritized_planning(step, vehicles, curr_order, lanes, subzones, num_random_orders=0):
    if step % H_c != 0:
        return curr_order, plan_vehicles_ordered(step, vehicles)
    fifo_order, replan_vehicles, replan_lane_vehs, reservation_fns = setup_high_level(step, curr_order, lanes, subzones )
    orders = [fifo_order]
    for _ in range(num_random_orders):
        # To construct an order use MCTS paper's two heuristics based on lane ordering and subzone approach times
        vehicle_queues = [lane_vehs[::-1] for lane_vehs in replan_lane_vehs if len(lane_vehs)]
        orders.append(order := [])
        while len(vehicle_queues):
            actions = get_nondominated_min_subzone_approach([lane_vehs[-1] for lane_vehs in vehicle_queues]) if use_approach_time_constraint else len(vehicle_queues)
            i = np.random.choice(actions)
            order.append(vehicle_queues[i].pop())
            if len(vehicle_queues[i]) == 0:
                vehicle_queues.pop(i)
    unique_orders = {'_'.join(veh.id.split('_')[0] for veh in order): order for order in orders}
    orders = list(unique_orders.values())

    approach_times = np.zeros((len(orders), len(replan_vehicles)))
    min_approach_times = np.zeros_like(approach_times)
    for i, order in enumerate(orders):
        for subzone in subzones.values():
            subzone.last_reservation = subzone.last_reservation_

        for j, veh in enumerate(order):
            min_approach_times[i, j] = veh.min_approach_time
            plan_vehicle(step, veh, use_leaders=False, **reservation_fns)
#             if veh.cross_available_step * dt > veh.max_approach_time: # Vehicle cannot stop in time for the current crossing order
#                 approach_times[i, j] = np.inf
#                 break
            approach_times[i, j] = veh.cross_start_step * dt
    delays = approach_times - min_approach_times
    total_delays = delays.sum(axis=1)
    best = np.argmin(total_delays)
    best_order = orders[best]

#         if len(total_delays) > 1 and total_delays[0] > total_delays[1:].min():
#             print('step', step, 'FIFO', total_delays[0], 'Rest best', np.round(total_delays[1:].min(), 2), 'Rest mean', np.round(total_delays[1:].min(), 2))
    if np.isinf(total_delays).all():
        print('step', step, f'All orders infeasible. Skipping replan')
        return curr_order, plan_vehicles_ordered(step, vehicles)

    if not try_plan_vehicles_ordered(step, lanes, subzones, best_order, vehicles, order_id_set=replan_vehicles):
        return curr_order, plan_vehicles_ordered(step, vehicles)
#         actual_approach_times = [np.argmax(veh.route_positions[step - veh.start_step:] >= veh.route.subzones[0][1]) * dt for veh in best_order]
#         print('Estimated total_delay:', round(total_delays[best], 1), 'Actual total_delay:', round(sum(actual_approach_times) - min_approach_times[best].sum(), 1))
    curr_order[:] = best_order

    return curr_order, True

def convert_to_Highway(from_d, to_d):

    if from_d == "U":
        if to_d == "U":
            return "0", "2"
        elif to_d == "L":
            return "0", "1"
        elif to_d == "R":
            return "0", "3"
    
    elif from_d == "L":
        if to_d == "U":
            return "3", "2"
        elif to_d == "L":
            return "3", "1"
        elif to_d == "D":
            return "3", "0"

    elif from_d == "D":
        if to_d == "D":
            return "2", "0"
        elif to_d == "L":
            return "2", "1"
        elif to_d == "R":
            return "2", "3"

    elif from_d == "R":
        if to_d == "U":
            return "1", "2"
        elif to_d == "R":
            return "1", "3"
        elif to_d == "D":
            return "1", "0"

class MCTSState:
    def __init__(self, lane_vehs, subzones, step, next_lane_veh_idxs=None, order=[]):
        self.lane_vehs = lane_vehs
        self.next_lane_veh_idxs = next_lane_veh_idxs or [0] * len(lane_vehs)
        self.order = order
        self.subzones = subzones
        self.step = step

    def get_legal_actions(self):
        vehs = {i: lane_vehs_i[idx] for i, (lane_vehs_i, idx) in enumerate(zip(self.lane_vehs, self.next_lane_veh_idxs)) if idx < len(lane_vehs_i)}
        return list(np.array(list(vehs.keys()))[get_nondominated_min_subzone_approach(list(vehs.values()))] if use_approach_time_constraint else vehs.keys())

    def move(self, action):
        lane_vehs_a, idx_a = self.lane_vehs[action], self.next_lane_veh_idxs[action]
        new_order = self.order + [lane_vehs_a[idx_a]]
        new_next_lane_veh_idxs = self.next_lane_veh_idxs.copy()
        new_next_lane_veh_idxs[action] += 1
        return MCTSState(self.lane_vehs, self.subzones, self.step, new_next_lane_veh_idxs, new_order)

    def is_terminal(self):
        return all(idx == len(lane_vehs_i) for lane_vehs_i, idx in zip(self.lane_vehs, self.next_lane_veh_idxs))

    def rollout_result(self):
        for subzone in self.subzones.values():
            subzone.last_reservation = subzone.last_reservation_

        self.min_approach_times = np.zeros(len(self.order))
        self.approach_times = self.min_approach_times.copy()
        for j, veh in enumerate(self.order):
            self.min_approach_times[j] = veh.min_approach_time
            plan_vehicle(self.step, veh, use_leaders=False, **MCTSState.reservation_fns)
            if veh.cross_available_step * dt > veh.max_approach_time: # Vehicle cannot stop in time for the current crossing order
                self.approach_times[j] = np.inf
                break
            self.approach_times[j] = veh.cross_start_step * dt
        self.veh_delays = self.approach_times - self.min_approach_times
        self.total_delay = self.veh_delays.sum()
        return -self.total_delay
