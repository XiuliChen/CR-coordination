

import numpy as np



###########################################################
def calc_dis(p,q):
    #calculate the Euclidean distance between points p and q 
    if not isinstance(p, (np.ndarray)):
        p=np.array(p)
    if not isinstance(q, (np.ndarray)):
        q=np.array(q)

    return np.sqrt(np.sum((p-q)**2))

###########################################################
# used in the hand_model to generate trajectory
def mjtg(current, setpoint, frequency, move_time):
    trajectory = []
    trajectory_derivative = []
    timefreq = int(move_time * frequency)

    for time in range(1, timefreq):
        trajectory.append(
            current + (setpoint - current) *
            (10.0 * (time/timefreq)**3
             - 15.0 * (time/timefreq)**4
             + 6.0 * (time/timefreq)**5))

        trajectory_derivative.append(
            frequency * (1.0/timefreq) * (setpoint - current) *
            (30.0 * (time/timefreq)**2.0
             - 60.0 * (time/timefreq)**3.0
             + 30.0 * (time/timefreq)**4.0))

    return trajectory, trajectory_derivative
