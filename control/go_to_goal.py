import numpy as np

""" Implement simple P control. """


def Pcontrol(current_q, Pgain, ref):
    return Pgain * (ref - current_q)


def Pcontrol_TimeVarying(current_q, ref, v0=1., beta=1.):
    error = (ref - current_q)
    norm_er_pos = np.hypot(error[0], error[1])
    kP = v0 * (1 - np.exp(-beta * norm_er_pos)) / norm_er_pos
    return kP * error


class waypoint_planner():
    def __init__(self, waypoints, orientations=None):

        # Expecting numpy array M x 3
        # M rows of 3 dimensional positions
        self.__wp = waypoints
        self.__wp_num = waypoints.shape[0]

        if orientations is None:
            self.__wp_orient = np.zeros(self.__wp_num)
        else:
            self.__wp_orient = orientations

        self.__current_num = 0

    # Check whether the position is within switching range
    # return status if wp is changed, wp position, and wp orientation
    def check_wp_status(self, pos, switch_dist_range=1., th=None, switch_angle_range=np.pi/8):
        is_switch_wp = False
        if self.__current_num < self.__wp_num - 1:

            dist = np.linalg.norm(pos - self.__wp[self.__current_num])
            is_within_dist = dist < switch_dist_range

            is_within_angle = True
            if th is not None: 
                angle_diff = th - self.__wp_orient[self.__current_num]
                angle_diff = (angle_diff + np.pi) % (2*np.pi) - np.pi
                is_within_angle = np.absolute(angle_diff) < switch_angle_range

            is_switch_wp = is_within_dist and is_within_angle
            if is_switch_wp: # switch to new wp
                self.__current_num += 1

        return is_switch_wp, self.__wp[self.__current_num], \
            self.__wp_orient[self.__current_num]
