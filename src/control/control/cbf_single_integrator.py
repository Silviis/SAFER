import numpy as np

from qpsolvers import Problem, solve_problem


class cbf_si():
    def __init__(self, P = None, q = None):
        # Total number of decision variables
        self._var_num = 3 # default 3: ux, uy, uz

        self.reset_cbf()

    def reset_cbf(self):
        # initialize G and h, Then fill it afterwards
        self.constraint_G = None
        self.constraint_h = None
        self.cbf_values = None

    def __set_constraint(self, G_mat, h_mat):
        if self.constraint_G is None:
            self.constraint_G = G_mat
            self.constraint_h = h_mat
        else:
            self.constraint_G = np.append(self.constraint_G, G_mat, axis=0)
            self.constraint_h = np.append(self.constraint_h, h_mat, axis=0)


    def compute_safe_controller(self, u_nom, P = None, q = None, speed_limit = None):

        if (P is None) and (q is None): P, q = 2*np.eye(3), -2*u_nom

        if self.constraint_G is not None:
            def_ublb = np.inf
            lb = np.ones(self._var_num)*(-def_ublb)
            ub = np.ones(self._var_num)*(def_ublb)

            if speed_limit is not None:
                array_limit = np.ones(3)* speed_limit
                lb[:3], ub[:3] = -array_limit, array_limit

            G_mat = self.constraint_G.copy()
            h_mat = self.constraint_h.copy()
            opt_tolerance = 1e-8

            qp_problem = Problem(P, q, G_mat, h_mat, lb = lb, ub = ub)
            qp_problem.check_constraints()
            solution = solve_problem(
                qp_problem,
                solver="daqp",
                dual_tol=opt_tolerance,
                primal_tol=opt_tolerance,
            )
            sol = solution.x

            if sol is None:
                print('WARNING QP SOLVER [no solution] stopping instead')
                u_star = np.array([0., 0., 0.])
            elif not solution.is_optimal(opt_tolerance):
                print('WARNING QP SOLVER [not optimal] stopping instead')
                u_star = np.array([0., 0., 0.])
            else:
                u_star = np.array([sol[0], sol[1], sol[2]])
            

        else: # No constraints imposed
            u_star = u_nom.copy()

        return u_star


    # ADDITION OF CONSTRAINTS
    # -----------------------------------------------------------------------------------------------------------
    def add_avoid_static_circle(self, pos, obs, ds, gamma=10, power=3):
        # h = norm2( pos - obs )^2 - norm2(ds)^2 > 0
        vect = pos - obs
        h_func = np.power(np.linalg.norm(vect), 2) - np.power(ds, 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(-2*vect.reshape((1,3)), gamma*np.power(h_func, power).reshape((1,1)))

        return h_func


    def add_maintain_distance_with_epsilon(self, pos, obs, ds, epsilon, gamma=10, power=3):
        vect = pos - obs
        # h = norm2( ds + epsilon )^2 - norm2( pos - obs )^2 > 0
        h_func_l = np.power((ds+epsilon), 2) - np.power(np.linalg.norm(vect), 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(2*vect.reshape((1,3)), gamma*np.power(h_func_l, power).reshape((1,1)))

        # h = norm2( pos - obs )^2 - norm2( ds - epsilon )^2 > 0
        h_func_u = np.power(np.linalg.norm(vect), 2) - np.power((ds-epsilon), 2)
        # -(dh/dpos)^T u < gamma(h)
        self.__set_constraint(-2*vect.reshape((1,3)), gamma*np.power(h_func_u, power).reshape((1,1)))

        return h_func_l, h_func_u


    def add_avoid_static_ellipse(self, pos, obs, theta, major_l, minor_l, gamma=10, power=3):
        # h = norm2( ellipse*[pos - obs] )^2 - 1 > 0
        theta = theta if np.ndim(theta) == 0 else theta.item()
        # TODO: assert a should be larger than b (length of major axis vs minor axis)
        vect = pos - obs # compute vector towards pos from centroid
        # rotate vector by -theta (counter the ellipse angle)
        # then skew the field due to ellipse major and minor axis
        # the resulting vector should be grater than 1
        # i.e. T(skew)*R(-theta)*vec --> then compute L2norm square
        ellipse = np.array([[2./major_l, 0, 0], [0, 2./minor_l, 0], [0, 0, 1]]) \
            @ np.array([[np.cos(-theta), -np.sin(-theta), 0], [np.sin(-theta), np.cos(-theta), 0], [0, 0, 1]], dtype=object)
        h_func = np.power(np.linalg.norm( ellipse @ vect.T ), 2) - 1
        # -(dh/dpos)^T u < gamma(h)
        # -(2 vect^T ellipse^T ellipse) u < gamma(h)
        G = -2*vect @ ( ellipse.T @ ellipse )
        self.__set_constraint( G.reshape((1,3)), gamma*np.power(h_func, power).reshape((1,1)) )

        return h_func


    def add_velocity_bound(self, vel_limit):
        G = np.vstack((np.eye(3), -np.eye(3)))
        h = np.ones([6, 1]) * vel_limit
        self.__set_constraint( G, h )

    # TODO: add area with boundary    

    @staticmethod
    def is_between_equal(angle, min_val, max_val):
        """
        Check if the angle is in the radian region of [min_val, max_val]
        π → -π leap checking involved

        :param angle: scalar value, radian angular input
        :param min_val: scalar value, minimum radian angular value
        :param max_val: scalar value, maximum radian angular value
        :return: boolean, True if in between, False otherwise
        """
        return (angle >= min_val) & (angle <= max_val) if (max_val > min_val) \
            else (angle >= min_val) | (angle <= max_val)


    def add_avoid_lidar_detected_obs(self, obs_pos, pos_i, kappa, ds, gamma=10, power=3):

        """
        Process the obstacle sensing data from robots and add constraints

        :param obs_pos: Nx3 array, stacked obstacle positions detected by i-th robot
        :param pos_i: 1x3 array, i-th robot position
        :param kappa: scalar value, kappa
        :param ds: scalar value, minimum distance to obstacle
        :param gamma: scalar value, coefficient of gamma function
        :param power: scalar value, degree of gamma function
        :return h_func: scalar values, safety estimations
        """

        # Default value of return if obs_pos is empty
        min_h = np.nan

        # Process the obstacle detected points
        data_num = obs_pos.shape[0]
        if data_num > 0:
            # Identify obstacles position (in polar coordinate of world frame)
            vec_iobs = obs_pos - pos_i
            obst_range = np.linalg.norm(vec_iobs, axis=1)

            h_obs = obst_range ** 2 - ds ** 2
            min_h = np.min(h_obs)
            # Determine which to compute
            is_computed = h_obs < min_h + kappa

            gamma_h = gamma * np.power(min_h, power).reshape((1, 1))

            for i in range(data_num):
                if is_computed[i]:
                    vect_extended = 2 * vec_iobs[i, :].reshape((1, 3))
                    self.__set_constraint(vect_extended, gamma_h)

        return min_h