"""
This Module contains drone class to simulate drone dynamics
"""


# ************************************************************************* #
#                          Import statements                                #
# ************************************************************************* #
from numpy import array, radians, pi, arange, zeros, size, sqrt
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlabel, ylabel, grid, axhline, figure, gca, show
from matplotlib.ticker import StrMethodFormatter
from munch import Munch
from RigidBodyModel import RigidBodyModel
from vedo_model import QuadVedoModel


# ************************************************************************* #
#                         RPM to rps converter                              #
# ************************************************************************* #
def rpm_to_rps(rpm: float = 0.0):
    return 2 * pi * rpm / 60.


# ************************************************************************* #
#                         RPS to rpm converter                              #
# ************************************************************************* #
def rps_to_rpm(rps: float = 0.0):
    return (rps * 60.) / (2 * pi)


# initialize rigid body model
rbd_model = RigidBodyModel()


# ************************************************************************* #
#                          RK4 STEP Algorithm                               #
# ************************************************************************* #
def rk4_step1(func, y0, t0, dt, params):
    k1 = dt * func(t0, y0, params)
    k2 = dt * func(t0 + 0.5 * dt, y0 + 0.5 * k1, params)
    k3 = dt * func(t0 + 0.5 * dt, y0 + 0.5 * k2, params)
    k4 = dt * func(t0 + dt, y0 + k3, params)

    ynew = y0 + (1. / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

    return ynew


# ************************************************************************* #
#                          PID input computer                               #
# ************************************************************************* #
def pid_force1(t, err, derr_dt, int_err, kp, kd, ki):
    return (kp * err) + (kd * derr_dt) + (ki * int_err)


# ************************************************************************* #
#                               Drone Class                                 #
# ************************************************************************* #
class DroneSim(object):

    def __init__(self, params=None):
        if not params and type(params) != dict:
            print("""Please input drone parameters... Relevant parameters are
            m --> mass
            
            """)
            return

        self.w_to_Loads_matrix = None
        self.loads_to_w_matrix = None
        self.plot_no = 1

        self.plot_dict = {
            'u': (0, 'x velocity(m/s)'),
            'v': (1, 'y velocity(m/s)'),
            'w': (2, 'z velocity(m/s)'),
            'p': (3, 'roll rate(rad/s)'),
            'q': (4, 'pitch rate(rad/s)'),
            'r': (5, 'yaw rate(rad/s)'),
            'phi': (6, 'Roll angle(deg)'),
            'tht': (7, 'pitch Angle(deg)'),
            'psi': (8, 'yaw Angle(deg)'),
            'xe': (9, 'x (m)'),
            'ye': (10, 'y (m'),
            'ze': (11, 'Altitude (m'),
            'rpm': (12, 'RPM')
        }

        self.params = params
        self.t = arange(0, self.params['tf'], self.params['dt'])

        self.n_states = 12  # Number of states
        self.n_inputs = 4  # Number of inputs
        self.x = zeros((size(self.t), self.n_states))  # time history of state vectors
        self.inp = zeros((size(self.t), self.n_inputs))  # time history of input vectors

        self.get_rpm_to_loads_matrix()

    def get_rpm_to_loads_matrix(self):
        km = self.params['km']
        bm = self.params['bm']
        self.w_to_Loads_matrix = array([
            [km, km, km, km],
            [km * self.params['d1y'], km * self.params['d2y'], km * self.params['d3y'], km * self.params['d4y']],
            [-km * self.params['d1x'], -km * self.params['d2x'], -km * self.params['d3x'], -km * self.params['d4x']],
            [bm, bm, -bm, -bm]
        ])

    def get_rpm_given_loads(self, Ft, L, M, N):
        """
        This function computes RPM for a specific Thrust, L, M, and N
        """
        self.loads_to_w_matrix = inv(self.w_to_Loads_matrix)
        w1_sq, w2_sq, w3_sq, w4_sq = self.loads_to_w_matrix @ array([Ft, L, M, N])
        return sqrt(w1_sq) * 60 / (2 * pi), sqrt(w2_sq) * 60 / (2 * pi), sqrt(w3_sq) * 60 / (2 * pi), sqrt(
            w4_sq) * 60 / (2 * pi)

    def get_hover_rpm(self):
        """
        This function returns the RPM at which the Thrust force exactly
        balances the weight of the UAM for the given configuration
        """

        w1, w1, w1, w1 = self.get_rpm_given_loads(self.params['m'] * self.params['g'], 0., 0., 0.)
        return w1

    def eval_eq_of_mot(self, t, y, params, raw_loads=False):
        """
        This method evaluates the non linear equations of motions given all the
        parameters of the system
        """

        ## expand the params
        mv = params['m']
        Ixxv = params['Ixx']
        Iyyv = params['Iyy']
        Izzv = params['Izz']
        gv = params['g']

        if not raw_loads:
            w1 = rpm_to_rps(params['w1'])
            w2 = rpm_to_rps(params['w2'])
            w3 = rpm_to_rps(params['w3'])
            w4 = rpm_to_rps(params['w4'])

            Ftv, Lv, Mv, Nv = self.w_to_Loads_matrix @ array([w1 ** 2, w2 ** 2, w3 ** 2, w4 ** 2])
        else:
            Ftv = params['Ft']
            Lv = params['L']
            Mv = params['M']
            Nv = params['N']
        # print(Ftv/(mv*gv))

        uev = y[0]
        vev = y[1]
        wev = y[2]
        pv = y[3]
        qv = y[4]
        rv = y[5]
        phiv = y[6]
        thtv = y[7]
        psiv = y[8]
        # xEv                                = y[9]
        # yEv                                = y[10]
        # zEv                                = y[11]

        output = zeros(12)
        for i in range(12):
            output[i] = rbd_model.eq_of_mot_func[i](uev, vev, wev, pv, qv, rv, phiv, thtv, psiv, mv, Ixxv, Iyyv, Izzv,
                                                     gv, Ftv, Lv, Mv, Nv)

        return output

    def getEulerAngleRate(self, euler_param: str = 'thtd', p: float = 0., q: float = 0.,
                          r: float = 0., phi: float = 0., tht: float = 0., psi: float = 0.):
        """
        This method evaluates phid, thtd, psid given p, q and r, phi, tht, psi
        """

        if euler_param == 'phid':
            return rbd_model.eq_of_mot_func[6](0., 0., 0., p, q, r, phi, tht, psi, 0., 1., 1., 1., 9.81, 0., 0., 0.,
                                                0.)

        elif euler_param == 'thtd':
            return rbd_model.eq_of_mot_func[7](0., 0., 0., p, q, r, phi, tht, psi, 0., 1., 1., 1., 9.81, 0., 0., 0.,
                                                0.)

        elif euler_param == 'psid':
            return rbd_model.eq_of_mot_func[8](0., 0., 0., p, q, r, phi, tht, psi, 0., 1., 1., 1., 9.81, 0., 0., 0.,
                                                0.)

    def time_simulate(self):
        if 'X0' in list(self.params.keys()):
            X0 = self.params['X0']
        else:
            X0 = zeros(12)
        self.x[0, :] = X0

        for t_idx in range(len(self.t) - 1):

            y_tmp = rk4_step1(self.eval_eq_of_mot, self.x[t_idx, :], self.t[t_idx], self.params['dt'],
                              self.params)

            # restricting altitude to physical solutions
            if y_tmp[11] < 0:
                break

            self.x[t_idx + 1, :] = y_tmp

        print('*******************Solver ran successfully********************')

    def move_to_xyz(self, xd: float = 0., yd: float = 0., zd: float = 0., psi_des: float = 0., conv_tol: float = 0.01):

        if 'X0' in list(self.params.keys()):
            X0 = self.params['X0']
        else:
            X0 = zeros(12)
        y_tmp, y_new = X0.copy(), X0.copy()
        self.x[0, :] = X0

        int_err_x = 0.
        int_err_y = 0.
        int_err_z = 0.
        int_err_phi = 0.
        int_err_tht = 0.
        int_err_psi = 0.
        psi_des = psi_des * pi / 180

        for t_idx in range(len(self.t) - 1):

            t0 = self.t[t_idx]

            ## Outer loop runs first to get desired attitude for target pos
            curr_u = y_tmp[0]
            curr_v = y_tmp[1]
            curr_w = y_tmp[2]
            curr_p = y_tmp[3]
            curr_q = y_tmp[4]
            curr_r = y_tmp[5]
            curr_x = y_tmp[9]
            curr_y = y_tmp[10]
            curr_z = y_tmp[11]
            curr_phi = y_tmp[6]
            curr_tht = y_tmp[7]
            curr_psi = y_tmp[8]

            err_x = curr_x - xd
            derr_x_dt = curr_u
            int_err_x = int_err_x + (self.params['dt'] / self.params['tf']) * err_x
            theta_des = -pid_force1(t0, err_x, derr_x_dt, int_err_x, self.params['kp_x'], self.params['kd_x'],
                                    self.params['ki_x'])

            err_y = curr_y - yd
            derr_y_dt = curr_v
            int_err_y = int_err_y + (self.params['dt'] / self.params['tf']) * err_y
            phi_des = -pid_force1(t0, err_y, derr_y_dt, int_err_y, self.params['kp_y'], self.params['kd_y'],
                                  self.params['ki_y'])

            err_z = curr_z - zd
            derr_dt_z = curr_w
            int_err_z = int_err_z + (self.params['dt'] / self.params['tf']) * err_z
            Ft = (self.params['m'] * self.params['g']) - pid_force1(t0, err_z, derr_dt_z, int_err_z,
                                                                    self.params['kp_h'], self.params['kd_h'],
                                                                    self.params['ki_h'])

            err_phi = curr_phi - phi_des
            derr_dt_phi = self.getEulerAngleRate(euler_param='phid', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_phi = int_err_phi + (self.params['dt'] / self.params['tf']) * err_phi
            L = -pid_force1(t0, err_phi, derr_dt_phi, int_err_phi, self.params['kp_phi'], self.params['kd_phi'],
                            self.params['ki_phi'])

            err_tht = curr_tht - theta_des
            derr_dt_tht = self.getEulerAngleRate(euler_param='thtd', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_tht = int_err_tht + (self.params['dt'] / self.params['tf']) * err_tht
            M = -pid_force1(t0, err_tht, derr_dt_tht, int_err_tht, self.params['kp_tht'], self.params['kd_tht'],
                            self.params['ki_tht'])

            err_psi = curr_psi - psi_des
            derr_dt_psi = self.getEulerAngleRate(euler_param='psid', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_psi = int_err_psi + (self.params['dt'] / self.params['tf']) * err_psi
            N = -pid_force1(t0, err_psi, derr_dt_psi, int_err_psi, self.params['kp_psi'], self.params['kd_psi'],
                            self.params['ki_psi'])

            self.params['Ft'], self.params['L'], self.params['M'], self.params['N'] = Ft, L, M, N
            k1 = self.params['dt'] * self.eval_eq_of_mot(t0, y_tmp, self.params, raw_loads=True)

            curr_u = (y_tmp + 0.5 * k1)[0]
            curr_v = (y_tmp + 0.5 * k1)[1]
            curr_w = (y_tmp + 0.5 * k1)[2]
            curr_p = (y_tmp + 0.5 * k1)[3]
            curr_q = (y_tmp + 0.5 * k1)[4]
            curr_r = (y_tmp + 0.5 * k1)[5]
            curr_x = (y_tmp + 0.5 * k1)[9]
            curr_y = (y_tmp + 0.5 * k1)[10]
            curr_z = (y_tmp + 0.5 * k1)[11]
            curr_phi = (y_tmp + 0.5 * k1)[6]
            curr_tht = (y_tmp + 0.5 * k1)[7]
            curr_psi = (y_tmp + 0.5 * k1)[8]

            err_x = curr_x - xd
            derr_x_dt = curr_u
            int_err_x = int_err_x + (self.params['dt'] / self.params['tf']) * err_x
            theta_des = -pid_force1(t0 + 0.5 * self.params['dt'], err_x, derr_x_dt, int_err_x, self.params['kp_x'],
                                    self.params['kd_x'], self.params['ki_x'])

            err_y = curr_y - yd
            derr_y_dt = curr_v
            int_err_y = int_err_y + (self.params['dt'] / self.params['tf']) * err_y
            phi_des = -pid_force1(t0 + 0.5 * self.params['dt'], err_y, derr_y_dt, int_err_y, self.params['kp_y'],
                                  self.params['kd_y'], self.params['ki_y'])

            err_z = curr_z - zd
            derr_dt_z = curr_w
            int_err_z = int_err_z + (self.params['dt'] / self.params['tf']) * err_z
            Ft = (self.params['m'] * self.params['g']) - pid_force1(t0 + 0.5 * self.params['dt'], err_z, derr_dt_z,
                                                                    int_err_z, self.params['kp_h'], self.params['kd_h'],
                                                                    self.params['ki_h'])

            err_phi = curr_phi - phi_des
            derr_dt_phi = self.getEulerAngleRate(euler_param='phid', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_phi = int_err_phi + (self.params['dt'] / self.params['tf']) * err_phi
            L = -pid_force1(t0 + 0.5 * self.params['dt'], err_phi, derr_dt_phi, int_err_phi, self.params['kp_phi'],
                            self.params['kd_phi'], self.params['ki_phi'])

            err_tht = curr_tht - theta_des
            derr_dt_tht = self.getEulerAngleRate(euler_param='thtd', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_tht = int_err_tht + (self.params['dt'] / self.params['tf']) * err_tht
            M = -pid_force1(t0 + 0.5 * self.params['dt'], err_tht, derr_dt_tht, int_err_tht, self.params['kp_tht'],
                            self.params['kd_tht'], self.params['ki_tht'])

            err_psi = curr_psi - psi_des
            derr_dt_psi = self.getEulerAngleRate(euler_param='psid', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_psi = int_err_psi + (self.params['dt'] / self.params['tf']) * err_psi
            N = -pid_force1(t0 + 0.5 * self.params['dt'], err_psi, derr_dt_psi, int_err_psi, self.params['kp_psi'],
                            self.params['kd_psi'], self.params['ki_psi'])

            self.params['Ft'], self.params['L'], self.params['M'], self.params['N'] = Ft, L, M, N
            k2 = self.params['dt'] * self.eval_eq_of_mot(t0 + 0.5 * self.params['dt'], y_tmp + 0.5 * k1, self.params,
                                                           raw_loads=True)

            curr_u = (y_tmp + 0.5 * k2)[0]
            curr_v = (y_tmp + 0.5 * k2)[1]
            curr_w = (y_tmp + 0.5 * k2)[2]
            curr_p = (y_tmp + 0.5 * k2)[3]
            curr_q = (y_tmp + 0.5 * k2)[4]
            curr_r = (y_tmp + 0.5 * k2)[5]
            curr_x = (y_tmp + 0.5 * k2)[9]
            curr_y = (y_tmp + 0.5 * k2)[10]
            curr_z = (y_tmp + 0.5 * k2)[11]
            curr_phi = (y_tmp + 0.5 * k2)[6]
            curr_tht = (y_tmp + 0.5 * k2)[7]
            curr_psi = (y_tmp + 0.5 * k2)[8]

            err_x = curr_x - xd
            derr_x_dt = curr_u
            int_err_x = int_err_x + (self.params['dt'] / self.params['tf']) * err_x
            theta_des = -pid_force1(t0 + 0.5 * self.params['dt'], err_x, derr_x_dt, int_err_x, self.params['kp_x'],
                                    self.params['kd_x'], self.params['ki_x'])

            err_y = curr_y - yd
            derr_y_dt = curr_v
            int_err_y = int_err_y + (self.params['dt'] / self.params['tf']) * err_y
            phi_des = -pid_force1(t0 + 0.5 * self.params['dt'], err_y, derr_y_dt, int_err_y, self.params['kp_y'],
                                  self.params['kd_y'], self.params['ki_y'])

            err_z = curr_z - zd
            derr_dt_z = curr_w
            int_err_z = int_err_z + (self.params['dt'] / self.params['tf']) * err_z
            Ft = (self.params['m'] * self.params['g']) - pid_force1(t0 + 0.5 * self.params['dt'], err_z, derr_dt_z,
                                                                    int_err_z, self.params['kp_h'], self.params['kd_h'],
                                                                    self.params['ki_h'])

            err_phi = curr_phi - phi_des
            derr_dt_phi = self.getEulerAngleRate(euler_param='phid', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_phi = int_err_phi + (self.params['dt'] / self.params['tf']) * err_phi
            L = -pid_force1(t0 + 0.5 * self.params['dt'], err_phi, derr_dt_phi, int_err_phi, self.params['kp_phi'],
                            self.params['kd_phi'], self.params['ki_phi'])

            err_tht = curr_tht - theta_des
            derr_dt_tht = self.getEulerAngleRate(euler_param='thtd', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_tht = int_err_tht + (self.params['dt'] / self.params['tf']) * err_tht
            M = -pid_force1(t0 + 0.5 * self.params['dt'], err_tht, derr_dt_tht, int_err_tht, self.params['kp_tht'],
                            self.params['kd_tht'], self.params['ki_tht'])

            err_psi = curr_psi - psi_des
            derr_dt_psi = self.getEulerAngleRate(euler_param='psid', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_psi = int_err_psi + (self.params['dt'] / self.params['tf']) * err_psi
            N = -pid_force1(t0 + 0.5 * self.params['dt'], err_psi, derr_dt_psi, int_err_psi, self.params['kp_psi'],
                            self.params['kd_psi'], self.params['ki_psi'])

            self.params['Ft'], self.params['L'], self.params['M'], self.params['N'] = Ft, L, M, N
            k3 = self.params['dt'] * self.eval_eq_of_mot(t0 + 0.5 * self.params['dt'], y_tmp + 0.5 * k2, self.params,
                                                           raw_loads=True)

            curr_u = (y_tmp + k3)[0]
            curr_v = (y_tmp + k3)[1]
            curr_w = (y_tmp + k3)[2]
            curr_p = (y_tmp + k3)[3]
            curr_q = (y_tmp + k3)[4]
            curr_r = (y_tmp + k3)[5]
            curr_x = (y_tmp + k3)[9]
            curr_y = (y_tmp + k3)[10]
            curr_z = (y_tmp + k3)[11]
            curr_phi = (y_tmp + k3)[6]
            curr_tht = (y_tmp + k3)[7]
            curr_psi = (y_tmp + k3)[8]

            err_x = curr_x - xd
            derr_x_dt = curr_u
            int_err_x = int_err_x + (self.params['dt'] / self.params['tf']) * err_x
            theta_des = -pid_force1(t0 + 0.5 * self.params['dt'], err_x, derr_x_dt, int_err_x, self.params['kp_x'],
                                    self.params['kd_x'], self.params['ki_x'])

            err_y = curr_y - yd
            derr_y_dt = curr_v
            int_err_y = int_err_y + (self.params['dt'] / self.params['tf']) * err_y
            phi_des = -pid_force1(t0 + 0.5 * self.params['dt'], err_y, derr_y_dt, int_err_y, self.params['kp_y'],
                                  self.params['kd_y'], self.params['ki_y'])

            err_z = curr_z - zd
            derr_dt_z = curr_w
            int_err_z = int_err_z + (self.params['dt'] / self.params['tf']) * err_z
            Ft = (self.params['m'] * self.params['g']) - pid_force1(t0 + 0.5 * self.params['dt'], err_z, derr_dt_z,
                                                                    int_err_z, self.params['kp_h'], self.params['kd_h'],
                                                                    self.params['ki_h'])

            err_phi = curr_phi - phi_des
            derr_dt_phi = self.getEulerAngleRate(euler_param='phid', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_phi = int_err_phi + (self.params['dt'] / self.params['tf']) * err_phi
            L = -pid_force1(t0 + 0.5 * self.params['dt'], err_phi, derr_dt_phi, int_err_phi, self.params['kp_phi'],
                            self.params['kd_phi'], self.params['ki_phi'])

            err_tht = curr_tht - theta_des
            derr_dt_tht = self.getEulerAngleRate(euler_param='thtd', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_tht = int_err_tht + (self.params['dt'] / self.params['tf']) * err_tht
            M = -pid_force1(t0 + 0.5 * self.params['dt'], err_tht, derr_dt_tht, int_err_tht, self.params['kp_tht'],
                            self.params['kd_tht'], self.params['ki_tht'])

            err_psi = curr_psi - psi_des
            derr_dt_psi = self.getEulerAngleRate(euler_param='psid', p=curr_p, q=curr_q, r=curr_r, phi=curr_phi,
                                                 tht=curr_tht, psi=curr_psi)
            int_err_psi = int_err_psi + (self.params['dt'] / self.params['tf']) * err_psi
            N = -pid_force1(t0 + 0.5 * self.params['dt'], err_psi, derr_dt_psi, int_err_psi, self.params['kp_psi'],
                            self.params['kd_psi'], self.params['ki_psi'])

            self.params['Ft'], self.params['L'], self.params['M'], self.params['N'] = Ft, L, M, N
            k4 = self.params['dt'] * self.eval_eq_of_mot(t0 + 0.5 * self.params['dt'], y_tmp + k3, self.params,
                                                           raw_loads=True)

            y_new = y_tmp + (1. / 6.) * (k1 + 2 * k2 + 2 * k3 + k4)

            y_tmp = y_new.copy()

            ## restricting altitude to physical solutions
            # if y_tmp[11] < 0:
            #     break

            self.x[t_idx + 1, :] = y_tmp

            x_per = abs(xd - y_tmp[9]) / xd * 100
            y_per = abs(yd - y_tmp[10]) / yd * 100
            z_per = abs(zd - y_tmp[11]) / zd * 100

            # print('time = ', self.params['t'][t_idx], ' x_error = ', x_per, ' y_error = ', y_per, ' z_error = ', z_per)

            if (x_per < conv_tol) and (y_per < conv_tol) and (z_per < conv_tol):
                print('converged on position at time = ', self.t[t_idx])
                for tt_idx in range(len(self.t[t_idx:]) - 1):
                    self.x[t_idx + tt_idx + 1, :] = y_tmp
                    # print(self.params['t'][t_idx+tt_idx])
                break

    def plotter_with_time(self, yvar: str = 'ze'):

        self.plot_no = self.plot_no + 1

        if yvar == 'phi' or yvar == 'tht' or yvar == 'psi':
            plot(self.t, self.x[:, self.plot_dict[yvar][0]] * 180 / pi)
        elif yvar == 'rpm':
            plot(self.t, self.inp[:, 0], color='red')
            plot(self.t, self.inp[:, 1], color='blue')
            plot(self.t, self.inp[:, 2], color='green')
            plot(self.t, self.inp[:, 3], color='black')
            axhline(y=self.get_hover_rpm(), color='red')
        else:
            plot(self.t, self.x[:, self.plot_dict[yvar][0]])
        grid('on')
        xlabel('time(seconds)')
        ylabel(self.plot_dict[yvar][1])

    def animate(self):
        paramss = Munch({
            'dx': self.params["dx"],
            'dy': self.params["dy"],
        })
        quad_vedo_model = QuadVedoModel(params=paramss)
        quad_vedo_model.animate_simulation(self)


if __name__ == "__main__":
    params = {

        # mass parameters
        'm': 0.468,  # takeoff mass in kg
        'Ixx': 4.856e-3,  # Moment of inertia kgm^2
        'Iyy': 4.856e-3,  # Moment of inertia
        'Izz': 8.801e-3,  # Moment of inertia (Assume nearly flat object, z=0)

        'g': 9.81,  # acceleration due to gravity

        # geometry parameters
        'd1x': 0.225,  # x offset of thrust axis
        'd1y': 0.225,  # y offset of thrust axis
        'd2x': -0.225,  # x offset of thrust axis
        'd2y': -0.225,  # y offset of thrust axis
        'd3x': -0.225,  # x offset of thrust axis
        'd3y': 0.225,  # y offset of thrust axis
        'd4x': 0.225,  # x offset of thrust axis
        'd4y': -0.225,  # y offset of thrust axis

        'dx': 0.225,
        'dy': 0.225,

        # loads parameters
        'km': 2.98e-6,  # thrust coeff of motor propeller
        'bm': 0.114e-6,  # yawing moment coeff of motor propeller

        # motor rpms
        'w1': 5926.396232119796 + 10,
        'w2': 5926.396232119796 + 10,
        'w3': 5926.396232119796 + 10,
        'w4': 5926.396232119796 + 10,

        # simulation parameters
        'dt': 0.01,  # Sampling time (sec)
        'tf': 10,  # Length of time to run simulation (sec),
        'X0': array([0,  # u0
                     0,  # v0
                     0.,  # w0
                     0,  # p0
                     0,  # q0
                     0,  # r0
                     radians(0.),  # phi0
                     radians(0.),  # tht0
                     radians(0.),  # psi0
                     0.,  # x0
                     0.,  # y0
                     0.]),  # z

        # control parameters
        'kp_h': 20,  # proportional constant for altitude control
        'kd_h': 5,  # derivative constant for altitude control
        'ki_h': 100,

        'kp_tht': 1,
        'kd_tht': 0.12,
        'ki_tht': 2.,

        'kp_phi': 3,
        'kd_phi': 0.12,
        'ki_phi': 0.,

        'kp_psi': 1,
        'kd_psi': 0.12,
        'ki_psi': 0.,

        'kp_x': 0.1,
        'kd_x': 0.2,
        'ki_x': 0.,

        'kp_y': -0.25,
        'kd_y': -0.2,
        'ki_y': 0.,

        'kp_p': 0.25,
        'kd_p': 0.001,
        'ki_p': 0.3,

        'kp_q': 0.25,
        'kd_q': 0.001,
        'ki_q': 0.3,

        'kp_r': 0.25,
        'kd_r': 0.001,
        'ki_r': 0.3

    }

    drone1 = DroneSim(params=params)
    print(drone1.get_hover_rpm())
    #drone1.time_simulate()
    drone1.move_to_xyz(3, 0, 2, 0)

    # #vars_to_plot = ['phi', 'tht', 'psi', 'xe', 'ye', 'ze', 'u', 'v', 'w', 'p', 'q', 'r']
    # vars_to_plot = ['phi', 'tht', 'psi', 'xe', 'ye', 'ze']
    # for var in vars_to_plot:
    #     figure(drone1.plot_no)
    #     drone1.plotter_with_time(yvar=var)
    #     if var == 'ze':
    #         gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.6f}'))
    #
    # show()

    drone1.animate()
