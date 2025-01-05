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
        'w2': 5926.396232119796 - 10,
        'w3': 5926.396232119796 + 10,
        'w4': 5926.396232119796 - 10,

        # simulation parameters
        'dt': 0.01,  # Sampling time (sec)
        'tf': 2,  # Length of time to run simulation (sec),
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
                     6.]),  # z

    }

    drone1 = DroneSim(params=params)
    print(drone1.get_hover_rpm())
    drone1.time_simulate()

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
