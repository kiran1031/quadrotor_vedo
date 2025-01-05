"""
Module contains classes which builds rigid body  model of a quadrotor
"""

# ************************************************************************* #
#                          Import statements                                #
# ************************************************************************* #
from sympy import symbols, Matrix, sin, cos, solve, lambdify
from sympy.physics.mechanics import ReferenceFrame, dynamicsymbols, Point, inertia, dot, cross


class RigidBodyModel(object):

    def __init__(self, lin_vel_in_inertial:bool=True):

        # create the reference frames
        self.I = ReferenceFrame('I')  # inertial frame
        self.B = ReferenceFrame('B')  # Body frame

        # create motor axis position parameters
        self.d1x, self.d1y = symbols('d_{1x}, d_{1y}', real=True)  # coordinates of thrust axis 1 in body frame
        self.d2x, self.d2y = symbols('d_{2x}, d_{2y}', real=True)  # coordinates of thrust axis 2 in body frame
        self.d3x, self.d3y = symbols('d_{3x}, d_{3y}', real=True)  # coordinates of thrust axis 3 in body frame
        self.d4x, self.d4y = symbols('d_{4x}, d_{4y}', real=True)  # coordinates of thrust axis 4 in body frame

        # create motor axis position vectors
        self.rm1C_B = self.d1x * self.B.x + self.d1y * self.B.y
        self.rm2C_B = self.d2x * self.B.x + self.d2y * self.B.y
        self.rm3C_B = self.d3x * self.B.x + self.d3y * self.B.y
        self.rm4C_B = self.d4x * self.B.x + self.d4y * self.B.y

        # create state vector variables
        """
        ub        --> x velocity of CG in body frame
        vb        --> y velocity of CG in body frame
        wb        --> z velocity of CG in body frame
        ue        --> x velocity of CG in inertial frame
        ve        --> y velocity of CG in inertial frame
        we        --> z velocity of CG in inertial frame
        p         --> roll rate in body frame
        q         --> pitch rate in body frame
        r         --> yaw rate in body frame
        phi       --> roll angle in euler zyx rotation
        theta     --> pitch angle in euler zyx rotation
        psi       --> yaw angle in euler zyx rotation
        """
        self.phi, self.tht, self.psi = dynamicsymbols('phi, theta, psi')
        self.ub, self.vb, self.wb = dynamicsymbols('u_b, v_b, w_b')
        self.ue, self.ve, self.we = dynamicsymbols('u_e, v_e, w_e')
        self.p, self.q, self.r = dynamicsymbols('p, q, r')

        # CG of the quadrotor
        self.CG = Point('CG')

        # orient body frame wrto inertial frame using ZYX euler angles
        self.B.orient_body_fixed(self.I, (self.psi, self.tht, self.phi), 'ZYX')
        self.R_B_I = self.I.dcm(self.B)
        self.R_B_I_func = lambdify([self.phi, self.tht, self.psi], self.R_B_I)

        # compute linear velocity of CG in body or inertial frame
        if lin_vel_in_inertial:
            self.CG.set_vel(self.I, self.ue * self.I.x + self.ve * self.I.y + self.we * self.I.z)
            self.I_vCG_I = self.CG.vel(self.I)
        else:
            self.CG.set_vel(self.I, self.ub * self.B.x + self.vb * self.B.y + self.wb * self.B.z)
            self.I_vCG_B = self.CG.vel(self.I)

        # compute angular velocity of frame B wrto frame I
        self.B.set_ang_vel(self.I, self.p * self.B.x + self.q * self.B.y + self.r * self.B.z)
        self.I_w_BI_B = self.B.ang_vel_in(self.I)

        # initialize rpm parameters
        self.w1, self.w2, self.w3, self.w4 = dynamicsymbols('omega_1, omega_2, omega_3, omega_4')

        # mass parameters (mtow is take of mass)
        self.mtow = symbols('m')

        # inertia tensor in body frame
        self.Ixx, self.Iyy, self.Izz = symbols('I_{xx}, I_{yy}, I_{zz}')
        self.ICG_B = inertia(self.B, self.Ixx, self.Iyy, self.Izz)

        # applied forces and moments
        """
        This class defines various loads applied, acting on the UAM
        Fx, Fy, Fz      --> External applied loads
        L, M, N         --> External applied moments
        """
        self.Fx, self.Fy, self.Fz = dynamicsymbols('F_x, F_y, F_z')
        self.Ft = dynamicsymbols('F_t')  # thrust force

        if lin_vel_in_inertial:
            self.FCG_I = self.Fx * self.I.x + self.Fy * self.I.y + self.Fz * self.I.z
        else:
            self.FCG_B = self.Fx * self.B.x + self.Fy * self.B.y + self.Fz * self.B.z

        self.L, self.M, self.N = dynamicsymbols('L, M, N')
        self.MCG_B = self.L * self.B.x + self.M * self.B.y + self.N * self.B.z

        # inertial force
        self.Fi = -self.mtow * self.CG.acc(self.I)

        # inertial moment
        self.Mi = -(dot(self.B.ang_acc_in(self.I), self.ICG_B) + dot(
            cross(self.B.ang_vel_in(self.I), self.ICG_B), self.B.ang_vel_in(self.I)))

        self.g = symbols('g', real=True)  # gravity constant

        # motor thrust and yaw constants
        self.km = symbols('k_m', real=True)
        self.bm = symbols('b_m', real=True)

        """
    
        INPUTS:
        ------
        (
            1. Sum of thrusts
            2. rolling moment
            3. pitching moment
            4. yawing moment
    
        )
    
        OUTPUTS:
        -------
            1. Inertial x velocity in Body Frame
            2. Inertial y velocity in Body Frame
            3. Inertial z velocity in Body Frame
            4. Inertial roll rate in Body Frame
            5. Inertial picth rate in Body Frame
            6. Inertial yaw rate in Body Frame
            7. roll angle in euler ZYX convention
            8. pitch angle in euler ZYX convention
            9. yaw angle in euler ZYX convention
    
        METHOD:
        -------
            It derives the equations of motion using the Kane's dynamics
    
        """

        # kinematic equations of motion, phid, thtd, psid into p, q, r
        mat = Matrix([[1, 0, -sin(self.tht)],
                      [0, cos(self.phi), sin(self.phi) * cos(self.tht)],
                      [0, -sin(self.phi), cos(self.phi) * cos(self.tht)]])
        mat1 = mat.inv()
        self.kin_eq_of_mot = mat1 * Matrix([self.p, self.q, self.r])

        # generalized linear velocities
        v_C_1 = self.I_vCG_I.diff(self.ue, self.I, var_in_dcm=False)
        v_C_2 = self.I_vCG_I.diff(self.ve, self.I, var_in_dcm=False)
        v_C_3 = self.I_vCG_I.diff(self.we, self.I, var_in_dcm=False)
        v_C_4 = self.I_vCG_I.diff(self.p, self.I, var_in_dcm=False)
        v_C_5 = self.I_vCG_I.diff(self.q, self.I, var_in_dcm=False)
        v_C_6 = self.I_vCG_I.diff(self.r, self.I, var_in_dcm=False)

        # generalized angular velocites
        w_B_1 = self.I_w_BI_B.diff(self.ue, self.I, var_in_dcm=False)
        w_B_2 = self.I_w_BI_B.diff(self.ve, self.I, var_in_dcm=False)
        w_B_3 = self.I_w_BI_B.diff(self.we, self.I, var_in_dcm=False)
        w_B_4 = self.I_w_BI_B.diff(self.p, self.I, var_in_dcm=False)
        w_B_5 = self.I_w_BI_B.diff(self.q, self.I, var_in_dcm=False)
        w_B_6 = self.I_w_BI_B.diff(self.r, self.I, var_in_dcm=False)

        F1r = v_C_1.dot(self.FCG_I) + w_B_1.dot(self.MCG_B)
        F2r = v_C_2.dot(self.FCG_I) + w_B_2.dot(self.MCG_B)
        F3r = v_C_3.dot(self.FCG_I) + w_B_3.dot(self.MCG_B)
        F4r = v_C_4.dot(self.FCG_I) + w_B_4.dot(self.MCG_B)
        F5r = v_C_5.dot(self.FCG_I) + w_B_5.dot(self.MCG_B)
        F6r = v_C_6.dot(self.FCG_I) + w_B_6.dot(self.MCG_B)

        # generalized applied forces
        Fr = Matrix([F1r, F2r, F3r, F4r, F5r, F6r])

        # generalized inertia forces
        F1s = v_C_1.dot(self.Fi) + w_B_1.dot(self.Mi)
        F2s = v_C_2.dot(self.Fi) + w_B_2.dot(self.Mi)
        F3s = v_C_3.dot(self.Fi) + w_B_3.dot(self.Mi)
        F4s = v_C_4.dot(self.Fi) + w_B_4.dot(self.Mi)
        F5s = v_C_5.dot(self.Fi) + w_B_5.dot(self.Mi)
        F6s = v_C_6.dot(self.Fi) + w_B_6.dot(self.Mi)

        Frs = Matrix([F1s, F2s, F3s, F4s, F5s, F6s])

        self.dyn_eq_of_mot_raw = Fr + Frs

        Fg_I = -self.mtow * self.g * self.I.z

        Fg_I = Fg_I.to_matrix(self.I)
        Ft_I = (self.Ft * self.B.z).to_matrix(self.I)
        F_app = Fg_I + Ft_I
        self.dyn_eq_of_mot_raw = self.dyn_eq_of_mot_raw.subs([
            (self.Fx, F_app[0]),
            (self.Fy, F_app[1]),
            (self.Fz, F_app[2])
        ])
        self.dyn_eq_of_mot = solve(self.dyn_eq_of_mot_raw, (self.ue.diff(),
                                                            self.ve.diff(),
                                                            self.we.diff(),
                                                            self.p.diff(),
                                                            self.q.diff(),
                                                            self.r.diff()))

        self.pos_eq_of_mot = Matrix([self.ue, self.ve, self.we])

        self.statevector = [
            self.ue,
            self.ve,
            self.we,
            self.p,
            self.q,
            self.r,
            self.phi,
            self.tht,
            self.psi
        ]

        self.eq_of_mot = [
            self.dyn_eq_of_mot[self.ue.diff()],
            self.dyn_eq_of_mot[self.ve.diff()],
            self.dyn_eq_of_mot[self.we.diff()],
            self.dyn_eq_of_mot[self.p.diff()],
            self.dyn_eq_of_mot[self.q.diff()],
            self.dyn_eq_of_mot[self.r.diff()],
            self.kin_eq_of_mot[0],
            self.kin_eq_of_mot[1],
            self.kin_eq_of_mot[2],
            self.pos_eq_of_mot[0],
            self.pos_eq_of_mot[1],
            self.pos_eq_of_mot[2]
        ]

        self.paramslst = [self.ue,
                          self.ve,
                          self.we,
                          self.p,
                          self.q,
                          self.r,
                          self.phi,
                          self.tht,
                          self.psi,
                          self.mtow,
                          self.Ixx,
                          self.Iyy,
                          self.Izz,
                          self.g,
                          self.Ft,
                          self.L,
                          self.M,
                          self.N]

        self.eq_of_mot_func = []
        for eq in self.eq_of_mot:
            self.eq_of_mot_func.append(lambdify(self.paramslst, eq))

        print('*******************Model created successfully*****************')

if __name__ == "__main__":
    rbd_model = RigidBodyModel()
