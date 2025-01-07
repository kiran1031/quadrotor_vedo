"""

This module creates 3d model of a quadrotor using vedo.
It is a parameterized model,

"""

# ************************************************************************* #
#                          Import statements                                #
# ************************************************************************* #
from vedo import Box, Cylinder, Plotter, LinearTransform, Box, vector
from munch import Munch
from numpy import sqrt, arctan2, sin, cos, pi, arange, isclose


camera1 = dict(
    pos=(0, 20, 0),
    focal_point=(0, 0, 0),
    viewup=(0, 0, 1)
)

camera2 = dict(
    pos=(0, 0, 5),
    focal_point=(0, 0, 0),
    viewup=(0, 1, 0)
)


# ************************************************************************* #
#                     Class for creating vedo model                         #
# ************************************************************************* #
class QuadVedoModel(object):

    def __init__(self, params=None):
        self.arm_len = None
        self.arm_ang = None
        self.prop1 = None
        self.prop2 = None
        self.prop3 = None
        self.prop4 = None
        self.quad_frame = None
        self.prop1_rot_pt = None
        self.prop2_rot_pt = None
        self.prop3_rot_pt = None
        self.prop4_rot_pt = None
        self.plt = None
        self.params = params
        self.create_additional_parameters()
        self.create_model()

    def create_additional_parameters(self):
        self.arm_len = 2*sqrt(self.params.dx**2 + self.params.dy**2)
        self.arm_ang = arctan2(self.params.dx, self.params.dy)

    def create_model(self):
        world = Box(pos=(0, 0, 0), size=(20, 20, 20)).wireframe()
        arm1 = Cylinder(
            pos=[(0.5 * self.arm_len * cos(self.arm_ang), 0.5 * self.arm_len * sin(self.arm_ang), 0.0),
                 (-0.5 * self.arm_len * cos(self.arm_ang), -0.5 * self.arm_len * sin(self.arm_ang), 0.0)],
            r=0.01*self.arm_len
        )
        arm2 = Cylinder(
            pos=[(-0.5 * self.arm_len * sin(self.arm_ang), 0.5 * self.arm_len * cos(self.arm_ang), 0.0),
                 (0.5 * self.arm_len * sin(self.arm_ang), -0.5 * self.arm_len * cos(self.arm_ang), 0.0)],
            r=0.01 * self.arm_len
        )
        motor1 = Cylinder(
            pos=[(0.5 * self.arm_len * cos(self.arm_ang), 0.5 * self.arm_len * sin(self.arm_ang), 0.0),
                 (0.5 * self.arm_len * cos(self.arm_ang), 0.5 * self.arm_len * sin(self.arm_ang), 0.1 * self.arm_len)],
            r=0.05 * self.arm_len,
            c="red"
        )
        motor2 = Cylinder(
            pos=[(-0.5 * self.arm_len * cos(self.arm_ang), -0.5 * self.arm_len * sin(self.arm_ang), 0.0),
                 (-0.5 * self.arm_len * cos(self.arm_ang), -0.5 * self.arm_len * sin(self.arm_ang), 0.1 * self.arm_len)],
            r=0.05 * self.arm_len,
            c="blue"
        )
        motor3 = Cylinder(
            pos=[(-0.5 * self.arm_len * sin(self.arm_ang), 0.5 * self.arm_len * cos(self.arm_ang), 0.0),
                 (-0.5 * self.arm_len * sin(self.arm_ang), 0.5 * self.arm_len * cos(self.arm_ang), 0.1 * self.arm_len)],
            r=0.05 * self.arm_len,
            c="green"
        )
        motor4 = Cylinder(
            pos=[(0.5 * self.arm_len * sin(self.arm_ang), -0.5 * self.arm_len * cos(self.arm_ang), 0.0),
                 (0.5 * self.arm_len * sin(self.arm_ang), -0.5 * self.arm_len * cos(self.arm_ang), 0.1 * self.arm_len)],
            r=0.05 * self.arm_len,
            c="yellow"
        )
        self.prop1 = Cylinder(
            pos=[(0.5 * self.arm_len * cos(self.arm_ang) - 0.2 * self.arm_len, 0.5 * self.arm_len * sin(self.arm_ang), 0.11 * self.arm_len),
                 (0.5 * self.arm_len * cos(self.arm_ang) + 0.2 * self.arm_len, 0.5 * self.arm_len * sin(self.arm_ang), 0.11 * self.arm_len)],
            r=0.01 * self.arm_len
        )
        self.prop2 = Cylinder(
            pos=[(-0.5 * self.arm_len * cos(self.arm_ang) - 0.2 * self.arm_len, -0.5 * self.arm_len * sin(self.arm_ang), 0.11 * self.arm_len),
                 (-0.5 * self.arm_len * cos(self.arm_ang) + 0.2 * self.arm_len, -0.5 * self.arm_len * sin(self.arm_ang), 0.11 * self.arm_len)],
            r=0.01 * self.arm_len
        )
        self.prop3 = Cylinder(
            pos=[(-0.5 * self.arm_len * sin(self.arm_ang) - 0.2 * self.arm_len, 0.5 * self.arm_len * cos(self.arm_ang), 0.11 * self.arm_len),
                 (-0.5 * self.arm_len * sin(self.arm_ang) + 0.2 * self.arm_len, 0.5 * self.arm_len * cos(self.arm_ang), 0.11 * self.arm_len)],
            r=0.01 * self.arm_len
        )
        self.prop4 = Cylinder(
            pos=[(0.5 * self.arm_len * sin(self.arm_ang) - 0.2 * self.arm_len, -0.5 * self.arm_len * cos(self.arm_ang), 0.11 * self.arm_len),
                 (0.5 * self.arm_len * sin(self.arm_ang) + 0.2 * self.arm_len, -0.5 * self.arm_len * cos(self.arm_ang), 0.11 * self.arm_len)],
            r=0.01 * self.arm_len
        )
        # self.quad_frame = arm1 + arm2 + motor1 + motor2 + motor3 + motor4 + self.prop1 + self.prop2 + self.prop3 + \
        #                   self.prop4
        self.quad_frame = arm1 + arm2 + motor1 + motor2 + motor3 + motor4

        self.plt = Plotter(title="Quad Model", axes=1, interactive=False)
        self.plt += world
        self.plt += self.quad_frame
        self.plt += self.prop1
        self.plt += self.prop2
        self.plt += self.prop3
        self.plt += self.prop4

        self.plt.show(camera=camera1)

    def test_translation(self, w1=42000, w2=42000, w3=-42000, w4=-42000, x_vel=1.0, y_vel=1.0, z_vel=1.0):
        """

        :param w1: prop 1 rotation speed in deg/s
        :param w2:
        :param w3:
        :param w4:
        :param x_vel: translation velocity in x axis
        :param y_vel:
        :param z_vel:
        :return:
        """
        xpos, ypos, zpos = 0.0, 0.0, 0.0
        ang1, ang2, ang3, ang4 = 0.0, 0.0, 0.0, 0.0
        dt = 0.01

        Ltrans = LinearTransform()
        LT1 = LinearTransform()
        LT2 = LinearTransform()
        LT3 = LinearTransform()
        LT4 = LinearTransform()
        prev_ang1, prev_ang2, prev_ang3, prev_ang4 = 0.0, 0.0, 0.0, 0.0
        prop1_pos_initial = self.prop1.pos().copy()
        prop2_pos_initial = self.prop2.pos().copy()
        prop3_pos_initial = self.prop3.pos().copy()
        prop4_pos_initial = self.prop4.pos().copy()
        for i in range(1000):
            ang1 = ang1 + w1 * dt
            ang2 = ang2 + w2 * dt
            ang3 = ang3 + w3 * dt
            ang4 = ang4 + w4 * dt

            Ltrans.translate([xpos, ypos, zpos])
            Ltrans.move(self.quad_frame)

            new_prop1_pos = prop1_pos_initial + vector(xpos, ypos, zpos)
            new_prop2_pos = prop2_pos_initial + vector(xpos, ypos, zpos)
            new_prop3_pos = prop3_pos_initial + vector(xpos, ypos, zpos)
            new_prop4_pos = prop4_pos_initial + vector(xpos, ypos, zpos)

            self.prop1.pos(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2])
            self.prop2.pos(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2])
            self.prop3.pos(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2])
            self.prop4.pos(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2])

            LT1.rotate(ang1 - prev_ang1, axis=(0, 0, 1), point=(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2]))
            self.prop1.apply_transform(LT1)

            LT2.rotate(ang2 - prev_ang2, axis=(0, 0, 1), point=(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2]))
            self.prop2.apply_transform(LT2)

            LT3.rotate(ang3 - prev_ang3, axis=(0, 0, 1), point=(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2]))
            self.prop3.apply_transform(LT3)

            LT4.rotate(ang4 - prev_ang4, axis=(0, 0, 1), point=(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2]))
            self.prop4.apply_transform(LT4)

            self.plt.render()

            xpos = xpos + x_vel * dt
            ypos = ypos + y_vel * dt
            zpos = zpos + z_vel * dt

            Ltrans.reset()
            LT1.reset()
            LT2.reset()
            LT3.reset()
            LT4.reset()
            self.quad_frame.transform.reset()

            prev_ang1 = ang1
            prev_ang2 = ang2
            prev_ang3 = ang3
            prev_ang4 = ang4

        self.plt.interactive().close()

    def test_rotation(self, w1=42000, w2=42000, w3=-42000, w4=-42000, rate=10, mode="yaw"):
        """

        :param w1: prop 1 rotation speed in deg/s
        :param w2:
        :param w3:
        :param w4:
        :param rate: rate of rotation
        :param mode: pitch, roll or  yaw
        :return:
        """
        ang1, ang2, ang3, ang4 = 0.0, 0.0, 0.0, 0.0
        angle = 0.0
        dt = 0.01

        LTrot = LinearTransform()
        LT1 = LinearTransform()
        LT2 = LinearTransform()
        LT3 = LinearTransform()
        LT4 = LinearTransform()
        prev_ang1, prev_ang2, prev_ang3, prev_ang4 = 0.0, 0.0, 0.0, 0.0

        for i in range(500):
            ang1 = ang1 + w1 * dt
            ang2 = ang2 + w2 * dt
            ang3 = ang3 + w3 * dt
            ang4 = ang4 + w4 * dt

            axis = (0, 0, 1)
            if mode == "pitch":
                axis = (0, 1, 0)
            elif mode == "roll":
                axis = (1, 0, 0)
            elif mode == "yaw":
                axis = (0, 0, 1)
            LTrot.rotate(angle, axis=axis, point=(0, 0, 0), rad=False)
            LTrot.move(self.quad_frame)

            new_prop1_pos = (vector(LTrot.matrix3x3 @ self.prop1.base) + vector(LTrot.matrix3x3 @ self.prop1.top)) / 2.0
            new_prop2_pos = (vector(LTrot.matrix3x3 @ self.prop2.base) + vector(LTrot.matrix3x3 @ self.prop2.top)) / 2.0
            new_prop3_pos = (vector(LTrot.matrix3x3 @ self.prop3.base) + vector(LTrot.matrix3x3 @ self.prop3.top)) / 2.0
            new_prop4_pos = (vector(LTrot.matrix3x3 @ self.prop4.base) + vector(LTrot.matrix3x3 @ self.prop4.top)) / 2.0

            new_prop_rot_axis = LTrot.matrix3x3 @ vector(0, 0, 1)

            self.prop1.pos(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2])
            self.prop2.pos(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2])
            self.prop3.pos(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2])
            self.prop4.pos(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2])

            LT1.rotate(ang1 - prev_ang1, axis=new_prop_rot_axis, point=(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2]))
            self.prop1.apply_transform(LT1)

            LT2.rotate(ang2 - prev_ang2, axis=new_prop_rot_axis, point=(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2]))
            self.prop2.apply_transform(LT2)

            LT3.rotate(ang3 - prev_ang3, axis=new_prop_rot_axis, point=(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2]))
            self.prop3.apply_transform(LT3)

            LT4.rotate(ang4 - prev_ang4, axis=new_prop_rot_axis, point=(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2]))
            self.prop4.apply_transform(LT4)

            self.plt.render()

            angle = angle + (rate * dt)

            LTrot.reset()
            LT1.reset()
            LT2.reset()
            LT3.reset()
            LT4.reset()
            self.quad_frame.transform.reset()

            prev_ang1 = ang1
            prev_ang2 = ang2
            prev_ang3 = ang3
            prev_ang4 = ang4

        self.plt.interactive().close()

    def test_general_motion(self, w1=30, w2=30, w3=-30, w4=-30,
                            pitch_rate=0.0, roll_rate=0.0, yaw_rate=0.0, x_vel=0.0, y_vel=0.0, z_vel=0.0):
        """

        :param w1:
        :param w2:
        :param w3:
        :param w4:
        :param pitch_rate:
        :param roll_rate:
        :param yaw_rate:
        :param x_vel:
        :param y_vel:
        :param z_vel:
        :param mode:
        :return:
        """
        xpos, ypos, zpos = 0.0, 0.0, 0.0
        ang1, ang2, ang3, ang4 = 0.0, 0.0, 0.0, 0.0
        pitch = 0.0
        roll = 0.0
        yaw = 0.0
        dt = 0.01

        LTrot = LinearTransform()
        Ltrans = LinearTransform()
        LT1 = LinearTransform()
        LT2 = LinearTransform()
        LT3 = LinearTransform()
        LT4 = LinearTransform()
        prev_ang1, prev_ang2, prev_ang3, prev_ang4 = 0.0, 0.0, 0.0, 0.0

        for i in range(1000):
            ang1 = ang1 + w1 * dt
            ang2 = ang2 + w2 * dt
            ang3 = ang3 + w3 * dt
            ang4 = ang4 + w4 * dt

            Ltrans.translate([xpos, ypos, zpos])
            Ltrans.move(self.quad_frame)
            LTrot.rotate(yaw, axis=(0, 0, 1), point=(xpos, ypos, zpos), rad=False)
            LTrot.rotate(pitch, axis=(0, 1, 0), point=(xpos, ypos, zpos), rad=False)
            LTrot.rotate(roll, axis=(1, 0, 0), point=(xpos, ypos, zpos), rad=False)
            LTrot.move(self.quad_frame)

            new_prop1_pos = vector(xpos, ypos, zpos) + (vector(LTrot.matrix3x3 @ self.prop1.base) + vector(LTrot.matrix3x3 @ self.prop1.top)) / 2.0
            new_prop2_pos = vector(xpos, ypos, zpos) + (vector(LTrot.matrix3x3 @ self.prop2.base) + vector(LTrot.matrix3x3 @ self.prop2.top)) / 2.0
            new_prop3_pos = vector(xpos, ypos, zpos) + (vector(LTrot.matrix3x3 @ self.prop3.base) + vector(LTrot.matrix3x3 @ self.prop3.top)) / 2.0
            new_prop4_pos = vector(xpos, ypos, zpos) + (vector(LTrot.matrix3x3 @ self.prop4.base) + vector(LTrot.matrix3x3 @ self.prop4.top)) / 2.0

            new_prop_rot_axis = LTrot.matrix3x3 @ vector(0, 0, 1)

            self.prop1.pos(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2])
            self.prop2.pos(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2])
            self.prop3.pos(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2])
            self.prop4.pos(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2])

            LT1.rotate(ang1 - prev_ang1, axis=new_prop_rot_axis,
                       point=(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2]))
            self.prop1.apply_transform(LT1)

            LT2.rotate(ang2 - prev_ang2, axis=new_prop_rot_axis,
                       point=(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2]))
            self.prop2.apply_transform(LT2)

            LT3.rotate(ang3 - prev_ang3, axis=new_prop_rot_axis,
                       point=(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2]))
            self.prop3.apply_transform(LT3)

            LT4.rotate(ang4 - prev_ang4, axis=new_prop_rot_axis,
                       point=(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2]))
            self.prop4.apply_transform(LT4)

            self.plt.render()

            xpos = xpos + x_vel * dt
            ypos = ypos + y_vel * dt
            zpos = zpos + z_vel * dt
            yaw = yaw + (yaw_rate * dt)
            pitch = pitch + (pitch_rate * dt)
            roll = roll + (roll_rate * dt)

            Ltrans.reset()
            LTrot.reset()
            LT1.reset()
            LT2.reset()
            LT3.reset()
            LT4.reset()
            self.quad_frame.transform.reset()

            prev_ang1 = ang1
            prev_ang2 = ang2
            prev_ang3 = ang3
            prev_ang4 = ang4

        self.plt.interactive().close()

    def animate_simulation(self, drone_object=None):
        """
        :param drone_object:
        :return:
        """
        ang1, ang2, ang3, ang4 = 0.0, 0.0, 0.0, 0.0
        dt = drone_object.params["dt"]

        LTrot = LinearTransform()
        Ltrans = LinearTransform()
        LT1 = LinearTransform()
        LT2 = LinearTransform()
        LT3 = LinearTransform()
        LT4 = LinearTransform()
        prev_ang1, prev_ang2, prev_ang3, prev_ang4 = 0.0, 0.0, 0.0, 0.0

        for i in range(len(drone_object.t)):
            ang1 = ang1 + 6 * drone_object.params["w1"] * dt
            ang2 = ang2 + 6 * drone_object.params["w2"] * dt
            ang3 = ang3 + 6 * drone_object.params["w3"] * dt
            ang4 = ang4 + 6 * drone_object.params["w4"] * dt

            xpos = drone_object.x[i, drone_object.plot_dict["xe"][0]]
            ypos = drone_object.x[i, drone_object.plot_dict["ye"][0]]
            zpos = drone_object.x[i, drone_object.plot_dict["ze"][0]]

            yaw = drone_object.x[i, drone_object.plot_dict["psi"][0]] * 180 / pi
            pitch = drone_object.x[i, drone_object.plot_dict["tht"][0]] * 180 / pi
            roll = drone_object.x[i, drone_object.plot_dict["phi"][0]] * 180 / pi

            Ltrans.translate([xpos, ypos, zpos])
            Ltrans.move(self.quad_frame)
            LTrot.rotate(yaw, axis=(0, 0, 1), point=(xpos, ypos, zpos), rad=False)
            LTrot.rotate(pitch, axis=(0, 1, 0), point=(xpos, ypos, zpos), rad=False)
            LTrot.rotate(roll, axis=(1, 0, 0), point=(xpos, ypos, zpos), rad=False)
            LTrot.move(self.quad_frame)

            new_prop1_pos = vector(xpos, ypos, zpos) + (
                        vector(LTrot.matrix3x3 @ self.prop1.base) + vector(LTrot.matrix3x3 @ self.prop1.top)) / 2.0
            new_prop2_pos = vector(xpos, ypos, zpos) + (
                        vector(LTrot.matrix3x3 @ self.prop2.base) + vector(LTrot.matrix3x3 @ self.prop2.top)) / 2.0
            new_prop3_pos = vector(xpos, ypos, zpos) + (
                        vector(LTrot.matrix3x3 @ self.prop3.base) + vector(LTrot.matrix3x3 @ self.prop3.top)) / 2.0
            new_prop4_pos = vector(xpos, ypos, zpos) + (
                        vector(LTrot.matrix3x3 @ self.prop4.base) + vector(LTrot.matrix3x3 @ self.prop4.top)) / 2.0

            new_prop_rot_axis = LTrot.matrix3x3 @ vector(0, 0, 1)

            self.prop1.pos(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2])
            self.prop2.pos(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2])
            self.prop3.pos(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2])
            self.prop4.pos(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2])

            LT1.rotate(ang1, axis=new_prop_rot_axis,
                       point=(new_prop1_pos[0], new_prop1_pos[1], new_prop1_pos[2]))
            self.prop1.apply_transform(LT1)

            LT2.rotate(ang2, axis=new_prop_rot_axis,
                       point=(new_prop2_pos[0], new_prop2_pos[1], new_prop2_pos[2]))
            self.prop2.apply_transform(LT2)

            LT3.rotate(ang3, axis=new_prop_rot_axis,
                       point=(new_prop3_pos[0], new_prop3_pos[1], new_prop3_pos[2]))
            self.prop3.apply_transform(LT3)

            LT4.rotate(ang4, axis=new_prop_rot_axis,
                       point=(new_prop4_pos[0], new_prop4_pos[1], new_prop4_pos[2]))
            self.prop4.apply_transform(LT4)

            # self.plt.camera.SetPosition(xpos, ypos + 10, zpos)
            # self.plt.camera.SetFocalPoint(xpos, ypos, zpos)

            self.plt.render()

            Ltrans.reset()
            LTrot.reset()
            LT1.reset()
            LT2.reset()
            LT3.reset()
            LT4.reset()
            self.quad_frame.transform.reset()

            prev_ang1 = ang1
            prev_ang2 = ang2
            prev_ang3 = ang3
            prev_ang4 = ang4

        self.plt.interactive().close()



# ************************************************************************* #
#                          Main for testing                                 #
# ************************************************************************* #
if __name__ == "__main__":

    params = Munch({
        'dx': 0.225,
        'dy': 0.225,
    })

    quad_vedo_model = QuadVedoModel(params=params)
    #quad_vedo_model.test_translation(x_vel=-1.0, y_vel=0.0, z_vel=0.0)
    #quad_vedo_model.test_rotation(mode="roll")
    quad_vedo_model.test_general_motion(x_vel=-0.0, y_vel=0.0, z_vel=0.0, yaw_rate=0, pitch_rate=0, roll_rate=0)
    #quad_vedo_model.plt.show(camera=camera)
