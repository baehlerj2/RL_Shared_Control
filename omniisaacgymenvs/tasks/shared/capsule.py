import pdb
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')

'''
this class creates a capsule collision model consisting of a rectangle and two half circles

the capsule is defined by the width and length of the rectangle (the circles are on the side defined by the width parameter)

the angle increment is the angle between two lidar rays in the observation. The ranges start at angle 0 in robot frame.
'''


class Capsule():
    def __init__(self, width, length, angle_increment, d_crit):
        self.a = length
        self.b = width
        self.a_crit = self.a
        self.b_crit = self.b + 2 * d_crit
        self.angle_increment = angle_increment
        self.angles = [angle*angle_increment for angle in range(int(360/angle_increment))]
        self.col_ranges = []
        self.crit_ranges = []
        self.theta_c = math.atan(self.a/self.b) * 180 / math.pi
        self.theta_c_crit = math.atan(self.a_crit/self.b_crit) * 180 / math.pi

    #def calculate_ranges(self):
        no_halfcircles = []
        halfcircles = []

        no_halfcircles_crit = []
        halfcircles_crit = []

        alpha = 0
        while alpha <= 90.0:
            beta = (90 - alpha) * math.pi / 180
            if alpha <= self.theta_c:
                z = self.b / (2 * math.cos(alpha * math.pi / 180))
                no_halfcircles.append(z)
            else:
                b = self.a / 2
                r = self.b / 2
                z = math.cos(beta) * (b + math.sqrt(r*r + (r-b)*(r+b)*math.tan(beta)*math.tan(beta)))
                halfcircles.append(z)

            if alpha <= self.theta_c_crit:
                z_crit = self.b_crit / (2 * math.cos(alpha * math.pi / 180))
                no_halfcircles_crit.append(z_crit)

            else:
                b_crit = self.a_crit / 2
                r_crit = self.b_crit / 2
                z_crit = math.cos(beta) * (b_crit + math.sqrt(r_crit*r_crit + (r_crit-b_crit)*(r_crit+b_crit)*math.tan(beta)*math.tan(beta)))
                halfcircles_crit.append(z_crit)

            alpha += self.angle_increment

        halfcircles_rev = halfcircles[::-1]
        no_halfcircles_rev = no_halfcircles[::-1]

        halfcircles_rev_crit = halfcircles_crit[::-1]
        no_halfcircles_rev_crit = no_halfcircles_crit[::-1]

        self.ranges = halfcircles_rev + no_halfcircles_rev + no_halfcircles[1:] + halfcircles[:-1]
        self.ranges += self.ranges

        self.crit_ranges = halfcircles_rev_crit + no_halfcircles_rev_crit + no_halfcircles_crit[1:] + halfcircles_crit[:-1]
        self.crit_ranges += self.crit_ranges


# Instantiate an object of MyClass
my_object = Capsule(0.65, 0.3, 1, 0.2)
#my_object.calculate_ranges()
fig, ax = plt.subplots(1, 1, figsize=(4, 4), subplot_kw={'projection': 'polar'})
angles = [angle * math.pi / 180 for angle in my_object.angles]
markersize = 1
ax.plot(angles, my_object.ranges, linestyle="", marker='o', markersize=markersize, color='r', label='Collision Range ($d_{col}$)')
ax.plot(angles, my_object.crit_ranges, linestyle="", marker='o', markersize=markersize, color='b', label='Critical Range ($d_{crit}$)')

ax.grid(True)

ax.set_title("Capsule Collision Model", va='bottom')
ax.legend(loc='upper right')
fig.tight_layout()
plt.show()

