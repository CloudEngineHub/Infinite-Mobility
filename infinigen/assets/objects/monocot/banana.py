# Copyright (C) 2023, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

# Authors: Lingjie Mei

import bmesh
import numpy as np
from numpy.random import uniform

from infinigen.assets.objects.monocot.growth import MonocotGrowthFactory
from infinigen.assets.utils.decorate import displace_vertices, read_co
from infinigen.assets.utils.draw import bezier_curve, leaf
from infinigen.assets.utils.nodegroup import geo_radius
from infinigen.assets.utils.object import (
    join_objects,
    origin2lowest,
    save_objects,
    save_parts_join_objects,
)
from infinigen.core import surface
from infinigen.core.tagging import tag_object
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed
from infinigen.core.util.random import log_uniform


class BananaMonocotFactory(MonocotGrowthFactory):
    def __init__(self, factory_seed, coarse=False):
        super(BananaMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(0.6, 1.0)
            self.angle = uniform(np.pi / 4, np.pi / 3)
            self.z_scale = uniform(1, 1.5)
            self.z_drag = uniform(0.1, 0.2)
            self.min_y_angle = uniform(np.pi * 0.05, np.pi * 0.1)
            self.max_y_angle = uniform(np.pi * 0.25, np.pi * 0.45)
            self.leaf_range = uniform(0.5, 0.7), 1
            self.count = int(log_uniform(16, 24))
            self.scale_curve = [(0, uniform(0.4, 1.0)), (1, uniform(0.6, 1.0))]
            self.radius = uniform(0.04, 0.06)
            self.bud_angle = uniform(np.pi / 8, np.pi / 6)
            self.cut_angle = self.bud_angle + uniform(np.pi / 20, np.pi / 12)
            self.freq = log_uniform(100, 300)
            self.n_cuts = np.random.randint(6, 10) if uniform(0, 1) < 0.8 else 0

    @staticmethod
    def build_base_hue():
        return uniform(0.15, 0.35)

    def cut_leaf(self, obj):
        coords = read_co(obj)
        x, y, z = coords.T
        coords = coords[(np.abs(y) < 0.08) & (np.abs(y) > 0.01)]
        positive_coords = coords[coords.T[1] > 0]
        positive_coords = positive_coords[np.argsort(positive_coords[:, 0])]
        negative_coords = coords[coords.T[1] < 0]
        negative_coords = negative_coords[np.argsort(negative_coords[:, 0])]
        positive_coords = positive_coords[
            np.random.choice(len(positive_coords), self.n_cuts, replace=False)
        ]
        negative_coords = negative_coords[
            np.random.choice(len(negative_coords), self.n_cuts, replace=False)
        ]

        for (x1, y1, _), (x2, y2, _) in zip(
            np.concatenate([positive_coords[:-1], negative_coords[:-1]], 0),
            np.concatenate([positive_coords[1:], negative_coords[1:]], 0),
        ):
            coeff = 1 if y1 > 0 else -1
            ratio = uniform(-2.0, 0.4)
            exponent = uniform(1.2, 1.6)

            def cut(x, y, z):
                m1 = x1 * np.sin(self.cut_angle) - y1 * np.cos(self.cut_angle) * coeff
                m2 = x2 * np.sin(self.cut_angle) - y2 * np.cos(self.cut_angle) * coeff
                m = x * np.sin(self.cut_angle) - y * np.cos(self.cut_angle) * coeff
                dist = ((x - x1) * (y1 - y2) + (y - y1) * (x1 - x2)) / np.sqrt(
                    (x1 - x2) ** 2 + (y1 - y2) ** 2 + 0.1
                )
                return (
                    0,
                    0,
                    np.where(
                        (m1 < m) & (m < m2) & (dist * coeff < 0),
                        ratio * np.abs(dist) ** exponent,
                        0,
                    ),
                )

            displace_vertices(obj, cut)
        with butil.ViewportMode(obj, "EDIT"):
            bm = bmesh.from_edit_mesh(obj.data)
            geom = [e for e in bm.edges if e.calc_length() > 0.02]
            bmesh.ops.delete(bm, geom=geom, context="EDGES")
            bmesh.update_edit_mesh(obj.data)

    def build_leaf(self, face_size):
        x_anchors = 0, 0.2 * np.cos(self.bud_angle), uniform(0.8, 1.2), 2.0
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.2, 0.25), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.cut_leaf(obj)
        self.displace_veins(obj)
        self.decorate_leaf(obj)
        tag_object(obj, "banana")
        return obj

    def displace_veins(self, obj):
        vg = obj.vertex_groups.new(name="distance")
        x, y, z = read_co(obj).T
        branch = np.cos(
            (np.abs(y) * np.cos(self.cut_angle) - x * np.sin(self.cut_angle))
            * self.freq
        ) > uniform(0.85, 0.9, len(x))
        leaf = np.abs(y) < uniform(0.002, 0.008, len(x))
        weights = branch | leaf
        for i, l in enumerate(weights):
            vg.add([i], l, "REPLACE")
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=-uniform(5e-3, 8e-3),
            mid_level=0,
            vertex_group="distance",
        )


class TaroMonocotFactory(BananaMonocotFactory):
    def __init__(self, factory_seed, coarse=False):
        super(TaroMonocotFactory, self).__init__(factory_seed, coarse)
        with FixedSeed(factory_seed):
            self.stem_offset = uniform(0.05, 0.1)
            self.radius = uniform(0.02, 0.04)
            self.z_drag = uniform(0.2, 0.3)
            self.bud_angle = uniform(np.pi * 0.6, np.pi * 0.7)
            self.freq = log_uniform(10, 20)
            self.count = int(log_uniform(12, 16))
            self.n_cuts = np.random.randint(1, 2) if uniform(0, 1) < 0.5 else 0
            self.min_y_angle = uniform(-np.pi * 0.25, -np.pi * 0.05)
            self.max_y_angle = uniform(-np.pi * 0.05, 0)

    def displace_veins(self, obj):
        vg = obj.vertex_groups.new(name="distance")
        x, y, z = read_co(obj).T
        branch = np.cos(
            uniform(0, np.pi * 2)
            + np.arctan2(
                y - np.where(y > 0, -1, 1) * uniform(0.1, 0.2), x - uniform(0.1, 0.4)
            )
            * self.freq
        ) > uniform(0.98, 0.99, len(x))
        leaf = np.abs(y) < uniform(0.002, 0.008, len(x))
        weights = branch | leaf
        for i, l in enumerate(weights):
            vg.add([i], l, "REPLACE")
        butil.modify_mesh(
            obj,
            "DISPLACE",
            strength=-uniform(5e-3, 8e-3),
            mid_level=0,
            vertex_group="distance",
        )

    def build_leaf(self, face_size):
        x_anchors = (
            0,
            0.2 * np.cos(self.bud_angle),
            uniform(0.4, 1.0),
            uniform(0.8, 1.0),
        )
        y_anchors = 0, 0.2 * np.sin(self.bud_angle), uniform(0.25, 0.3), 0
        obj = leaf(x_anchors, y_anchors, face_size=face_size)
        self.cut_leaf(obj)
        self.displace_veins(obj)
        self.decorate_leaf(obj, 2, leftmost=False)
        bezier = self.build_branch()
        obj = join_objects([obj, bezier])
        origin2lowest(obj)
        tag_object(obj, "taro")
        return obj

    def build_branch(self):
        offset = uniform(0.2, 0.3)
        length = uniform(1, 2)
        x_anchors = 0, -0.05, -offset - uniform(0.01, 0.02), -offset
        z_anchors = 0, 0, -length + 0.1, -length
        bezier = bezier_curve([x_anchors, 0, z_anchors])
        surface.add_geomod(
            bezier, geo_radius, apply=True, input_args=[uniform(0.02, 0.03), 32]
        )
        return bezier

    def build_instance(self, i, face_size):
        return self.build_leaf(face_size)
