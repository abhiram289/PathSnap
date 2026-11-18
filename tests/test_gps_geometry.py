import unittest
import numpy as np

from src.preprocessing.pipeline import (
    project_to_meters,
    point_to_segment_dist_and_proj,
    calculate_bearing,
    angle_difference,
)


class TestGPSGeometry(unittest.TestCase):

    def test_project_to_meters(self):
        x, y = project_to_meters(
            np.array([0.001]), np.array([0.0]), 0.0, 0.0
        )
        self.assertAlmostEqual(x[0], 111.19, places=1)
        self.assertAlmostEqual(y[0], 0.0, places=5)

    def test_point_to_segment_distance(self):
        distance, ratio, length = point_to_segment_dist_and_proj(
            5, 3, 0, 0, 10, 0
        )
        self.assertAlmostEqual(distance, 3.0)
        self.assertAlmostEqual(ratio, 0.5)
        self.assertAlmostEqual(length, 10.0)

    def test_projection_clamps_to_segment_endpoint(self):
        distance, ratio, length = point_to_segment_dist_and_proj(
            15, 0, 0, 0, 10, 0
        )
        self.assertAlmostEqual(distance, 5.0)
        self.assertAlmostEqual(ratio, 1.0)
        self.assertAlmostEqual(length, 10.0)

    def test_zero_length_segment(self):
        distance, ratio, length = point_to_segment_dist_and_proj(
            1, 2, 0, 0, 0, 0
        )
        self.assertAlmostEqual(distance, np.sqrt(5))
        self.assertEqual(ratio, 0.0)
        self.assertEqual(length, 0.0)

    def test_bearing_north(self):
        bearing = calculate_bearing(0, 0, 0, 1)
        self.assertAlmostEqual(bearing, np.pi / 2)

    def test_angle_difference_wraparound(self):
        angle = angle_difference(-np.pi + 0.1, np.pi - 0.1)
        self.assertAlmostEqual(angle, 0.2)

    def test_identical_angles(self):
        angle = angle_difference(1.2, 1.2)
        self.assertAlmostEqual(angle, 0.0)


if __name__ == "__main__":
    unittest.main()
