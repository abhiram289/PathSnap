import unittest
import pandas as pd

from src.preprocessing.pipeline import validate_gps_track


class TestGPSValidation(unittest.TestCase):

    def test_valid_coordinates(self):
        df = pd.DataFrame({
            "lon": [77.5, 77.6],
            "lat": [12.9, 13.0],
            "time": [100, 200]
        })
        result = validate_gps_track(df)
        self.assertEqual(len(result), 2)

    def test_invalid_coordinates_removed(self):
        df = pd.DataFrame({
            "lon": [77.5, 200],
            "lat": [12.9, 13.0],
            "time": [100, 200]
        })
        result = validate_gps_track(df)
        self.assertEqual(len(result), 1)

    def test_missing_values_removed(self):
        df = pd.DataFrame({
            "lon": [77.5, None],
            "lat": [12.9, 13.0],
            "time": [100, 200]
        })
        result = validate_gps_track(df)
        self.assertEqual(len(result), 1)

    def test_empty_dataframe_raises_error(self):
        df = pd.DataFrame(columns=["lon", "lat", "time"])
        with self.assertRaises(ValueError):
            validate_gps_track(df)

    def test_missing_column_raises_error(self):
        df = pd.DataFrame({
            "lon": [77.5],
            "lat": [12.9]
        })
        with self.assertRaises(ValueError):
            validate_gps_track(df)


if __name__ == "__main__":
    unittest.main()
