import json
import numpy
import unittest

# from . import utilities
import utilities

# utilities.build_camera_matrices(vals, width, height)

# distort_points(points, cameraMatrix, distortionMatrix)

# generate_grid_points(lines_num, line_res, w, h)


class TestUtilities(unittest.TestCase):

    def test_build_camera_matrices(self):
        # expected resultDistortionParameters
        expectedCameraMatrix = '[[12.0, 0.0, 50.0], [0.0, 12.0, 50.0], [0.0, 0.0, 1.0]]'
        expectedDistortionParameters = '[[-0.001], [0.001], [-0.001], [-0.001]]'

        # basic parameters
        vals = [ 12, 12, -0.001, 0.001, -0.001, -0.001 ]
        width = 100
        height = 100

        # run method
        resultCameraMatrix, resultDistortionParameters = utilities.build_camera_matrices(vals, width, height)

        # check types of results
        self.assertTrue( numpy.ndarray == type(resultCameraMatrix) )
        self.assertTrue( numpy.ndarray == type(resultDistortionParameters) )

        # check values of results
        self.assertTrue(expectedCameraMatrix == json.dumps(resultCameraMatrix.tolist()))
        self.assertTrue( expectedDistortionParameters == json.dumps(resultDistortionParameters.tolist()) )


if __name__ == '__main__':
    unittest.main()
