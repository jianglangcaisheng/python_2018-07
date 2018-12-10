
import numpy as np

def pose2AxisAngle(input_pose):
    """in: 1 * 72 ____ out: 4 * 24"""
    pose_3_24 = np.reshape(input_pose, [3, 24], order='F')
    output_aixsAngle = np.zeros([4, 24])

    for i in range(23):
        thetaX = pose_3_24[1 - 1, i]
        Rx = np.array([[1,              0,               0],
                       [0, np.cos(thetaX), -np.sin(thetaX)],
                       [0, np.sin(thetaX),  np.cos(thetaX)]])

        thetaY = pose_3_24[2 - 1, i]
        Ry = np.array([[ np.cos(thetaY), 0,  np.sin(thetaY)],
                       [              0, 1,               0],
                       [-np.sin(thetaY), 0,  np.cos(thetaY)]])

        thetaZ = pose_3_24[3 - 1, i]
        Rz = np.array([[np.cos(thetaZ), -np.sin(thetaZ), 0],
                       [np.sin(thetaZ),  np.cos(thetaZ), 0],
                       [             0,               0, 1]])

        R = np.matmul(np.matmul(Rz, Ry), Rx)
        thetaAxis = np.arccos((np.trace(R) - 1) / 2)

        if(thetaAxis == 0):
            omegaAxis = np.array([[1.732/3],
                                  [1.732/3],
                                  [1.732/3]])
        else:
            omegaAxis = np.array([[R[3 - 1, 2 - 1] - R[2 - 1, 3 - 1]],
                                  [R[1 - 1, 3 - 1] - R[3 - 1, 1 - 1]],
                                  [R[2 - 1, 1 - 1] - R[1 - 1, 2 - 1]]]) / (2 * np.sin(thetaAxis))

        output_aixsAngle[0:3, i] = omegaAxis[0:3, 0]
        output_aixsAngle[3, i] = thetaAxis

    output_aixsAngle[0:3, 23] = pose_3_24[0:3, 23]

    return output_aixsAngle


def axisAngle2Pose(input_axisAngle):
    """in: 4 * 24 ____ out: 1 * 72"""
    output_pose = np.zeros([3, 24])

    for i in range(23):
        wx = np.array([[                         0, -input_axisAngle[3 - 1, i],  input_axisAngle[2 - 1, i]],
                        [input_axisAngle[3 - 1, i],                          0, -input_axisAngle[1 - 1, i]],
                       [-input_axisAngle[2 - 1, i],  input_axisAngle[1 - 1, i],                          0]])
        # R = expm(wx * input_axisAngle[4 - 1, i])
        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) + \
            wx * np.sin(input_axisAngle[4 - 1, i]) + \
            np.matmul(wx, wx) * (1 - np.cos(input_axisAngle[4 - 1, i]))

        theta_X = np.arctan2(R[3 - 1, 2 - 1],
                             R[3 - 1, 3 - 1])
        theta_Y = np.arctan2(-R[3 - 1, 1 - 1],
                             np.sqrt(R[3 - 1, 2 - 1] * R[3 - 1, 2 - 1] +
                                     R[3 - 1, 3 - 1] * R[3 - 1, 3 - 1]))
        theta_Z = np.arctan2(R[2 - 1, 1 - 1],
                             R[1 - 1, 1 - 1])

        output_pose[1 - 1, i] = theta_X
        output_pose[2 - 1, i] = theta_Y
        output_pose[3 - 1, i] = theta_Z

    output_pose[0:3, 23] = input_axisAngle[0:3, 23]
    output_pose = np.reshape(output_pose, [1, 72], order='F')

    return output_pose
