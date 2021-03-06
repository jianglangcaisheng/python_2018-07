
import numpy as np

joint = np.array([15, 5, 9, 13, 15, 15, 17, 21, 25, 28,
                  31, 34, 37, 40, 40, 40, 39, 41, 48, 49,
                  58, 59, 40])

pose_low=np.array([[-45,-90,-45],                       # 1
            [-5,-90,-45],                     # 2
            [-90,-5,-150],                       # 3
            [-5,-45,-130],                       # 4
            [-5,-15,-15],                     # 5
            [-5,-15,-15],                       # 6
            [-5,-170,-100],                        # 7
            [-90,-5,-15],                        # 8
            [-5,-90,-45],                        # 9
            [-4,-6,-6],                       # 10
            [-4,-6,-6],                       # 11
            [-4,-6,-6],                        # 12
            [-4,-6,-6],                       # 13
            [-4,-6,-6],                        # 14
            [-4,-6,-6],                       # 15
            [-4,-6,-6],                       # 16
            [-145,-90,-90],                       # 17
            [-145,-90,-45],                        # 18
            [-15,-5,-5],                        # 19
            [-15,-5,-5],                        # 20
            [-30,-35,-5],                       # 21
            [-30,-35,-5]]) * np.pi / 180.                   # 22

pose_up=np.array([   [45,90,45],                        # 1
            [5,90,45],                        # 2
            [90,5,15],                       # 3
            [180,170,100],                       # 4
            [5,15,15],                        # 5
            [5,15,15],                        # 6
            [180,45,130],                        # 7
            [90,5,150],                       # 8
            [5,90,45],                       # 9
            [20,6,6],                       # 10
            [20,6,6],                       # 11
            [20,6,6],                        # 12
            [20,6,6],                        # 13
            [20,6,6],                        # 14
            [20,6,6],                        # 15
            [20,6,6],                        # 16
            [90,90,45],                       # 17
            [90,90,90],                        # 18
            [150,5,5],                        # 19
            [150,5,5],                        # 20
            [50,35,5],                       # 21
            [50,35,5]]) * np.pi / 180.                   # 22

bodyBall2r = np.array([1, 2, 3,
                                               4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                               14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4,
                                               16, 17, 16,
                                               18, 19, 18,
                                               20, 21, 20,
                                               22, 23, 22,
                                               24, 25, 24,      # 跟关节
                                               26, 26,
                                               27, 27,
                                               28, 28 ,
                                               29 ,29,          # 膝盖
                                               30, 30,
                                               31, 31,
                                               32, 32,
                                               33 ,33,
                                               34, 34,          # 脚踝
                                               35, 35,
                                               36, 36])