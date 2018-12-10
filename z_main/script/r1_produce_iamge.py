
import script.m5_get_image_clustered

file_root_dir = r"D:\0_image_1811_2_cali/"
param_K = 1
param_RT = 1
param_D = 1
path_K = file_root_dir + ("K%d/" % param_K)
path_KR = file_root_dir + ("K%d_RT%d/" % (param_K, param_RT))
path_KRD = file_root_dir + ("K%d_RT%d_D%d/" % (param_K, param_RT, param_D))
image_goal_list = [150, 150]

# send % receive
if 1:
    is_receive = False
    if is_receive:
        import function.e3_file_receive_2
    if not is_receive:
        cf_for_m5 = script.m5_get_image_clustered.CF_for_m5(is_inspect=False, is_clusterd=False, is_send_image=True)
        cf_for_m5.path_image = path_KRD
        cf_for_m5.num_frames = 60 * 60
        script.m5_get_image_clustered.get_image_clustered(is_tobeSent=True, cf_for_m5=cf_for_m5)

# inspect image?
if 0:
    is_inspect = True
    cf_for_m5 = script.m5_get_image_clustered.CF_for_m5(is_inspect=is_inspect, is_clusterd=False, is_send_image=False)
    cf_for_m5.path_image = path_KRD
    cf_for_m5.num_frames = 60 * 60
    script.m5_get_image_clustered.get_image_clustered(is_tobeSent=False, cf_for_m5=cf_for_m5)

# get clustered
if 0:
    cf_for_m5 = script.m5_get_image_clustered.CF_for_m5_2(is_send=False, is_send_image=False, is_inspect_image=False, is_clusterd=True)
    cf_for_m5.path_image = path_KRD
    cf_for_m5.num_frames = 60 * 60
    script.m5_get_image_clustered.get_image_clustered2(cf_for_m5=cf_for_m5)

# pose init
if 0:
    import detectPoseFromLi_180510.atest
    detectPoseFromLi_180510.atest.get_pose_by_network(path_image=path_KRD + "extract_body/")

# col
if 0:
    import script.m2_preProduce
    script.m2_preProduce.extract_2bmpAnd2jpg(file_root_dir=path_KRD, image_goal_list=image_goal_list)