import os
import pickle
import open3d as o3d
import numpy as np
import cv2

def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data

# 根据mri参数和原始mask，获取单帧的点云真实坐标
def pc_from_mri(mri, mask_path):
    # 单帧总点云
    point_xyz = o3d.geometry.PointCloud()

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mri 4项参数
    try:
        spacing = mri['spacing']
        orient = mri['orientation']
        position = mri['position']
        zlocation = mri['slicelocation']
    except:
        print('error:', mask_path)
        return None
    rowvector = np.multiply(orient[:3], spacing)
    colvector = np.multiply(orient[3:], spacing)
    zvector = np.cross(rowvector, colvector)
    for contour in contours:
        contour = contour.squeeze()
        # 小于50个点的轮廓不要
        if contour.shape[0] < 50:
            continue
        for point in contour:
            xyz = np.array([point[0], point[1], zlocation, 1])

            # 仿射矩阵
            transvector = np.vstack((np.append(rowvector, 0), np.append(colvector, 0), np.append(zvector, 0), np.append(position, 1))).transpose()

            # 当前坐标真实坐标
            tempxyz = np.dot(transvector, xyz)
            point_xyz.points.append(tempxyz[:3])
    
    return point_xyz





# 根据pid生成对应的点云
def pc_from_pid(root_path, pid_dict, target_pid_list):
    # pid_dict为空字典则跳过
    if not pid_dict:
        return None, None, None
    # 获取当前pid
    for p_id in pid_dict:
        pid = p_id.split('_')[1]
        break
    # 若当前pid不在目标pid列表中，则跳过，普通膝关节不需要这一步
    # if pid not in target_pid_list:
    #     return None, None, pid

    # 创建点云
    target_pc_sag = o3d.geometry.PointCloud()
    target_pc_cor = o3d.geometry.PointCloud()

    # 根据当前pid的每一项名字找到对应的mask，根据mri参数，获取单帧的点云真实坐标
    # 区分冠位和矢位
    for item in pid_dict:
        slice_num = item.split('_')[-1]
        orient = item.split('_')[0]
        target_mask_name = f'{slice_num}_{orient}_mask.png'
        mask_path = os.path.join(root_path, pid, target_mask_name)
        try:
            assert os.path.exists(mask_path)
        except AssertionError:
            print(f'{mask_path} not exist')
            continue
        mri_info = pid_dict[item]
        if orient == 'sag':
            pc = pc_from_mri(mri_info, mask_path)
            target_pc_sag += pc
        elif orient == 'cor':
            pc = pc_from_mri(mri_info, mask_path)
            target_pc_cor += pc
        # # 将当前帧点云添加到总点云中
        # if pc is not None:
        #     target_pc += pc
    
    # 返回当前pid的点云
    return target_pc_sag, target_pc_cor, pid

def main(root_path, pkl_path, targetpid_path, save_path):
    diesase_id = root_path.split('/')[-1]
    with open(pkl_path, 'rb') as f:
        mri_data = pickle.load(f)

    with open(targetpid_path, 'rb') as f:
        target_pid_list = pickle.load(f)

    for i in range(len(mri_data)):
        pc_sag, pc_cor, pid = pc_from_pid(root_path, mri_data[i], target_pid_list)
        if pc_sag is None or pc_cor is None:
            continue
        print(f'{diesase_id}__{pid} point done, sag has {len(pc_sag.points)} points, cor has {len(pc_cor.points)} remain {len(mri_data) - i} to do.')
        o3d.io.write_point_cloud(os.path.join(save_path, f'{pid}_sag.ply'), pc_sag)
        o3d.io.write_point_cloud(os.path.join(save_path, f'{pid}_cor.ply'), pc_cor)

# 采样点云至指定数量
def sample_point_cloud(ply_path,  num_points=2048):
    ply_cloud = o3d.io.read_point_cloud(ply_path)
    point_num = np.asarray(ply_cloud.points).shape[0]

    if point_num < num_points:
        sampled_cloud = ply_cloud.uniform_down_sample(1)
        while np.asarray(sampled_cloud.points).shape[0] < num_points:
            sampled_cloud += ply_cloud.uniform_down_sample(1)
        sampled_cloud = np.asarray(sampled_cloud.points)[:num_points]
        # 转回点云
        sampled_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sampled_cloud))
    else:
        sampled_cloud = ply_cloud.farthest_point_down_sample(num_points)
    o3d.io.write_point_cloud(ply_path.replace(".ply", "_" + str(num_points) + ".ply"), sampled_cloud)

# 计算中心点和平均距离
def calculate_center_and_distance(point_cloud):
    center = point_cloud.get_center()
    distance = 0
    for point in point_cloud.points:
        distance += np.linalg.norm(point - center)
    distance /= len(point_cloud.points)
    return center, distance

def normalization(sag_ply, cor_ply):
    sag_point_cloud = o3d.io.read_point_cloud(sag_ply)
    cor_point_cloud = o3d.io.read_point_cloud(cor_ply)
    # 计算sagittal和coronal两个方向的点云的中心点，进行对齐
    sag_center, sag_distance = calculate_center_and_distance(sag_point_cloud)
    cor_center, cor_distance = calculate_center_and_distance(cor_point_cloud)
    # 对齐
    sag_point_cloud.translate(-sag_center)
    cor_point_cloud.translate(-cor_center)
    # 点云normalization 
    sag_point_cloud.scale(1/sag_distance, center=[0, 0, 0])
    cor_point_cloud.scale(1/cor_distance, center=[0, 0, 0])
    # 保存对齐且归一化后的点云
    o3d.io.write_point_cloud(sag_ply.replace('sag.ply', 'aligned_with_normal_sag.ply'), sag_point_cloud)
    o3d.io.write_point_cloud(cor_ply.replace('cor.ply', 'aligned_with_normal_cor.ply'), cor_point_cloud)
    # 合并两个方向的点云并保存
    sag_point_cloud += cor_point_cloud
    o3d.io.write_point_cloud(sag_ply.replace('sag.ply', 'aligned_with_normal.ply'), sag_point_cloud)

def sag_cor_norm():
    sag_ply_list = [i for i in os.listdir(save_path) if i.endswith('sag.ply')]
    for ply in sag_ply_list:
        sag_ply = os.path.join(save_path, ply)
        cor_ply = os.path.join(save_path, ply.replace('sag', 'cor'))
        try:
            assert os.path.exists(cor_ply)
        except AssertionError:
            print(f'{cor_ply} not exist')
            continue
        normalization(sag_ply, cor_ply)
        print(f'{ply} done.')


##-----------------------------------------
##-----------------------------------------##-----------------------------------------
##-----------------------------------------


# 胫骨股骨后处理点云生成
def main_2(root_path, pkl_path, targetpid_path, save_path, target_num):
    diesase_id = root_path.split('/')[-1]
    with open(pkl_path, 'rb') as f:
        mri_data = pickle.load(f)

    with open(targetpid_path, 'rb') as f:
        target_pid_list = pickle.load(f)

    for i in range(len(mri_data)):
        pc_femur, pc_tibia, pid = pc_from_pid_2(root_path, mri_data[i], target_pid_list)
        if pc_femur is None or pc_tibia is None:
            continue
        print(f'{diesase_id}_{pid} point done, femur has {len(pc_femur.points)} points, tibia has {len(pc_tibia.points)} remain {len(mri_data) - i} to do.')
        o3d.io.write_point_cloud(os.path.join(save_path, f'{pid}_femur.ply'), pc_femur)
        o3d.io.write_point_cloud(os.path.join(save_path, f'{pid}_tibia.ply'), pc_tibia)
        target_num -= 1
        print (target_num)
        if target_num == 0:
            break


# 根据pid生成对应的点云
def pc_from_pid_2(root_path, pid_dict, target_pid_list):
    # pid_dict为空字典则跳过
    if not pid_dict:
        return None, None, None
    # 获取当前pid
    for p_id in pid_dict:
        pid = p_id.split('_')[1]
        break
    # 若当前pid不在目标pid列表中，则跳过，普通膝关节不需要这一步
    # if pid not in target_pid_list and 'normal' not in root_path:
    #     return None, None, pid

    # 创建点云
    target_pc_femur = o3d.geometry.PointCloud()
    target_pc_tibia = o3d.geometry.PointCloud()

    # 根据当前pid的每一项名字找到对应的mask，根据mri参数，获取单帧的点云真实坐标
    # 区分冠位和矢位
    for item in pid_dict:
        slice_num = item.split('_')[-1]
        orient = item.split('_')[0]
        target_mask_name = f'{slice_num}_{orient}_mask.png'
        mask_path = os.path.join(root_path, pid, target_mask_name)
        try:
            assert os.path.exists(mask_path)
        except AssertionError:
            print(f'{mask_path} not exist')
            continue
        mri_info = pid_dict[item]
        if orient == 'sag':
            pc_femur,pc_tibia = pc_from_mri_2(mri_info, mask_path)
            target_pc_femur += pc_femur
            target_pc_tibia += pc_tibia
    
    # 返回当前pid的点云
    return target_pc_femur, target_pc_tibia, pid


# 根据mri参数和原始mask，获取单帧的点云胫骨和股骨真实坐标
def pc_from_mri_2(mri, mask_path):
    # 单帧总点云
    point_xyz_femur = o3d.geometry.PointCloud()
    point_xyz_tibia = o3d.geometry.PointCloud()

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # 获取胫骨和股骨的轮廓
    bone_0, bone_1 = get_max_two_contour(mask)

    point_xyz_tibia = pc_from_contour(bone_1, mri)
    point_xyz_femur = pc_from_contour(bone_0, mri)

    return point_xyz_femur, point_xyz_tibia

# 根据轮廓和mri信息生成对应点云
def pc_from_contour(contour, mri_info):
    point_xyz = o3d.geometry.PointCloud()
    if contour is None:
        return point_xyz
    # mri 4项参数
    try:
        spacing = mri_info['spacing']
        orient = mri_info['orientation']
        position = mri_info['position']
        zlocation = mri_info['slicelocation']
    except:
        print('mri info error')
        return None
    rowvector = np.multiply(orient[:3], spacing)
    colvector = np.multiply(orient[3:], spacing)
    zvector = np.cross(rowvector, colvector)
    contour = contour.squeeze()
    for point in contour:
        xyz = np.array([point[0], point[1], zlocation, 1])

        # 仿射矩阵
        transvector = np.vstack((np.append(rowvector, 0), np.append(colvector, 0), np.append(zvector, 0), np.append(position, 1))).transpose()

        # 当前坐标真实坐标
        tempxyz = np.dot(transvector, xyz)
        point_xyz.points.append(tempxyz[:3])  
    return point_xyz

# 根据mask，返回胫骨和股骨的轮廓
def get_max_two_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) < 1:
        return None, None
    if len(contours) == 1:
        center_0 = get_center_point(contours[0])
        if center_0 is None:
            return None, None
        # 以中心点y坐标判定为胫骨或股骨
        if center_0[1] > 200:
            return None, contours[0]
        else:
            return contours[0], None
    # 按照面积排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    center_0 = get_center_point(contours[0])
    center_1 = get_center_point(contours[1])
    if center_0 is None or center_1 is None:
        return None, None
    # 以中心点y坐标判定为胫骨或股骨
    if center_0[1] > center_1[1]:
        return contours[1], contours[0]
    else:
        return contours[0], contours[1]

def sag_cor_norm_2():
    sag_ply_list = [i for i in os.listdir(save_path) if i.endswith('femur.ply')]
    for ply in sag_ply_list:
        sag_ply = os.path.join(save_path, ply)
        cor_ply = os.path.join(save_path, ply.replace('femur', 'tibia'))
        try:
            assert os.path.exists(cor_ply)
        except AssertionError:
            print(f'{cor_ply} not exist')
            continue
        normalization_2(sag_ply, cor_ply)
        print(f'{ply} done.')

def normalization_2(sag_ply, cor_ply):
    sag_point_cloud = o3d.io.read_point_cloud(sag_ply)
    cor_point_cloud = o3d.io.read_point_cloud(cor_ply)
    # 计算sagittal和coronal两个方向的点云的中心点，进行对齐
    sag_center, sag_distance = calculate_center_and_distance(sag_point_cloud)
    cor_center, cor_distance = calculate_center_and_distance(cor_point_cloud)
    # 在xy平面上对齐
    sag_point_cloud.translate([-sag_center[0], -sag_center[1], 0])
    cor_point_cloud.translate([-cor_center[0], -cor_center[1], 0])
    # 点云normalization 
    sag_point_cloud.scale(1/sag_distance, center=[0, 0, 0])
    cor_point_cloud.scale(1/cor_distance, center=[0, 0, 0])
    # 保存对齐且归一化后的点云
    o3d.io.write_point_cloud(sag_ply.replace('femur.ply', 'aligned_femur.ply'), sag_point_cloud)
    o3d.io.write_point_cloud(cor_ply.replace('tibia.ply', 'aligned_tibia.ply'), cor_point_cloud)
    # 合并两个方向的点云并保存
    sag_point_cloud += cor_point_cloud
    o3d.io.write_point_cloud(sag_ply.replace('femur.ply', 'aligned_femur_tibia.ply'), sag_point_cloud)

# 返回轮廓中心点
def get_center_point(contour):
    # 计算轮廓若干点的中心点
    M = cv2.moments(contour)
    if M['m00'] == 0:
        return None
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])
    return center_x, center_y

if __name__ == '__main__':
    # diease_list = ['777', '1343', '932', '975', '3583', '6016', 'normal']
    diease_list = ['532']
    target_num = {'532':10000, '486':10000, '777': 10000, '932': 10000, '975': 10000, '3583': 10000, '6016': 10000, '1343': 10000, 'normal': 10000}
    for diesase_id in diease_list:

        root_path = f'/data1/liuchao/dataset/evaluate_15000/{diesase_id}'
        pkl_path = f'/data1/liuchao/dataset/evaluate_15000/{diesase_id}_mri_info.pkl'
        targetpid_path = f'/data1/liuchao/dataset/evaluate_15000/acc_jinbiaozhun_dict.pkl'
        save_path = f'/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/femur_tibia/{diesase_id}'
        main_2(root_path, pkl_path, targetpid_path, save_path, target_num[diesase_id])

        
        # # 归一化平移合并点云
        sag_cor_norm_2()

        # 对目录下的所有点云进行采样
        cnt = 0
        for file in os.listdir(save_path):
            if not file.endswith('_2048.ply'):
                sample_point_cloud(os.path.join(save_path, file), 2048)
                print(f'{diesase_id} {file} femur_tibia done, total {cnt} done.')
                cnt += 1

        # 不区分股骨胫骨
        save_path = f'/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/{diesase_id}'
        main(root_path, pkl_path, targetpid_path, save_path)
        # 归一化平移合并点云
        sag_cor_norm()

        # 对目录下的所有点云进行采样
        cnt = 0
        for file in os.listdir(save_path):
            if not file.endswith('_2048.ply'):
                sample_point_cloud(os.path.join(save_path, file), 2048)
                print(f'{diesase_id} {file} done, total {cnt} done.')
                cnt += 1
