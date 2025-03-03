import os
from matplotlib import pyplot as plt
import open3d as o3d
import numpy as np

def pcd_joint(pcd_a, pcd_b, save_path):
    # 对其两个点云
    center_a = pcd_a.get_center()
    center_b = pcd_b.get_center()
    # 平移
    pcd_b.translate(center_a - center_b)
    # 以每0.1厚度在x轴方向切片，取在yz平面上的所有点，保存为yz_{面}.png
    # 获取a，b在x轴上的最大最小值
    max_a_x, _, _ = pcd_a.get_max_bound()
    min_a_x, _, _ = pcd_a.get_min_bound()
    # 以最大最小值为边界，每scale/16切片
    bound = max_a_x - min_a_x
    scale = bound / 20
    # 切片
    for i in range(20):
        # 获取切片
        x_min = min_a_x + i * scale
        x_max = min_a_x + (i + 1) * scale
        min_bound = np.array([x_min, -0.7, -0.7])
        max_bound = np.array([x_max, 0.7, 0.7])
        box_bound = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_a_slice = pcd_a.crop(box_bound)
        pcd_b_slice = pcd_b.crop(box_bound)
        # 忽略x轴，保留yz面至png
        yz_a = pcd_a_slice.points
        yz_b = pcd_b_slice.points
        # 转为float
        yz_a = np.array(yz_a).astype(np.float)
        yz_b = np.array(yz_b).astype(np.float)
        # 画图, 固定画布大小为(10,10)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(yz_a[:, 1], yz_a[:, 2], c='r')
        ax.scatter(yz_b[:, 1], yz_b[:, 2], c='b')
        plt.savefig(f'{save_path}/{i}_sag.png')
        plt.close()
        print(f'{save_path}/{i}_sag.png is saved')
    return

def pcd_joint_cor(pcd_a, pcd_b, save_path):
    # 对其两个点云
    center_a = pcd_a.get_center()
    center_b = pcd_b.get_center()
    # 平移
    pcd_b.translate(center_a - center_b)
    # 以每0.1厚度在x轴方向切片，取在yz平面上的所有点，保存为yz_{面}.png
    # 获取a，b在y轴上的最大最小值
    _, max_a_y, _ = pcd_a.get_max_bound()
    _, min_a_y, _ = pcd_a.get_min_bound()
    # 以最大最小值为边界，每scale/16切片
    bound = max_a_y - min_a_y
    scale = bound / 20
    # 切片
    for i in range(20):
        # 获取切片
        y_min = min_a_y + i * scale
        y_max = min_a_y + (i + 1) * scale
        min_bound = np.array([-1, y_min, -1])
        max_bound = np.array([1, y_max, 1])
        box_bound = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        pcd_a_slice = pcd_a.crop(box_bound)
        pcd_b_slice = pcd_b.crop(box_bound)
        # 忽略y轴，保留xz面至png
        xz_a = pcd_a_slice.points
        xz_b = pcd_b_slice.points
        # 转为float
        xz_a = np.array(xz_a).astype(np.float)
        xz_b = np.array(xz_b).astype(np.float)
        # 画图, 固定画布大小为(10,10)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.scatter(xz_a[:, 0], xz_a[:, 2], c='r')
        ax.scatter(xz_b[:, 0], xz_b[:, 2], c='b')
        plt.savefig(f'{save_path}/{i}_cor.png')
        plt.close()
        print(f'{save_path}/{i}_cor.png is saved')
    return

# 采样点云至指定数量
def sample_point_cloud(pcd,  num_points=2048):
    ply_cloud = pcd
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
    return sampled_cloud


# 用于对比手动标注和自动标注的点云
def auto_manual():
    manual_path = '/home/lc/code/BYSY/bone_for_ligament/3D/PCT_Pytorch-main/data/Kneebone/sampled_pointclouds_from_normal_fps'
    auto_path = '/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/aclnettest_ply/sampled_pointclouds_from_test'
    label_path = '/home/lc/code/BYSY/bone_for_ligament/3D/v0.3_D3D_PCT/utils/label_test.txt'
    with open(label_path, 'r') as f:
        label_info = f.readlines()
    for line in label_info:
        pid = line.strip('\n').split()[0]
        auto_pcd_path = os.path.join(auto_path, f'{pid}_2048.ply')
        manual_pcd_path = os.path.join(manual_path, f'{pid}_2048.ply')
        auto_pcd = o3d.io.read_point_cloud(auto_pcd_path)
        manual_pcd = o3d.io.read_point_cloud(manual_pcd_path)
        try:
            assert auto_pcd is not None and manual_pcd is not None
        except:
            print(f'Error: some path is not exist')
            continue
        save_path = f'/home/lc/code/BYSY/bone_for_ligament/3D/PCT_Pytorch-main/data/compare_result_auto_manual/{pid}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pcd_joint(auto_pcd, manual_pcd, save_path)
        pcd_joint_cor(auto_pcd, manual_pcd, save_path)
        print(f'{pid} is done')
    return



# 用于对比配对距离最近的点云
def cloud_compare():
    # test_p_path = '/data1/liuchao/dataset/evaluate_15000/cloud_compare/3583_6016_02201009210055_02200617208040_0_1'
    # cloud_list = [i for i in os.listdir(test_p_path) if 'py' not in i]
    # cloud_1 = cloud_list[0]
    # cloud_2 = cloud_list[1]

    # pcd_a = o3d.io.read_point_cloud(os.path.join(test_p_path, cloud_1))
    # pcd_b = o3d.io.read_point_cloud(os.path.join(test_p_path, cloud_2))
    # save_path = '/home/lc/code/BYSY/bone_for_ligament/3D/PCT_Pytorch-main/data/cloud_compare_sample_result'
    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # pcd_joint(pcd_a, pcd_b, save_path)
    pcloud_path = '/data1/liuchao/dataset/evaluate_15000/cloud_compare'
    cloud_list = [i for i in os.listdir(pcloud_path) if 'py' not in i]
    for cloud in cloud_list:
        cloud_path = os.path.join(pcloud_path, cloud)
        if cloud.split('_')[-1] == '0':
            pid_a = cloud.split('_')[2]
            pid_b = cloud.split('_')[3]
        else:
            pid_a = cloud.split('_')[3]
            pid_b = cloud.split('_')[2]
        pcd_a_name = f'{pid_a}_aligned_with_normal_cor.ply'
        pcd_b_name = f'{pid_b}_aligned_with_normal_cor.ply'
        pcd_a = o3d.io.read_point_cloud(os.path.join(cloud_path, pcd_a_name))
        pcd_b = o3d.io.read_point_cloud(os.path.join(cloud_path, pcd_b_name))
        save_path = f'/home/lc/code/BYSY/bone_for_ligament/3D/PCT_Pytorch-main/data/cloud_compare_result_cor/{cloud}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pcd_sample_a = sample_point_cloud(pcd_a)
        pcd_sample_b = sample_point_cloud(pcd_b)
        pcd_joint_cor(pcd_sample_a, pcd_sample_b, save_path)
        print(f'{cloud} is done')

if __name__ == '__main__':

    auto_manual()