import os
import numpy as np
import pickle
import open3d as o3d
from torch.utils.data import Dataset

def load_data(num_points, partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    with open(os.path.join(DATA_DIR, 'Kneebone', 'sampled_data_%d_%s.pkl'%(num_points, partition)), 'rb') as file:
        data = pickle.load(file)
    for i in range(len(data)):
        all_data.append(np.array(data[i]['data']))
        if data[i]['label'] == 't1':
            all_label.append(np.array([0]))
        elif data[i]['label'] == 't2':
            all_label.append(np.array([1]))
    return all_data, all_label


def load_data_2(num_points, partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    # DATA_DIR = '/home/lc/code/BYSY/bone_for_ligament/segmentation/Seg/aclnettest_ply/sampled_pointclouds_from_test'
    DATA_DIR = '/home/lc/code/BYSY/bone_for_ligament/3D/PCT_Pytorch-main/data/Kneebone/sampled_pointclouds_from_normal_fps'
    all_data = []
    all_label = []
    t1_cnt = 0
    t2_cnt = 0
    # 根据partition从对应的label_{partition}.txt中读取标签, 然后根据txt中每行的pid和label加载对应的ply文件
    with open(os.path.join(BASE_DIR, 'data', 'Kneebone', 'label_%s.txt'%partition), 'r') as file:
        for line in file:
            pid, label = line.strip('\t').split()
            # 读取对应的ply点云文件
            ply_path = os.path.join(DATA_DIR, '%s_%d.ply' %(pid, num_points))
            ply_cloud = o3d.io.read_point_cloud(ply_path)
            points = np.array(ply_cloud.points)
            # 读取标签
            if label == 't1':
                label = 0
                t1_cnt += 1
            elif label == 't2':
                label = 1
                t2_cnt += 1
            else:
                raise Exception('label error')
            all_data.append(points)
            all_label.append(label)
    print('t1 samples:', t1_cnt)
    print('t2 samples:', t2_cnt)
    return all_data, all_label

def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

class BonePointNet(Dataset):
    def __init__(self, num_points, partition='train'):

        self.num_points = num_points
        self.partition = partition

        if self.partition == 'train':
            # 训练点云路径
            dpath = '/data1/liuchao/dataset/Point_Circle_All/Point/train_patient_info_final.txt'
        elif self.partition == 'test':
            # 测试点云路径
            dpath = '/data1/liuchao/dataset/Point_Circle_All/Point/test_patient_info_final.txt'
        else:
            # 不同的验证路径
            dpath = '/data1/liuchao/dataset/Point_Circle_All/Point/val_532.txt'

        # 外部验证点云路径
        out_dpath = '/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/'
        # out_dpath = '/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/femur_tibia/'

        # 原始训练点云路径
        source_dpath = '/home/lc/code/BYSY/bone_for_ligament/3D/PCT_Pytorch-main/data/Kneebone/sampled_pointclouds_from_normal_sag_fps'
        # source_dpath = '/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/femur_tibia/source'

        # 读取点云，若为val，保存对应pid
        self.data = []
        self.label = []
        if self.partition == 'val':
            self.pid = []
        with open(dpath, 'r') as file:
            for line in file:
                ptype, pid, plabel = line.strip().split('_')
                if ptype == 'source':
                    ply_path = os.path.join(source_dpath, '%s_%d.ply' %(pid, num_points))
                    # ply_path = os.path.join(source_dpath, '%s_aligned_femur_tibia_%d.ply' %(pid, num_points))
                else:
                    ply_path = os.path.join(out_dpath, ptype, '%s_aligned_with_normal_sag_%d.ply' %(pid, num_points))
                    # ply_path = os.path.join(out_dpath, ptype, '%s_aligned_femur_tibia_%d.ply' %(pid, num_points))
                if not os.path.exists(ply_path):
                    # print('not exist:', ply_path)
                    continue
                self.data.append(ply_path)

                if self.partition == 'val':
                    self.pid.append(pid)
                if plabel == 't1':
                    self.label.append(0)
                elif plabel == 't2':
                    self.label.append(1)


    def __getitem__(self, item):
        ply_cloud = o3d.io.read_point_cloud(self.data[item])
        pointcloud = np.array(ply_cloud.points)
        # 有一些点云的点数不够2048(2047等)，补0
        if pointcloud.shape[0] < self.num_points:
            pointcloud = np.concatenate([pointcloud, np.zeros((self.num_points - pointcloud.shape[0], 3))], axis=0)
        label = self.label[item]


        # 转为float32
        pointcloud = pointcloud.astype('float32')
        
        # if self.partition == 'train':
        #     pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
        #     pointcloud = translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)

        if self.partition == 'val':
            pid = self.pid[item]
            return pointcloud, label, pid

        return pointcloud, label

    def __len__(self):
        return len(self.data)

class BonePointNet2(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data_2(num_points, partition)
        # 转为array
        self.data = np.array(self.data, dtype = "object")
        self.num_points = num_points
        self.partition = partition
        print('data shape:', self.data.shape)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        # 有一些点云的点数不够2048(2047等)，补0
        if pointcloud.shape[0] < self.num_points:
            pointcloud = np.concatenate([pointcloud, np.zeros((self.num_points - pointcloud.shape[0], 3))], axis=0)
        label = self.label[item]
        # 转为float32
        pointcloud = pointcloud.astype('float32')
        # if self.partition == 'train':
        #     pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
        #     pointcloud = translate_pointcloud(pointcloud)
        #     np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

class FemurTibiaPointNet(Dataset):
    def __init__(self, num_points, partition='train'):

        self.num_points = num_points
        self.partition = partition

        if self.partition == 'train':
            # 训练点云路径
            dpath = '/data1/liuchao/dataset/Point_Circle_All/Point/train_patient_info_final.txt'
        elif self.partition == 'test':
            # 测试点云路径
            dpath = '/data1/liuchao/dataset/Point_Circle_All/Point/test_patient_info_final.txt'
        else:
            # 不同的验证路径
            dpath = '/data1/liuchao/dataset/Point_Circle_All/Point/val_patient_info_final.txt'

        # 外部验证点云路径
        out_dpath = '/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/femur_tibia/'

        # 原始训练点云路径
        source_dpath = '/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/femur_tibia/source'
        # source_dpath = '/home/lc/code/BYSY/bone_for_ligament/3D/PCT_Pytorch-main/data/Kneebone/sampled_pointclouds_from_normal_sag_fps'

        # 读取点云，若为val，保存对应pid
        self.data = []
        self.label = []
        if self.partition == 'val':
            self.pid = []
        with open(dpath, 'r') as file:
            for line in file:
                ply_path_list = []
                ptype, pid, plabel = line.strip().split('_')
                if ptype == 'source':
                    ply_path_1 = os.path.join(source_dpath, '%s_aligned_femur_%d.ply' %(pid, num_points))
                    ply_path_2 = os.path.join(source_dpath, '%s_aligned_tibia_%d.ply' %(pid, num_points))
                else:
                    ply_path_1 = os.path.join(out_dpath, ptype, '%s_aligned_femur_%d.ply' %(pid, num_points))
                    ply_path_2 = os.path.join(out_dpath, ptype, '%s_aligned_tibia_%d.ply' %(pid, num_points))
                if not os.path.exists(ply_path_1) or not os.path.exists(ply_path_2):
                    continue
                ply_path_list.append(ply_path_1)
                ply_path_list.append(ply_path_2)
                self.data.append(ply_path_list)
                if self.partition == 'val':
                    self.pid.append(pid)
                if plabel == 't1':
                    self.label.append(0)
                elif plabel == 't2':
                    self.label.append(1)


    def __getitem__(self, item):
        ply_cloud_femur = o3d.io.read_point_cloud(self.data[item][0])
        ply_cloud_tibia = o3d.io.read_point_cloud(self.data[item][1])
        pointcloud_femur = np.array(ply_cloud_femur.points)
        pointcloud_tibia = np.array(ply_cloud_tibia.points)
        label = self.label[item]


        # 转为float32
        pointcloud_femur = pointcloud_femur.astype('float32')
        pointcloud_tibia = pointcloud_tibia.astype('float32')
        
        if self.partition == 'val':
            pid = self.pid[item]
            return pointcloud_femur, pointcloud_tibia, label, pid

        return pointcloud_femur, pointcloud_tibia, label

    def __len__(self):
        return len(self.data)
    
class BoneEvaluateSet(Dataset):
    def __init__(self, data_path, num_points):
        targetpid_path = '/data1/liuchao/dataset/evaluate_15000/acc_jinbiaozhun_dict.pkl'
        with open(targetpid_path, 'rb') as file:
            self.targetpid = pickle.load(file)
        self.num_points = num_points
        self.data_path = data_path
        self.label = []
        self.data = []
        ply_list = [ ply for ply in os.listdir(data_path) if ply.endswith('sag_2048.ply') ]

        # 统计正负样本比例
        pos = 0
        neg = 0
        for ply in ply_list:
            pid = ply.split('_')[0]
            if pid not in self.targetpid:
                ply_list.remove(ply)
            else:
                self.data.append(ply)
                if self.targetpid[pid] == 't2':
                    self.label.append(1)
                    pos += 1
                else:
                    self.label.append(0)
                    neg += 1
        
        print('t2 samples:', pos)
        print('t1 samples:', neg)

        # 转为array
        # self.data = np.array(self.data)

    def __getitem__(self, item):
        # 读取点云
        ply = self.data[item]
        ply_path = os.path.join(self.data_path, ply)
        ply_cloud = o3d.io.read_point_cloud(ply_path)
        pointcloud = np.array(ply_cloud.points)
        # 有一些点云的点数不够2048(2047等)，补0
        if pointcloud.shape[0] < self.num_points:
            pointcloud = np.concatenate([pointcloud, np.zeros((self.num_points - pointcloud.shape[0], 3))], axis=0)

        # 转为float32
        pointcloud = pointcloud.astype('float32')
        # 读取标签
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # bone_data_path = '/data1/liuchao/dataset/evaluate_15000/npy_point_cloud/777'
    # test = BoneEvaluateSet(bone_data_path, 2048)
    test = BonePointNet(2048, 'val')
    for data, label in test:
        print(data.dtype)
        print(label)
        if data.shape[0] < 2048:
            print('error')
            print(data.shape)
