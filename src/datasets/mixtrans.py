import pickle as pkl
import numpy as np
import os
from .dataset import Dataset


class MixTrans(Dataset):
    dataname = "mixtrans"

    def __init__(self, datapath="./data/InfActPlus", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)
        
        ###### real set
        pkldatafilepath1 = os.path.join(datapath, 'train_trans_data.pkl')
        ori_data1 = pkl.load(open(pkldatafilepath1, "rb"))
        all_poses = []
        all_kpts = []
        all_labels = []
        all_names = []
        
        for sample in ori_data1:
            all_poses.append(sample['pose'][0])
            all_kpts.append(sample['keypoint'][0])
            all_names.append(sample['frame_dir'])
            if sample['action_label'] == 'Crawling':
                all_labels.append(0)
            elif sample['action_label'] == 'Sitting':
                all_labels.append(1)
            elif sample['action_label'] == 'Standing':
                all_labels.append(2)
            elif sample['action_label'] == 'Rolling':
                all_labels.append(3)
        
        ###### syn set
        datafilepath = os.path.join('./exps/infacttrans_rc_rcxyz_velxyz_init1100_epoch200', "generated_samples.npy") #"generated_samples.npy")
        syn_data = np.load(datafilepath, allow_pickle=True).item()
        mapping = [0, 1, 4, 7, 2, 5, 8, 6, 9, 12, 15, 16, 18, 20, 17, 19, 21]

        for i in range(len(syn_data['pose'])):
            if syn_data['epoch_idx'][i] <= 200:
                all_poses.append(syn_data['pose'][i])
                all_kpts.append(syn_data['keypoint'][i][:, mapping, :])
                all_labels.append(syn_data['y'][i])
                all_names.append('syn')      
        
        data = {
            'pose': all_poses,
            'keypoint': all_kpts,
            'frame_dir': all_names,
            'action_label': all_labels
        }
        print('Size of mixed train set:')
        print(len(data["pose"]))

        self._pose = [x for x in data["pose"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["keypoint"]]
        
        self._actions = [x for x in data["action_label"]]

        self._scores = [1.0 for x in data["action_label"]]
        self.epoch_idx = [0 for x in data["action_label"]]

        self._train = list(range(len(self._pose)))

        #self._test = list(range(len(self._pose)))


        ##### general info
        total_num_actions = 4
        self.num_classes = total_num_actions

        keep_actions = np.arange(0, total_num_actions)

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._action_classes = infact_postures_enumerator

    def _load_joints3D(self, ind, frame_ix):
        return self._joints[ind][frame_ix]

    def _load_rotvec(self, ind, frame_ix):
        return self._pose[ind][frame_ix]

    def add_samples(self, new_data):

        self._actions.extend(new_data['y'])
        self._pose.extend(new_data['pose'])
        self._joints.extend(new_data['keypoint'])
        self._scores.extend(new_data['score'])
        self.epoch_idx.extend(new_data['epoch_idx'])
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._train = list(range(len(self._pose)))
        self._test = list(range(len(self._pose)))
        #self._train.append(len(self._pose) - 1)  # Assuming you want to add it to the training set
        #self._test.append(len(self._pose) - 1)  # Assuming you want to add it to the test set

infact_postures_enumerator = {
    0: "Crawling",
    1: "Sitting",
    2: "Standing",
    3: "Rolling"
}
