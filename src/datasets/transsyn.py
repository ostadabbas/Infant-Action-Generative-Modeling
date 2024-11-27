import pickle as pkl
import numpy as np
import os
from .dataset import Dataset


class TransSyns(Dataset):
    dataname = "transsyn"

    def __init__(self, datapath="./data/Syn/transistional_syn", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        datafilepath = os.path.join('./exps/infacttrans_rc_rcxyz_velxyz_init1100_epoch200', "generated_samples.npy") #os.path.join(datapath, "posture_generated_samples.npy")
        ori_data = np.load(datafilepath, allow_pickle=True).item()
        print(ori_data.keys())
        all_poses = ori_data['pose']
        all_kpts = ori_data['keypoint']
        all_labels = ori_data['y']
        all_names = []

        data = {
            'pose': all_poses,
            'keypoint': all_kpts,
            'frame_dir': all_names,
            'action_label': all_labels
        }
        print('synthetic set:')
        print(len(all_poses))

        self._pose = [x for x in data["pose"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["keypoint"]]
        
        self._actions = [x for x in data["action_label"]]

        self._scores = [1.0 for x in data["action_label"]]
        self.epoch_idx = [0 for x in data["action_label"]]

        total_num_actions = 4
        self.num_classes = total_num_actions

        self._train = list(range(len(self._pose)))
        self._test = list(range(len(self._pose)))

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
