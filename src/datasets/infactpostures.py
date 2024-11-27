import pickle as pkl
import numpy as np
import os
from .dataset import Dataset


class InfActPostures(Dataset):
    dataname = "infactposture"

    def __init__(self, datapath="data/InfActPrimitive/Data", **kargs):
        self.datapath = datapath

        super().__init__(**kargs)

        ###### train set
        pkldatafilepath = os.path.join(datapath, "primitive_train.pkl")
        ori_data = pkl.load(open(pkldatafilepath, "rb"))
        all_poses = []
        all_kpts = []
        all_labels = []
        all_names = []
        
        for sample in ori_data:
            all_poses.append(sample['pose'][0])
            all_kpts.append(sample['keypoint'][0])
            all_names.append(sample['frame_dir'])
            if sample['pos_label'] == 'Supine':
                all_labels.append(0)
            elif sample['pos_label'] == 'Prone':
                all_labels.append(1)
            elif sample['pos_label'] == 'Sitting':
                all_labels.append(2)
            elif sample['pos_label'] == 'Standing':
                all_labels.append(3)
            elif sample['pos_label'] == 'All-fours':
                all_labels.append(4)

        print(len(all_poses))

        data = {
            'pose': all_poses,
            'keypoint': all_kpts,
            'frame_dir': all_names,
            'pos_label': all_labels
        }

        self._pose = [x for x in data["pose"]]
        self._num_frames_in_video = [p.shape[0] for p in self._pose]
        self._joints = [x for x in data["keypoint"]]
        
        self._actions = [x for x in data["pos_label"]]

        self._scores = [1.0 for x in data["pos_label"]]
        self.epoch_idx = [0 for x in data["pos_label"]]

        total_num_actions = 5
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
    0: "Supine",
    1: "Prone",
    2: "Sitting",
    3: "Standing",
    4: "All-fours"
}
