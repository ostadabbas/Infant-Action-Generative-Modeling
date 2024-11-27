import pickle as pkl
import numpy as np
import os
from .dataset import Dataset


class CareerPostures(Dataset):
    dataname = "careerposture"

    def __init__(self, datapath="./data/Career", **kargs):
        self.datapath = datapath
        
        super().__init__(**kargs)
        pkldatafilepath = os.path.join(datapath, "Career_posture.pkl")
        ori_data = pkl.load(open(pkldatafilepath, "rb"))
        all_poses = []
        all_kpts = []
        all_labels = []
        all_names = []

        cnt0 = 0
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        for sample in ori_data:
            if sample['frame_dir'][:3] != 'D01':
                continue
            if sample['pos_label'] == 'Supine':
                cnt0 += 1
                all_labels.append(0)
            elif sample['pos_label'] == 'Prone':
                cnt1 += 1
                all_labels.append(1)
            elif sample['pos_label'] == 'Sitting':
                cnt2 += 1
                all_labels.append(2)
            elif sample['pos_label'] == 'Standing':
                cnt3 += 1
                all_labels.append(3)
            elif sample['pos_label'] == 'All-fours':
                cnt4 += 1
                all_labels.append(4)
            else:
                continue

            nframes = sample['pose'][0].shape[0]
            all_poses.append(sample['pose'][0].reshape(nframes, 24, 3))
            #all_poses.append(sample['pose'][0])
            all_kpts.append(sample['keypoint'][0])
            all_names.append(sample['frame_dir'])
        
        '''
        pkldatafilepath = os.path.join(datapath, "Career_posture.pkl")
        ori_data = pkl.load(open(pkldatafilepath, "rb"))
        print(ori_data['annotations'][0])
        all_poses = []
        all_kpts = []
        all_labels = []
        all_names = []

        cnt0 = 0
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        cnt4 = 0
        for i in range(len(ori_data['annotations'])):
            sample = ori_data['annotations'][i]

            if sample['frame_dir'][:3] != 'D02':
                continue
            if sample['pos_label'] == 'Supine':
                cnt0 += 1
                all_labels.append(0)
            elif sample['pos_label'] == 'Prone':
                cnt1 += 1
                all_labels.append(1)
            elif sample['pos_label'] == 'Sitting':
                cnt2 += 1
                all_labels.append(2)
            elif sample['pos_label'] == 'Standing':
                cnt3 += 1
                all_labels.append(3)
            elif sample['pos_label'] == 'All-fours':
                cnt4 += 1
                all_labels.append(4)
            else:
                continue
            print(sample['keypoint'].shape)
            nframes = sample['keypoint'][0].shape[0]
            all_poses.append(sample['pose'][0].reshape(nframes, 24, 3))
            #all_poses.append(sample['pose'][0])
            all_kpts.append(sample['keypoint'][0])
            all_names.append(sample['frame_dir'])
        '''
        print('Career Posture Data Distribution')
        print(cnt0)
        print(cnt1)
        print(cnt2)
        print(cnt3)
        print(cnt4)
        
        data = {
            'pose': all_poses,
            'keypoint': all_kpts,
            'frame_dir': all_names,
            'pos_label': all_labels
        }

        print(len(data["pose"]))
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
