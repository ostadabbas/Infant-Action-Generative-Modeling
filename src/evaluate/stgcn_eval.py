import torch
from tqdm import tqdm

from src.utils.fixseed import fixseed

from src.evaluate.stgcn.evaluate import Evaluation as STGCNEvaluation
# from src.evaluate.othermetrics.evaluation import Evaluation

from torch.utils.data import DataLoader
from src.utils.tensors import collate

import os

from .tools import save_metrics, format_metrics
from src.models.get_model import get_model as get_gen_model
from src.datasets.get_dataset import get_datasets
import src.utils.rotation_conversions as geometry
from src.evaluate.stgcn.accuracy import calculate_accuracy
from src.evaluate.stgcn.fid import calculate_fid
from src.evaluate.stgcn.diversity import calculate_diversity_multimodality
import numpy as np
import src.utils.rotation_conversions as geometry
from src.visualize.visualize import generate_by_ori_video_sequences

def convert_x_to_rot6d(x, pose_rep):
    # convert rotation to rot6d
    if pose_rep == "rotvec":
        x = geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(x))
    elif pose_rep == "rotmat":
        x = x.reshape(*x.shape[:-1], 3, 3)
        x = geometry.matrix_to_rotation_6d(x)
    elif pose_rep == "rotquat":
        x = geometry.matrix_to_rotation_6d(geometry.quaternion_to_matrix(x))
    elif pose_rep == "rot6d":
        x = x
    else:
        raise NotImplementedError("No geometry for this one.")
    return x


class NewDataloader:
    def __init__(self, mode, model, parameters, dataiterator, device):
        assert mode in ["gen", "gen_action", "rc", "gt", "gen_test"]

        pose_rep = parameters["pose_rep"]

        translation = parameters["translation"]

        J_regressor_np = np.load('./J_regressor_h36m.npy')
        J_regressor = torch.tensor(J_regressor_np, dtype=torch.float32, device=device)

        self.batches = []

        with torch.no_grad():
            for databatch in tqdm(dataiterator, desc=f"Construct dataloader: {mode}.."):
                if mode == "gen":
                    #parameters['fps'] = 20
                    classes = databatch["y"]
                    gendurations = databatch["lengths"]
                    print(classes)
                    print(gendurations)
                    batch = model.generate(classes, gendurations)
                    #figname = "{}_{}".format(parameters["dataset"], parameters["pose_rep"])
                    #tmp_path = os.path.join('./test', f"subfigures_{figname}")
                    #os.makedirs(tmp_path, exist_ok=True)
                    #generate_by_ori_video_sequences(0, batch, parameters, tmp_path)
                    feats = "output"
                elif mode == "gt":
                    batch = {key: val.to(device) for key, val in databatch.items()}
                    feats = "x"
                elif mode == "rc":
                    databatch = {key: val.to(device) for key, val in databatch.items()}
                    batch = model(databatch)
                    feats = "output"
                elif mode == "gen_action":
                    classes = []
                    nspa = len(databatch["y"]) // parameters["num_classes"]
                    remain = len(databatch["y"]) % parameters["num_classes"]
                    for i in range(parameters["num_classes"]):
                        classes.extend([i] * nspa)
                    if remain != 0:   
                        class_idx = 0
                        while remain:
                            classes.append(class_idx)
                            class_idx += 1
                            remain -= 1
     
                    classes = torch.tensor(classes, device=device)
                    gendurations = databatch["lengths"]
                    
                    batch = model.generate(classes, gendurations)
                    feats = "output"

                batch = {key: val.to(device) for key, val in batch.items()}

                if translation:
                    x = batch[feats][:, :-1]
                else:
                    x = batch[feats]
                
                x = x.permute(0, 3, 1, 2)
                x = convert_x_to_rot6d(x, pose_rep)
                x = x.permute(0, 2, 3, 1)
                
                batch["x"] = x

                self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        num_seq_max = getattr(self, "num_seq_max", -1)
        if num_seq_max == -1:
            from math import inf
            num_seq_max = inf

        if self.split == 'train':
            return min(len(self._train), num_seq_max)
        else:
            return min(len(self._test), num_seq_max)

def evaluate(parameters, folder, checkpointname, epoch, niter):
    torch.multiprocessing.set_sharing_strategy('file_system')

    bs = parameters["batch_size"]
    doing_recons = False

    device = parameters["device"]
    dataname = parameters["dataset"]

    # dummy => update parameters info
    # get_datasets(parameters)
    # faster: hardcode value for uestc

    parameters["nfeats"] = 6
    parameters["njoints"] = 24

    model = get_gen_model(parameters)

    print("Restore weights..")
    checkpointpath = os.path.join(folder, checkpointname)
    state_dict = torch.load(checkpointpath, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    model.outputxyz = False
    
    recogparameters = parameters.copy()
    recogparameters["pose_rep"] = "rot6d"
    recogparameters["nfeats"] = 6

    # Action2motionEvaluation
    stgcnevaluation = STGCNEvaluation(dataname, recogparameters, device)

    stgcn_metrics = {}
    # joints_metrics = {}
    # pose_metrics = {}

    compute_gt_gt = False
    if compute_gt_gt:
        datasetGT = {key: [get_datasets(parameters)[key],
                           get_datasets(parameters)[key]]
                     for key in ["train", "test"]}
    else:
        datasetGT = {key: [get_datasets(parameters)[key]]
                     for key in ["train", "test"]}

    print("Dataset loaded")

    allseeds = list(range(niter))

    for seed in allseeds:
        fixseed(seed)
        for key in ["train", "test"]:
            for data in datasetGT[key]:
                data.reset_shuffle()
                data.shuffle()

        dataiterator = {key: [DataLoader(data, batch_size=bs,
                                         shuffle=False, num_workers=8,
                                         collate_fn=collate)
                              for data in datasetGT[key]]
                        for key in ["train", "test"]}

        if doing_recons:
            reconsLoaders = {key: NewDataloader("rc", model, parameters,
                                                dataiterator[key][0],
                                                device)
                             for key in ["train", "test"]}

        gtLoaders = {key: NewDataloader("gt", model, parameters,
                                        dataiterator[key][0],
                                        device)
                     for key in ["train", "test"]}

        if compute_gt_gt:
            gtLoaders2 = {key: NewDataloader("gt", model, parameters,
                                             dataiterator[key][1],
                                             device)
                          for key in ["train", "test"]}

        genLoaders = {key: NewDataloader("gen", model, parameters,
                                         dataiterator[key][0],
                                         device)
                      for key in ["train", "test"]}

        loaders = {"gen": genLoaders,
                   "gt": gtLoaders}
        if doing_recons:
            loaders["recons"] = reconsLoaders

        if compute_gt_gt:
            loaders["gt2"] = gtLoaders2

        stgcn_metrics[seed] = stgcnevaluation.evaluate(model, loaders)
        del loaders

        # joints_metrics = evaluation.evaluate(model, loaders, xyz=True)
        # pose_metrics = evaluation.evaluate(model, loaders, xyz=False)

    metrics = {"feats": {key: [format_metrics(stgcn_metrics[seed])[key] for seed in allseeds] for key in stgcn_metrics[allseeds[0]]}}
    # "xyz": {key: [format_metrics(joints_metrics[seed])[key] for seed in allseeds] for key in joints_metrics[allseeds[0]]},
    # model.pose_rep: {key: [format_metrics(pose_metrics[seed])[key] for seed in allseeds] for key in pose_metrics[allseeds[0]]}}
    
    epoch = checkpointname.split("_")[1].split(".")[0]
    metricname = "evaluation_metrics_{}_all.yaml".format(epoch)

    evalpath = os.path.join(folder, metricname)
    print(f"Saving evaluation: {evalpath}")
    save_metrics(evalpath, metrics)
