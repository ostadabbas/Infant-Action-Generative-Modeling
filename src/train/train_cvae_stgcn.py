import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.train.trainer import train, test
from src.utils.tensors import collate
import src.utils.fixseed  # noqa

from src.parser.training import parser
from src.utils.get_model_and_data import get_model_and_data
##
from src.recognition.get_model import get_model as get_rec_model
from src.parser.recognition import training_parser
from src.evaluate.stgcn_eval import *
from torch.utils.data import DataLoader
from copy import copy
import src.utils.rotation_conversions as geometry
from src.datasets.get_dataset import get_datasets, get_testset
from collections import Counter
from heapq import nlargest
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from src.visualize.visualize import generate_by_ori_video_sequences
import os
from collections import defaultdict
import numpy as np
from sklearn.manifold import TSNE


# Start and endpoints of our representation H36M 17 joints
I = np.array([0,1,2,8,14,15,0,4,5,8,11,12,0,7,8,9])
J = np.array([1,2,3,14,15,16,4,5,6,11,12,13,7,8,9,10])
# Left / right indicator L in red "1", R in blue "0"
LR = np.array([1,1,1,0,0,0,0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=bool)

lcolor="#3498db"
rcolor="#e74c3c"

def vis_3d_skel(kpts, label):
    ax = plt.axes(projection='3d')

    xdata = kpts[:,0]
    ydata = kpts[:,1]
    zdata = kpts[:,2]
    ax.scatter3D(xdata, ydata, zdata)

    for k in range(kpts.shape[0]):
        ax.text(kpts[k,0], kpts[k,1], kpts[k,2], k, color='red')

    for i in np.arange(len(I)):
      x = np.array( [kpts[I[i], 0], kpts[J[i], 0]] )
      y = np.array( [kpts[I[i], 1], kpts[J[i], 1]] )
      z = np.array( [kpts[I[i], 2], kpts[J[i], 2]] )
      ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f'Class Label: {label}')
    
    r = 10;
    xroot, yroot, zroot = kpts[0,0], kpts[0,1], kpts[0,2]
    #ax.set_xlim3d([-r+xroot, r+xroot])
    #ax.set_zlim3d([-r+zroot, r+zroot])
    #ax.set_ylim3d([-r+yroot, r+yroot])
    print('show')
    plt.show()

def save_data_to_npy(data_dict, file_path):
    if os.path.exists(file_path):
        existing_data = np.load(file_path, allow_pickle=True).item()
        for key, val in data_dict.items():
            if key in existing_data:
                existing_data[key].extend(val)
            else:
                existing_data[key] = val
        np.save(file_path, existing_data)
    else:
        np.save(file_path, data_dict)


def calculate_within_class_distance(new_data, gt_data):
    within_class_distances = {}

    for class_label, new_samples in new_data.items():
        distances = []
        
        if class_label in gt_data:
            gt_samples = gt_data[class_label]
            
            for idx, new_sample in new_samples:
                distance = 0
                for gt_sample in gt_samples:
                    distance += np.linalg.norm(new_sample - gt_sample)
                distances.append((idx, distance / len(gt_samples)))
        
        if distances:
            within_class_distances[class_label] = distances
        else:
            within_class_distances[class_label] = None

    return within_class_distances

def calculate_between_class_distances(new_data, gt_data):
    between_class_distances = {}

    for class_label, new_samples in new_data.items():
        distances = []
        
        for idx, new_sample in new_samples:
            distance = 0
            num = 0
            for other_class_label, gt_samples in gt_data.items():
                if class_label != other_class_label:
                    num += len(gt_samples)
                    for gt_sample in gt_samples:
                        distance += np.linalg.norm(new_sample - gt_sample)
            distances.append((idx, distance / num))
        
        if distances:
            between_class_distances[class_label] = distances
        else:
            between_class_distances[class_label] = None

    return between_class_distances

def normalize_class_distances(distances_dict):
    normalized_dict = {}

    for class_label, distances_list in distances_dict.items():
        distances = [distance for _, distance in distances_list]

        min_distance = min(distances)
        max_distance = max(distances)

        normalized_distances = [
            (index, (distance - min_distance) / (max_distance - min_distance) if max_distance > min_distance else 0)
            for index, distance in distances_list
        ]
        #print(normalized_distances)
        normalized_dict[class_label] = normalized_distances

    return normalized_dict

def calculate_and_sort_ratios(distances_within_class, distances_to_other_classes_avg, weight_1=1, weight_2=1):
    sorted_ratios_per_class = {}

    for class_label in distances_within_class.keys():
        within_class_distances = distances_within_class[class_label]
        to_other_classes_distances = distances_to_other_classes_avg[class_label]

        ratios = [(index, (weight_1 * within_distance) / (weight_2 * to_other_distance))
                  for (index, within_distance), (_, to_other_distance) in zip(within_class_distances, to_other_classes_distances)]
        
        sorted_ratios = sorted(ratios, key=lambda x: x[1])

        sorted_ratios_per_class[class_label] = sorted_ratios
    #(sorted_ratios_per_class)
    return sorted_ratios_per_class

def plot_tsne(features, labels):
    # Apply t-SNE to reduce to 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(features)

    # Plot
    plt.figure(figsize=(10, 7))
    for i in np.unique(labels):
        plt.scatter(X_tsne[labels== i, 0], X_tsne[labels == i, 1], label=f'Class {i}')
    plt.title('t-SNE of features')
    plt.xlabel('t-SNE feature 1')
    plt.ylabel('t-SNE feature 2')
    plt.legend()
    plt.show()


def visualize_tsne(test_data, real_data, syn_data, selected_indices, filename, num_classes, tsne_perplexity=30, tsne_learning_rate=200):
    data_features = []
    data_labels = []
    data_sources = []
    for c, samples in real_data.items():
        for s in samples:
            data_features.append(s)
            data_labels.append(c)
            data_sources.append('train_real')

    for c, samples in test_data.items():
        for s in samples:
            data_features.append(s)
            data_labels.append(c)
            data_sources.append('test_set')

    for c, samples in syn_data.items():
        for j, (i, s) in enumerate(samples):
            data_features.append(s)
            data_labels.append(c)
            if i in selected_indices:
                data_sources.append('selected')  
            else: 
                data_sources.append('synthetic') 

    # Define markers and colors
    markers = ['o', '+', '^', 'x', '*']  # Extend as needed
    colors = ['red', 'blue', 'green', 'black', 'cyan'] 
    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=tsne_perplexity, learning_rate=tsne_learning_rate, random_state=42)
    tsne_results = tsne.fit_transform(data_features)
    
    # Plot
    plt.figure(figsize=(12, 9))
    unique_labels = sorted(set(data_labels))
    for i, label in enumerate(unique_labels):
        for source in set(data_sources):
            indices = [j for j, (l, s) in enumerate(zip(data_labels, data_sources)) if l == label and s == source]
            if source == 'train_real':
                source = 'train real'
                marker = markers[2]
            elif source == 'synthetic':
                source = 'synthetic'
                marker = markers[0]
            elif source == 'selected':
                source = 'synthetic'
                marker = markers[0]
            elif source == 'test_set':
                source = 'test set'
                marker = markers[4]
            color = colors[i % len(colors)]
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                        label=f'{label} ({source})', 
                        c=color, 
                        marker=marker)
    
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=num_classes)
    plt.tight_layout()
    plt.title('Real and Synthetic Data Features t-SNE Visualization')
    plt.savefig(filename, dpi=300)
    #plt.show()

def calculate_per_class_accuracy(batches, num_classes):
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for batch in batches:
        _, yhat = torch.max(F.softmax(batch['yhat'], dim=1), dim=1)
        y = batch['y']

        for c in range(num_classes):
            class_mask = (y == c)
            
            correct_counts =  sum((yhat[class_mask] == y[class_mask]))
            total_counts = y[class_mask].shape[0]

            class_correct[c] += correct_counts
            class_total[c] += total_counts

    class_accuracy = {class_id: correct / class_total[class_id] for class_id, correct in class_correct.items()}

    for class_id, accuracy in class_accuracy.items():
        print(f"Class {class_id} accuracy: {accuracy:.2f}")

def do_epochs(model, recog_model, datasets, parameters, optimizer, gcn_optimizer, writer, gcn_train, recogparameters, syn_path):
    train_dataset = datasets["train"]
    
    train_iterator = DataLoader(train_dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=1, collate_fn=collate)
    
    max_acc = 0.0
    logpath = os.path.join(parameters["folder"], "training.log")


    recog_epoch = 0
    with open(logpath, "w") as logfile:
        for epoch in range(1, parameters["num_epochs"]+1):
            
            dict_loss, _ = train(model, optimizer, train_iterator, model.device, epoch, parameters)
            
            for key in dict_loss.keys():
                dict_loss[key] /= len(train_iterator)
                writer.add_scalar(f"Gen_train/{key}", dict_loss[key], epoch)

            epochlog = f"Epoch {epoch}, train losses: {dict_loss}"
            print(epochlog)
            print(epochlog, file=logfile)

            if ((epoch % parameters["snapshot"]) == 0) or (epoch == parameters["num_epochs"]):
                checkpoint_path = os.path.join(parameters["folder"],
                                               'checkpoint_{:04d}.pth.tar'.format(epoch))
                print('Saving checkpoint {}'.format(checkpoint_path))
                torch.save(model.state_dict(), checkpoint_path)
            
            #train STGCN and get CLA
            stgcn_metrics = {}

            bs = parameters["batch_size"]
            device = parameters["device"]
            #dataname = parameters["dataset"]
            
            datasetGT = {key: [train_dataset]
                             for key in ["train"]}

            datasetgen = {key: [get_datasets(parameters)[key]]
                             for key in ["test"]}

            datasetTest = {key: [get_testset(parameters)]
                             for key in ["test"]}
            
            gt_dataiterator = {key: [DataLoader(data, batch_size=bs,
                                             shuffle=True, num_workers=1,
                                             collate_fn=collate)
                                  for data in datasetGT[key]]
                            for key in ["train"]}

            gen_dataiterator = {key: [DataLoader(data, batch_size=bs,
                                             shuffle=False, num_workers=1,
                                             collate_fn=collate)
                                  for data in datasetgen[key]]
                            for key in ["test"]}

            test_dataiterator = {key: [DataLoader(data, batch_size=bs,
                                             shuffle=False, num_workers=1,
                                             collate_fn=collate)
                                  for data in datasetTest[key]]
                            for key in ["test"]}
            
            
            gtLoaders = {key: NewDataloader("gt", model, parameters,
                                            gt_dataiterator[key][0],
                                            device)
                         for key in ["train"]}
            
            
            genLoaders = {key: NewDataloader("gen_action", model, parameters,
                                             gen_dataiterator[key][0],
                                             device)
                         for key in ["test"]}

            testLoaders = {key: NewDataloader("gt", model, parameters,
                                            test_dataiterator[key][0],
                                            device)
                         for key in ["test"]}

                       
            
            if gcn_train:
                for sub_epoch in range(1, 4):
                    recog_epoch += 1
                    train_dict_loss, rec_train_output = train(recog_model, gcn_optimizer, gtLoaders['train'], recog_model.device, recog_epoch, parameters)
           
                    test_dict_loss, gen_test_output = test(recog_model, gcn_optimizer, genLoaders['test'], recog_model.device, recog_epoch, parameters)

                    test_set_loss, rec_test_output = test(recog_model, gcn_optimizer, testLoaders['test'], recog_model.device, recog_epoch, parameters)
        
                    for key in train_dict_loss.keys():
                        train_dict_loss[key] /= len(gt_dataiterator['train'][0])
                
                        test_dict_loss[key] /= len(gen_dataiterator['test'][0])
                        test_set_loss[key] /= len(test_dataiterator['test'][0])
                        writer.add_scalar(f"Recog_train/{key}", train_dict_loss[key], recog_epoch)
                        writer.add_scalar(f"Recog_test/{key}", test_dict_loss[key], recog_epoch)

                
                    print(f"Recognition Epoch {recog_epoch}, train losses: {train_dict_loss}, gen_test_losses: {test_dict_loss}, test_set_losses: {test_set_loss}")
                gcn_train = False  
                recog_checkpoint_path = os.path.join(parameters["folder"],
                                               'recog_checkpoint_{:04d}.pth.tar'.format(recog_epoch))
                print('Saving checkpoint {}'.format(recog_checkpoint_path))
                torch.save(recog_model.state_dict(), recog_checkpoint_path)               
       
            else:     
                _, rec_train_output = test(recog_model, gcn_optimizer, gtLoaders['train'], recog_model.device, recog_epoch, parameters)

                test_dict_loss, gen_test_output = test(recog_model, gcn_optimizer, genLoaders['test'], recog_model.device, recog_epoch, parameters)

                test_set_loss, rec_test_output = test(recog_model, gcn_optimizer, testLoaders['test'], recog_model.device, recog_epoch, parameters)
                for key in test_dict_loss.keys():
                    test_dict_loss[key] /= len(gen_dataiterator['test'][0])
                    test_set_loss[key] /= len(test_dataiterator['test'][0])

                    writer.add_scalar(f"Recog_test/{key}", test_dict_loss[key], recog_epoch)

                calculate_per_class_accuracy(gen_test_output, recogparameters["num_classes"])
                print(f"Recognition Epoch {recog_epoch}, gen_test_losses: {test_dict_loss}, test_set_losses: {test_set_loss}")


                
           
            
            acc = test_dict_loss['accuracy']
            
            
            if acc > max_acc:
                print('Update Max Acc.')
                max_acc = acc
                outputs = copy(gen_test_output)
                gt_outputs = copy(rec_train_output)
                test_outputs = copy(rec_test_output)
                #print(outputs[0]['features'].size())

            if epoch % 10 == 0:
                new_data = {'pose':[], 'keypoint':[], 'y':[], 'score':[], 'epoch_idx':[], 'feat':[]}
                gt_data = {'y':[], 'feat':[]}
                test_data = {'y':[], 'feat':[]}

                for gt_output in gt_outputs:
                    gt_data['y'].extend(gt_output['y'].numpy())  
                    if gt_output['features'].numpy().shape[0] == 256:
                        gt_data['feat'].extend(gt_output['features'].numpy()[np.newaxis, :])
                    else:
                        gt_data['feat'].extend(gt_output['features'].numpy())

                for test_output in test_outputs:
                    test_data['y'].extend(test_output['y'].numpy())  
                    if test_output['features'].numpy().shape[0] == 256:
                        test_data['feat'].extend(test_output['features'].numpy()[np.newaxis, :])
                    else:
                        test_data['feat'].extend(test_output['features'].numpy())

                for output in outputs:
                    confidence_scores = F.softmax(output['yhat'], dim=1)
                    max_confidences, pred_y = torch.max(confidence_scores, dim=1)

                    #mask = (pred_y == output['y'] and max_confidences >= 0.85)
                    correct_predictions = pred_y == output['y']
                    high_confidence = max_confidences >= 0.75
                    mask = correct_predictions & high_confidence
                    #print(output.keys())
                    correct_scores = max_confidences[mask]
                    correct_labels = output['y'][mask]
                    correct_output = output['output_xyz'][mask].permute(0,3,1,2)
                    correct_poses = output['output_pose'][mask].permute(0,3,1,2)
                    if output['features'].numpy().shape[0] == 256:
                        correct_feats = output['features'].numpy()[np.newaxis, :][torch.where(mask)[0], :]
                    else:
                        correct_feats = output['features'].numpy()[mask]

                    #print(correct_labels)
                    #print(correct_output.size())

                    new_data['score'].extend(correct_scores.numpy())
                    new_data['keypoint'].extend(correct_output.numpy())
                    new_data['pose'].extend(correct_poses.numpy())
                    new_data['y'].extend(correct_labels.numpy())  
                    new_data['epoch_idx'].extend([epoch] * len(mask))
                    new_data['feat'].extend(correct_feats)#.numpy())

                # check balance of new data samples
                class_counts = Counter(new_data['y'])
                print(class_counts)
                min_class_count = min(class_counts.values())
                if min_class_count <= 1 or len(class_counts.keys()) < recogparameters["num_classes"]:
                    continue 

                # feature visualization
                #plot_tsne(new_data['feat'], new_data['y'], epoch)

                # calculate and sort within-class and cross_class distance for each sample
                data_per_class = {class_label: [] for class_label in range(recogparameters["num_classes"])}
                for i, (sample, label) in enumerate(zip(new_data['feat'], new_data['y'])):
                    data_per_class[label].append((i, sample))

                gt_per_class = {class_label: [] for class_label in range(recogparameters["num_classes"])}
                for i, (sample, label) in enumerate(zip(gt_data['feat'], gt_data['y'])):
                    gt_per_class[label].append(sample)

                test_per_class = {class_label: [] for class_label in range(recogparameters["num_classes"])}
                for i, (sample, label) in enumerate(zip(test_data['feat'], test_data['y'])):
                    test_per_class[label].append(sample)
                  

                distances_within_class = calculate_within_class_distance(data_per_class, gt_per_class)
                distances_to_other_classes_avg = calculate_between_class_distances(data_per_class, gt_per_class)

                normalized_distances_within_class = normalize_class_distances(distances_within_class)
                normalized_distances_to_other_classes_avg = normalize_class_distances(distances_to_other_classes_avg)

                numerator_w = 0.6
                denominator_w = 0.4
                sorted_ratios_per_class = calculate_and_sort_ratios(normalized_distances_within_class, normalized_distances_to_other_classes_avg, numerator_w, denominator_w)
                #print(sorted_vectors_per_class)

                selection_threshold = 0.5
                if min_class_count == 1:
                    num_selection = 1
                else:
                    num_selection = int(selection_threshold * min_class_count)   
                
                random_indices_mask = []
                top_indices_mask = []
                for class_label, ratios in sorted_ratios_per_class.items():
                    random_indices_class = np.random.choice(len(ratios), size=num_selection, replace=False)
                    selected_elements = [ratios[index] for index in random_indices_class]
                    random_indices_mask.extend([ratio_idx for ratio_idx, _ in selected_elements])                  
                    
                    num_samples = len(ratios)
                    middle_index = num_samples // 2
                    half_selection = num_selection // 2
                    start_index = middle_index - half_selection
                    end_index = start_index + num_selection

                    top_indices_mask.extend([index for index, _ in ratios[start_index:end_index]])
                
                new_data['score'] = [new_data['score'][i] for i in top_indices_mask]
                new_data['keypoint'] = [new_data['keypoint'][i] for i in top_indices_mask]
                new_data['pose'] = [new_data['pose'][i] for i in top_indices_mask]
                new_data['y'] = [new_data['y'][i] for i in top_indices_mask]
                new_data['epoch_idx'] = [new_data['epoch_idx'][i] for i in top_indices_mask]
                new_data['feat'] = [new_data['feat'][i] for i in top_indices_mask]
                '''
                new_data['score'] = [new_data['score'][i] for i in random_indices_mask]
                new_data['keypoint'] = [new_data['keypoint'][i] for i in random_indices_mask]
                new_data['pose'] = [new_data['pose'][i] for i in random_indices_mask]
                new_data['y'] = [new_data['y'][i] for i in random_indices_mask]
                new_data['epoch_idx'] = [new_data['epoch_idx'][i] for i in random_indices_mask]
                new_data['feat'] = [new_data['feat'][i] for i in random_indices_mask]
                '''
                #print(new_data['feat'][0].size)


                #for j in range(len(new_data['y'])):
                #    vis_3d_skel(new_data['keypoint'][j][50, :, :], new_data['y'][j])
                
               
                print('Visualize t-SNE')
                filename = os.path.join(parameters["folder"], str(epoch) + "_feat_vis.png")

                visualize_tsne(test_per_class, gt_per_class, data_per_class, top_indices_mask, filename, recogparameters["num_classes"], tsne_perplexity=30, tsne_learning_rate=200)
                #visualize_tsne(test_per_class, gt_per_class, data_per_class, random_indices_mask, filename, recogparameters["num_classes"], tsne_perplexity=30, tsne_learning_rate=200)
                
                

                print('Add new data!')
                train_dataset.add_samples(new_data)

                train_iterator = DataLoader(train_dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=1, collate_fn=collate)
            
                print(train_dataset.__len__())
                gcn_train = True

                save_data_to_npy(new_data, syn_path)
                print('Saved new data!')

                max_acc = 0.0
            
            
            writer.flush()


if __name__ == '__main__':
    # parse options
    parameters = parser()

    recogparameters = parameters.copy()
    pose_rep = recogparameters['pose_rep']
    if pose_rep == "rotvec":
        recogparameters['nfeats'] = 3
        recogparameters["njoints"] = 24
    elif pose_rep == "rotmat":
        recogparameters['nfeats'] = 3
        recogparameters["njoints"] = 24
    elif pose_rep == "rotquat":
        recogparameters['nfeats'] = 4
        recogparameters["njoints"] = 24
    elif pose_rep == "rot6d":
        recogparameters['nfeats'] = 6
        recogparameters["njoints"] = 24
    elif pose_rep == "xyz":
        recogparameters['nfeats'] = 3
        recogparameters["njoints"] = 17
        

    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])

    model, datasets = get_model_and_data(parameters)
    
    modelpath = "./exps/infacttrans_rc_rcxyz_velxyz_epoch1300/checkpoint_1100.pth.tar"
    state_dict = torch.load(modelpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)
    
    gcn_model = get_rec_model(recogparameters) 
    train_gcn = False
    
    if not train_gcn:
        modelpath = "./pretrained_recognition_15epochs/checkpoint_best_0010.pth.tar"

        state_dict = torch.load(modelpath, map_location=recogparameters["device"])
        gcn_model.load_state_dict(state_dict)
        gcn_model.eval()

    else:
        gcn_model.train()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])
    gcn_optimizer = torch.optim.AdamW(gcn_model.parameters(), lr=recogparameters["lr"])

    generation_path = parameters['folder'] + '/generated_samples.npy'

    print('Total params of generative network: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print('Total params of action recognition network: %.2fM' % (sum(p.numel() for p in gcn_model.parameters()) / 1000000.0))
    print("Training model..")
    do_epochs(model, gcn_model, datasets, parameters, optimizer, gcn_optimizer, writer, train_gcn, recogparameters, generation_path)

    writer.close()
