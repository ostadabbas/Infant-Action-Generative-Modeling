import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.train.trainer import train, test
from src.utils.tensors import collate
from src.utils.get_model_and_data import get_model_and_data

from src.parser.recognition import training_parser
import src.utils.fixseed  # noqa
from collections import defaultdict
import numpy as np
import torch.nn.functional as F

from src.evaluate.stgcn_eval import *
from torch.utils.data import DataLoader

from src.datasets.get_dataset import get_datasets, get_testset
from src.recognition.get_model import get_model as get_rec_model
import matplotlib.pyplot as plt
from src.visualize.visualize import generate_by_ori_video_sequences
import os
from collections import defaultdict
import numpy as np
from sklearn.manifold import TSNE

def visualize_tsne(test_data, real_data, syn_data, idx, filename, tsne_perplexity=30, tsne_learning_rate=200):
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
            data_sources.append('test_real')

    for c, samples in syn_data.items():
        for j, (i, s) in enumerate(samples):
            data_features.append(s)
            data_labels.append(c)
            data_sources.append('syn') 

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
        if label == 0:
            new_label = 'Supine'
        elif label == 1:
            new_label = 'Prone'
        elif label == 2:
            new_label = 'Sitting'
        elif label == 3:
            new_label = 'Standing'
        elif label == 4:
            new_label = 'All-fours'
      
        for source in set(data_sources):
            indices = [j for j, (l, s) in enumerate(zip(data_labels, data_sources)) if l == label and s == source]
            if source == 'train_real':
                source = 'youtube'
                marker = markers[0]
            elif source == 'syn':
                source = 'synthetic'
                marker = markers[1]
            elif source == 'selected':
                source = 'synthetic'
                marker = markers[2]
            elif source == 'test_real':
                source = 'career'
                marker = markers[4]
            color = colors[i % len(colors)]
            plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                        label=f'{new_label} ({source})', 
                        c=color, 
                        marker=marker)
    
    #plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=True, ncol=5)
    plt.tight_layout()
    plt.title('Real and Synthetic Data Features t-SNE Visualization at Epoch ' + str(idx))
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




def main():    
    # parse options
    parameters = training_parser()

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

    gcn_model = get_rec_model(recogparameters) 
    
    bs = parameters["batch_size"]
    device = parameters["device"]
    ###########
    datapath = './exps/infactposture_rc_rcxyz_velxyz_init200_epoch200_new'

    save_path = os.path.join(datapath, 't_sne_figs')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # synthetic data loading
    syn_path = os.path.join(datapath, "generated_samples.npy")
    syn_data = np.load(syn_path, allow_pickle=True).item()
    print(syn_data.keys())
    syn_feat = syn_data['feat']
    syn_label = syn_data['y']
    syn_idx = syn_data['epoch_idx']
    print(syn_idx)
    idx_dict = {}
    idx_list = []
    start_index = 0
    for i in range(len(syn_idx) - 1):
        if syn_idx[i] != syn_idx[i + 1]:
            end_index = i
            idx_dict[syn_idx[i]] = (start_index, end_index)
            idx_list.append(syn_idx[i])
            start_index = i + 1
    idx_dict[syn_idx[-1]] = (start_index, len(syn_idx) - 1)
    idx_list.append(syn_idx[-1])

    print(idx_dict)
    # Youtube data and Career data loading
    dataset_youtube = {key: [datasets['train']]
                     for key in ["train"]}

    dataset_career = {key: [get_testset(parameters)]
                     for key in ["test"]}
            
    youtube_dataiterator = {key: [DataLoader(data, batch_size=bs,
                                     shuffle=False, num_workers=1,
                                     collate_fn=collate)
                          for data in dataset_youtube[key]]
                     for key in ["train"]}

    career_dataiterator = {key: [DataLoader(data, batch_size=bs,
                                      shuffle=False, num_workers=1,
                                      collate_fn=collate)
                          for data in dataset_career[key]]
                     for key in ["test"]}
            
            
    youtubeLoaders = {key: NewDataloader("gt", model, parameters,
                                       youtube_dataiterator[key][0],
                                       device)
                     for key in ["train"]}
            

    careerLoaders = {key: NewDataloader("gt", model, parameters,
                                       career_dataiterator[key][0],
                                       device)
                     for key in ["test"]}
    for epoch_i in range(len(idx_list)):
        epoch = idx_list[epoch_i]
        start_i, end_i = idx_dict[epoch]
        recog_epoch = (epoch_i + 1) * 3
        formatted_string = "{:04d}".format(recog_epoch)
        ##############
        modelpath = os.path.join(datapath, 'recog_checkpoint_' + formatted_string + '.pth.tar')

        state_dict = torch.load(modelpath, map_location=recogparameters["device"])
        gcn_model.load_state_dict(state_dict)
        gcn_model.eval()
        gcn_optimizer = torch.optim.AdamW(gcn_model.parameters(), lr=recogparameters["lr"])


        youtube_dict_loss, youtube_output = test(gcn_model, gcn_optimizer, youtubeLoaders['train'], gcn_model.device, 0, parameters)

        career_dict_loss, career_output = test(gcn_model, gcn_optimizer, careerLoaders['test'], gcn_model.device, 0, parameters)

        for key in youtube_dict_loss.keys():
            youtube_dict_loss[key] /= len(youtube_dataiterator['train'][0])
            career_dict_loss[key] /= len(career_dataiterator['test'][0])

        calculate_per_class_accuracy(career_output, parameters["num_classes"])
        print(f"Recognition Epoch {recog_epoch}, youtube_losses: {youtube_dict_loss}, career_losses: {career_dict_loss}")

        new_data = {'y':[], 'epoch_idx':[], 'feat':[]}
        gt_data = {'y':[], 'feat':[]}
        test_data = {'y':[], 'feat':[]}

        for gt_output in youtube_output:
            gt_data['y'].extend(gt_output['y'].numpy())  
            if gt_output['features'].numpy().shape[0] == 256:
                gt_data['feat'].extend(gt_output['features'].numpy()[np.newaxis, :])
            else:
                gt_data['feat'].extend(gt_output['features'].numpy())

        for test_output in career_output:
            test_data['y'].extend(test_output['y'].numpy())  
            if test_output['features'].numpy().shape[0] == 256:
                test_data['feat'].extend(test_output['features'].numpy()[np.newaxis, :])
            else:
                test_data['feat'].extend(test_output['features'].numpy())

        new_data['y'] = syn_label[:end_i+1] 
        new_data['epoch_idx'] = syn_idx[:end_i+1]
        new_data['feat'] = syn_feat[:end_i+1]

        data_per_class = {class_label: [] for class_label in range(recogparameters["num_classes"])}
        for i, (sample, label) in enumerate(zip(new_data['feat'], new_data['y'])):
            data_per_class[label].append((i, sample))

        gt_per_class = {class_label: [] for class_label in range(recogparameters["num_classes"])}
        for i, (sample, label) in enumerate(zip(gt_data['feat'], gt_data['y'])):
            gt_per_class[label].append(sample)

        test_per_class = {class_label: [] for class_label in range(recogparameters["num_classes"])}
        for i, (sample, label) in enumerate(zip(test_data['feat'], test_data['y'])):
            test_per_class[label].append(sample)
                  
                      
        print('Visualize t-SNE')
        filename = os.path.join(save_path, str(epoch) + '_feat_vis.png')

        visualize_tsne(test_per_class, gt_per_class, data_per_class, epoch, filename, tsne_perplexity=30, tsne_learning_rate=200)


if __name__ == '__main__':
    main()


