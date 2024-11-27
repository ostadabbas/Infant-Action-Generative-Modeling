import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.train.trainer import train, test
from src.utils.tensors import collate
from src.utils.get_model_and_data import get_model_and_data
from src.datasets.get_dataset import get_datasets, get_testset, get_synset
from src.parser.recognition import training_parser
import src.utils.fixseed




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

from src.datasets.get_dataset import get_datasets, get_testset, get_synset
from src.recognition.get_model import get_model as get_rec_model
import matplotlib.pyplot as plt
from src.visualize.visualize import generate_by_ori_video_sequences
import os
from collections import defaultdict
import numpy as np
from sklearn.manifold import TSNE

def visualize_tsne(test_data, real_data, syn_data, filename, tsne_perplexity=30, tsne_learning_rate=200):
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
        for s in samples:
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




def main():    
    # parse options
    parameters = training_parser()

    model, datasets = get_model_and_data(parameters)

    ######################
    datapath = './train_action_recognition_100epochs_infact388_syn500'
    checkpoint_name = 'checkpoint_0100.pth.tar'
    filename = os.path.join(datapath, 'best_feat_vis.png') 

    modelpath = os.path.join(datapath, checkpoint_name)
    state_dict = torch.load(modelpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])

    youtube_dataset = datasets["train"]
    youtube_iterator = DataLoader(youtube_dataset, batch_size=parameters["batch_size"],
                                shuffle=False, num_workers=1, collate_fn=collate)

    career_dataset = datasets["test"]
    career_iterator = DataLoader(career_dataset, batch_size=parameters["batch_size"],
                                shuffle=False, num_workers=1, collate_fn=collate)

    syn_dataset = get_synset(parameters)
    syn_iterator = DataLoader(syn_dataset, batch_size=parameters["batch_size"],
                                shuffle=False, num_workers=1, collate_fn=collate)

    career_dict_loss, career_outputs = test(model, optimizer, career_iterator, model.device, 0, parameters)
    youtube_dict_loss, youtube_outputs = test(model, optimizer, youtube_iterator, model.device, 0, parameters)
    syn_dict_loss, syn_outputs = test(model, optimizer, syn_iterator, model.device, 0, parameters)

    for key in youtube_dict_loss.keys():
        youtube_dict_loss[key] /= len(youtube_iterator)
        career_dict_loss[key] /= len(career_iterator)
        syn_dict_loss[key] /= len(syn_iterator)
        
    '''
    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])

    model, datasets = get_model_and_data(parameters)

    gcn_model = get_rec_model(recogparameters) 
    
    bs = parameters["batch_size"]
    device = parameters["device"]
    ###########

    datapath = './train_action_recognition_100epochs_infact388_syn500'
    checkpoint_name = 'checkpoint_0100.pth.tar'

    filename = os.path.join(datapath, 'best_feat_vis.png')   


    # Youtube, Career, Synthetic data loading
    dataset_youtube = {key: [datasets['train']]
                     for key in ["train"]}

    dataset_career = {key: [get_testset(parameters)]
                     for key in ["test"]}

    dataset_syn = {key: [get_synset(parameters)]
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

    syn_dataiterator = {key: [DataLoader(data, batch_size=bs,
                                      shuffle=False, num_workers=1,
                                      collate_fn=collate)
                          for data in dataset_syn[key]]
                     for key in ["test"]}
            
            
            
    youtubeLoaders = {key: NewDataloader("gt", model, parameters,
                                       youtube_dataiterator[key][0],
                                       device)
                     for key in ["train"]}
            

    careerLoaders = {key: NewDataloader("gt", model, parameters,
                                       career_dataiterator[key][0],
                                       device)
                     for key in ["test"]}

    synLoaders = {key: NewDataloader("gt", model, parameters,
                                       syn_dataiterator[key][0],
                                       device)
                     for key in ["test"]}

    modelpath = os.path.join(datapath, checkpoint_name)
    state_dict = torch.load(modelpath, map_location=recogparameters["device"])
    gcn_model.load_state_dict(state_dict)
    gcn_model.eval()
    gcn_optimizer = torch.optim.AdamW(gcn_model.parameters(), lr=recogparameters["lr"])

    career_dict_loss, career_outputs = test(gcn_model, gcn_optimizer, careerLoaders['test'], gcn_model.device, 0, parameters)
    youtube_dict_loss, youtube_outputs = test(gcn_model, gcn_optimizer, youtubeLoaders['train'], gcn_model.device, 0, parameters)
    syn_dict_loss, syn_outputs = test(gcn_model, gcn_optimizer, synLoaders['test'], gcn_model.device, 0, parameters)

    for key in youtube_dict_loss.keys():
        youtube_dict_loss[key] /= len(youtube_dataiterator['train'][0])
        career_dict_loss[key] /= len(career_dataiterator['test'][0])
        syn_dict_loss[key] /= len(syn_dataiterator['test'][0])

    '''

    calculate_per_class_accuracy(career_outputs, parameters["num_classes"])
    print(f"Recognition Model, youtube_losses: {youtube_dict_loss}, career_losses: {career_dict_loss}, syn_losses: {syn_dict_loss}")

    syn_data = {'y':[], 'feat':[]}
    gt_data = {'y':[], 'feat':[]}
    test_data = {'y':[], 'feat':[]}

    for gt_output in youtube_outputs:
        gt_data['y'].extend(gt_output['y'].numpy())  
        if gt_output['features'].numpy().shape[0] == 256:
            gt_data['feat'].extend(gt_output['features'].numpy()[np.newaxis, :])
        else:
            gt_data['feat'].extend(gt_output['features'].numpy())

    for test_output in career_outputs:
        test_data['y'].extend(test_output['y'].numpy())  
        if test_output['features'].numpy().shape[0] == 256:
            test_data['feat'].extend(test_output['features'].numpy()[np.newaxis, :])
        else:
            test_data['feat'].extend(test_output['features'].numpy())

    for syn_output in syn_outputs:
        syn_data['y'].extend(syn_output['y'].numpy())  
        if syn_output['features'].numpy().shape[0] == 256:
            syn_data['feat'].extend(syn_output['features'].numpy()[np.newaxis, :])
        else:
            syn_data['feat'].extend(syn_output['features'].numpy())

    syn_per_class = {class_label: [] for class_label in range(parameters["num_classes"])}
    for i, (sample, label) in enumerate(zip(syn_data['feat'], syn_data['y'])):
        syn_per_class[label].append(sample)

    gt_per_class = {class_label: [] for class_label in range(parameters["num_classes"])}
    for i, (sample, label) in enumerate(zip(gt_data['feat'], gt_data['y'])):
        gt_per_class[label].append(sample)

    test_per_class = {class_label: [] for class_label in range(parameters["num_classes"])}
    for i, (sample, label) in enumerate(zip(test_data['feat'], test_data['y'])):
        test_per_class[label].append(sample)
                  
                      
    print('Visualize t-SNE')
    visualize_tsne(test_per_class, gt_per_class, syn_per_class, filename, tsne_perplexity=30, tsne_learning_rate=200)


if __name__ == '__main__':
    main()


