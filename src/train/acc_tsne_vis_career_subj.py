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
import pickle as pkl

def visualize_tsne(test_data, filename, tsne_perplexity=30, tsne_learning_rate=200):
    data_features = []
    data_labels = []
    data_sources = []

    for c, samples in test_data.items():
        for s, id in samples:
            data_features.append(s)
            data_labels.append(c)
            data_sources.append(id)

    # Define markers and colors
    markers = ['.', '+', '^', 'x', '*', 's']  # Extend as needed
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
            if source == 'D01':
                marker = markers[0]
            elif source == 'D02':
                marker = markers[1]
            elif source == 'D03':
                marker = markers[2]
            elif source == 'D05':
                marker = markers[3]
            elif source == 'D08':
                marker = markers[4]
            elif source == 'D09':
                marker = markers[5]

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



def main():    
    # parse options
    parameters = training_parser()

    model, datasets = get_model_and_data(parameters)

    ######################
    datapath = './train_recognition_real_syn'
    checkpoint_name = 'checkpoint_best_0005.pth.tar'
    filename = os.path.join(datapath, 'real_syn_best_feat_vis.png') 

    modelpath = os.path.join(datapath, checkpoint_name)
    state_dict = torch.load(modelpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])

    career_dataset = datasets["test"]
    career_iterator = DataLoader(career_dataset, batch_size=parameters["batch_size"],
                                shuffle=False, num_workers=1, collate_fn=collate)
    career_dict_loss, career_outputs = test(model, optimizer, career_iterator, model.device, 0, parameters)
  
    test_data = {'y':[], 'feat':[], 'subj': []}
    for test_output in career_outputs:
        test_data['y'].extend(test_output['y'].numpy())  
        if test_output['features'].numpy().shape[0] == 256:
            test_data['feat'].extend(test_output['features'].numpy()[np.newaxis, :])
        else:
            test_data['feat'].extend(test_output['features'].numpy())

    pkldatafilepath = "./data/Career/Career_posture.pkl"
    ori_data = pkl.load(open(pkldatafilepath, "rb"))
    all_labels = []
    all_subj = []

    for sample in ori_data:
        if sample['pos_label'] == 'Supine':
            all_labels.append(0)
            all_subj.append(sample['frame_dir'][:3])
        elif sample['pos_label'] == 'Prone':
            all_labels.append(1)
            all_subj.append(sample['frame_dir'][:3])
        elif sample['pos_label'] == 'Sitting':
            all_labels.append(2)
            all_subj.append(sample['frame_dir'][:3])
        elif sample['pos_label'] == 'Standing':
            all_labels.append(3)
            all_subj.append(sample['frame_dir'][:3])
        elif sample['pos_label'] == 'All-fours':
            all_labels.append(4)
            all_subj.append(sample['frame_dir'][:3])
        else:
            continue
    #print(all_labels)      
    test_data['subj'] = all_subj  


    test_per_class = {class_label: [] for class_label in range(parameters["num_classes"])}
    for i, (sample, label, id) in enumerate(zip(test_data['feat'], test_data['y'], test_data['subj'])):
        test_per_class[label].append((sample, id))
    #print(test_data['y'])
    

               
    print('Visualize t-SNE')
    visualize_tsne(test_per_class, filename, tsne_perplexity=30, tsne_learning_rate=200)


if __name__ == '__main__':
    main()


