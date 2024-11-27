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
    class_accuracy = {}
    samples_total = 0
    correct_total = 0
    for class_id, correct in class_correct.items():
        if class_total[class_id] == 0:
            class_accuracy[class_id] = [-1, 0] 
        else:
            class_accuracy[class_id] = [(correct * 1.0) / (class_total[class_id] * 1.0), correct]
            samples_total += class_total[class_id]
            correct_total += correct

    for class_id, accuracy in class_accuracy.items():
        print(f"Class {class_id} accuracy: {accuracy[0]:.4f} ({accuracy[1]:.4f})")

    accuracy_total = (correct_total * 1.0) / (samples_total * 1.0)
    print(f"Total accuracy ({correct_total:.4f}/{samples_total:.4f}) : {accuracy_total:.4f}")

def do_epochs(model, datasets, parameters, optimizer, writer):
    epoch = 0
    train_dataset = datasets["train"]
    train_iterator = DataLoader(train_dataset, batch_size=parameters["batch_size"],
                                shuffle=True, num_workers=8, collate_fn=collate)
    

    if "test" in datasets:
        test_dataset = datasets["test"]
        test_iterator = DataLoader(test_dataset, batch_size=parameters["batch_size"],
                                   shuffle=False, num_workers=8, collate_fn=collate)

        test_dict_loss, test_output = test(model, optimizer, test_iterator, model.device, epoch, parameters)

        for key in test_dict_loss.keys():
            test_dict_loss[key] /= len(test_iterator)
            writer.add_scalar(f"TestLoss/{key}", test_dict_loss[key], epoch)

        calculate_per_class_accuracy(test_output, parameters["num_classes"])
        print(f"Recognition Epoch {epoch}, test_losses: {test_dict_loss}")


        writer.flush()





def main():    
    # parse options
    parameters = training_parser()

    # logging tensorboard
    writer = SummaryWriter(log_dir=parameters["folder"])

    model, datasets = get_model_and_data(parameters)

    #datasets.pop("test")
    
    modelpath = "train_recognition_real_syn/checkpoint_best_0030.pth.tar" 
    #modelpath = "./pretrained_recognition_15epochs/checkpoint_best_0013.pth.tar" #'./recog_models_200/recog_models/recog_checkpoint_0060.pth.tar' 

    state_dict = torch.load(modelpath, map_location=parameters["device"])
    model.load_state_dict(state_dict)
  

    if parameters["dataset"] == "uestcpartial":
        dt = datasets["train"]
        normal_length = dt._oldN
        realratio = parameters["realratio"]
        withgen = parameters["withgen"]
        withreal = parameters["withreal"]
        print(f"Real ratio: {realratio}, withgen: {withgen==1}, withreal: {withreal==1}")
        expected = normal_length*realratio/100 * (2 if (withgen == 1) and (withreal == 1) else 1)
        print(f"Normal len: {len(dt)}, expected: {expected}")

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=parameters["lr"])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print("Training model..")
    do_epochs(model, datasets, parameters, optimizer, writer)

    writer.close()


if __name__ == '__main__':
    main()