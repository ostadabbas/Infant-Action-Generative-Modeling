def get_dataset(name="ntu13"):
    if name == "ntu13":
        from .ntu13 import NTU13
        return NTU13
    elif name == "uestc":
        from .uestc import UESTC
        return UESTC
    elif name == "humanact12":
        from .humanact12poses import HumanAct12Poses
        return HumanAct12Poses
    elif name == "infactaction":
        from .infactactions import InfActActions
        return InfActActions
    elif name == "infactsyn":
        from .infactsyn import InfActSyns
        return InfActSyns
    elif name == "careeraction":
        from .careeractions import CareerActions
        return CareerActions

    elif name == "infactposture":
        from .infactpostures import InfActPostures
        return InfActPostures
    elif name == "careerposture":
        from .careerpostures import CareerPostures
        return CareerPostures
    elif name == "mixposture":
        from .mixpostures import MixPostures
        return MixPostures
    elif name == "infacttest":
        from .infacttests import InfActTests
        return InfActTests
    elif name == "infactsyn":
        from .infactsyn import InfActSyns
        return InfActSyns


    elif name == "infacttrans":
        from .infacttrans import InfActTrans
        return InfActTrans
    elif name == "careertrans":
        from .careertrans import CareerTrans
        return CareerTrans
    elif name == "mixtrans":
        from .mixtrans import MixTrans
        return MixTrans
    elif name == "transtest":
        from .transtest import TransTests
        return TransTests
    elif name == "transsyn":
        from .transsyn import TransSyns
        return TransSyns



def get_datasets(parameters):
    name = parameters["dataset"]
    
    if name != "mixposture" and name != "mixtrans":
        DATA = get_dataset(name)
        dataset = DATA(split="train", **parameters)

        train = dataset

        # test: shallow copy (share the memory) but set the other indices
        from copy import copy
        test = copy(train)
        test.split = test
    else:
        DATA = get_dataset(name)
        dataset = DATA(split="train", **parameters)

        train = dataset

        if parameters['num_classes'] == 5:
            test_name = "infacttest"
        if parameters['num_classes'] == 4:
            test_name = "transtest"

        DATA_test = get_dataset(test_name)
        dataset_test = DATA_test(split="test", **parameters)

        test = dataset_test
        test.split = test

    datasets = {"train": train,
                "test": test}

    # add specific parameters from the dataset loading
    dataset.update_parameters(parameters)

    return datasets

def get_testset(parameters):
    if parameters['num_classes'] == 5:
        name = "infacttest"
    if parameters['num_classes'] == 4:
        name = "transtest"
 
    DATA = get_dataset(name)
    testset = DATA(split="test", **parameters)

    return testset

def get_synset(parameters):
    if parameters['num_classes'] == 5:
        name = "infactsyn"
    if parameters['num_classes'] == 4:
        name = "transsyn"
 
    DATA = get_dataset(name)
    testset = DATA(split="test", **parameters)

    return testset

