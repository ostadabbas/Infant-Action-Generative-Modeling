from .models.stgcn import STGCN


def get_model(parameters):
    layout = "smpl" if parameters["glob"] else "smpl_noglobal"
    if parameters['pose_rep'] == 'xyz':
        layout = 'infact'
    #layout = "smpl_upperbody"  #"smpl_lowerbody"
    model = STGCN(in_channels=parameters["nfeats"],
                  num_class=parameters["num_classes"],
                  graph_args={"layout": layout, "strategy": "spatial"},
                  edge_importance_weighting=True,
                  device=parameters["device"])
    
    model = model.to(parameters["device"])
    return model
    
