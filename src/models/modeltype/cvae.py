import torch
from .cae import CAE


class CVAE(CAE):
    def reparameterize(self, batch, noise_factor=1.0, seed=None):
        mu, logvar = batch["mu"], batch["logvar"]
        std = torch.exp(logvar / 2) * noise_factor

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        return z

    def forward(self, batch):
        
        if self.outputxyz:
            batch["x_xyz"], _ = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        #fact = batch['noise_factor']
        fact = 1.0
        batch["z"] = self.reparameterize(batch, fact)
        
        # decode
        batch.update(self.decoder(batch))
        
        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"], batch["output_pose"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def return_latent(self, batch, seed=None):
        distrib_param = self.encoder(batch)
        batch.update(distrib_param)
        fact = 1.5 #batch['noise_factor']
        return self.reparameterize(batch, fact, seed=seed)
