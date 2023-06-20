import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import sys
sys.path.append("/home/ubuntu/dad/ControlNet")

from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from vis_prior_dataset import VisPriorDataset

annotation_path = "/home/ubuntu/dad/roboflow-100-benchmark/rf100/apples-fvpl5/train_aug_bboxhed.json"
dataset = VisPriorDataset(annotation_file=annotation_path)


# Configs
model_config = '../../models/cldm_v15.yaml'
resume_path = '../../models/control_sd15_hed.pth'
save_folder = "/media/data/dad/cnet/experiments/"
save_name = "apple_hed_2e-6_0619"
gpus = -1
batch_size = 1
accumulate_grad_batches = 4
max_epochs = 100
every_n_epochs = 10
logger_freq = 300
learning_rate = 1e-6
sd_locked = True
only_mid_control = False

save_path = os.path.join(save_folder, save_name)


if __name__ == "__main__":
    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model(model_config).cpu()
    model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    # Misc
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)

    logger = ImageLogger(batch_frequency=logger_freq, imglog_folder=save_path)
    callbacks_ckpt = pl.callbacks.ModelCheckpoint(dirpath=save_path, every_n_epochs=every_n_epochs, save_top_k=-1)

    trainer = pl.Trainer(gpus=gpus, precision=32, callbacks=[logger, callbacks_ckpt], accumulate_grad_batches=accumulate_grad_batches, max_epochs=max_epochs)

    # Train!
    trainer.fit(model, dataloader)
