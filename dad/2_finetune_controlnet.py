import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import sys
sys.path.append("/home/ubuntu/dad/ControlNet")
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

from vis_prior.vis_prior_dataset import VisPriorDataset

def finetune_controlnet(
        annotation_path,
        model_config, 
        resume_path, 
        save_folder, 
        save_name, 
        gpus, 
        batch_size, 
        accumulate_grad_batches,
        max_epochs,
        every_n_epochs,
        logger_freq,
        learning_rate,
        sd_locked,
        only_mid_control,
    ):

    dataset = VisPriorDataset(annotation_file=annotation_path)

    save_path = os.path.join(save_folder, save_name)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--annotation_path", default=None, type=str)  # the visual prior json annotation file path
    parser.add_argument("--model_config", default='/home/ubuntu/dad/ControlNet/models/cldm_v15.yaml', type=str)
    parser.add_argument("--resume_path", default='control_sd15_hed.pth', type=str)
    parser.add_argument("--save_folder", default="/media/data/dad/cnet/experiments/", type=str)
    parser.add_argument("-n", "--save_name", default=None, type=str)
    parser.add_argument("--gpus", default=-1, type=int)
    parser.add_argument("--batch_size", default=1, type=int)  # can only be 1 for 24G GPU
    parser.add_argument("--accumulate_grad_batches", default=4, type=int)  # real_batch_size = batch_size * accumulate_grad_batches
    parser.add_argument("-e", "--max_epochs", default=100, type=int) 
    parser.add_argument("--every_n_epochs", default=10, type=int)
    parser.add_argument("--logger_freq", default=100, type=int)
    parser.add_argument("-l", "--learning_rate", default=1e-6, type=float)
    parser.add_argument("-s", "--sd_locked", action='store_false')
    parser.add_argument("-m", "--only_mid_control", action='store_true')
    args = parser.parse_args()

    finetune_controlnet(
            annotation_path=args.annotation_path,
            model_config=args.model_config,
            resume_path=args.resume_path,
            save_folder=args.save_folder,
            save_name=args.save_name,
            gpus=args.gpus,
            batch_size=args.batch_size,
            accumulate_grad_batches=args.accumulate_grad_batches,
            max_epochs=args.max_epochs,
            every_n_epochs=args.every_n_epochs,
            logger_freq=args.logger_freq,
            learning_rate=args.learning_rate,
            sd_locked=args.sd_locked,
            only_mid_control=args.only_mid_control,
        )


if __name__ == "__main__":
    main()