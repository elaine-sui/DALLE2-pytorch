import os
import argparse
from src import builder, constants
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger

seed_everything(1234)

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute_output_embeds", action="store_true")
    parser.add_argument("--compute_avg_output_embeds", action="store_true")
    parser.add_argument("--compute_input_output_distances", action="store_true")
    parser.add_argument("--compute_sampling_distances", action="store_true")
    parser.add_argument("--not_diffusion_prior_output", action="store_true", help="when computing input output distances, if output is diffusion prior output vs. true clip image embed")
    
    parser.add_argument("--num_gpus", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    
    args = parser.parse_args()
    return args

def setup(args):
    os.makedirs(constants.AVG_OUTPUT_EMBED_FOLDER, exist_ok=True)
    
    # get datamodule
    dm = builder.build_data_module(args.batch_size)

    # define lightning module
    model = builder.build_lightning_model()
    
    # setup pytorch-lightning trainer
    strategy = "ddp" if args.num_gpus > 1 else None
    trainer = Trainer(accelerator='gpu', devices=args.num_gpus, max_epochs=1, strategy=strategy)

    return trainer, model, dm


if __name__ == '__main__':
    args = get_parser_args()
    
    # if args.compute_avg_output_embeds:
    trainer, model, dm = setup(args)
    trainer.predict(model=model, datamodule=dm)