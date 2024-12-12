from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import os


# Configs
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# resume_path = '/project/cigserver5/export1/david.w/MixViewDiff/ControlNet/lightning_logs/version_1/checkpoints/epoch=36-step=63084.ckpt'
resume_path = '/project/cigserver5/export1/david.w/MixViewDiff/ControlNet/models/control_sd15_ini_debug_mix.ckpt'
batch_size = 8
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False

print('resume_path:', resume_path)
# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=4, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
# trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])

trainer = pl.Trainer(
    devices=2,           # Number of GPUs to use
    accelerator="gpu",   # Specify GPU usage
    strategy="ddp",      # Distributed Data Parallel
    precision=16,        # 32-bit precision
    callbacks=[logger]
)


# Train!
trainer.fit(model, dataloader)
