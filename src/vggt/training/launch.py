from omegaconf import DictConfig, OmegaConf
from trainer import Trainer


path = "config/default.yaml"
cfg = OmegaConf.load(path)


trainer = Trainer(**cfg)
import pdb;pdb.set_trace()
m=1
