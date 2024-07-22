import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from math import exp

from modules import Encoder, PosEmbeds, CoordQuantizer
from modules.slot_attention import SlotAttentionBase
from utils import spatial_flatten, hungarian_huber_loss, average_precision_clevr


class QuantizedClassifier(pl.LightningModule):
    """
    Slot Attention based classifier for set prediction task
    """
    def __init__(self, resolution=(128, 128),
                    num_slots=10, num_iters=4, in_channels=3,
                    hidden_size=64, slot_size=64, 
                    lr=0.00035, coord_scale=1., nums=[8,8,8,8], **kwargs):
        super().__init__()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iters = num_iters
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.coord_scale = coord_scale
        
        self.encoder_cnn = Encoder(in_channels=self.in_channels, hidden_size=hidden_size)
        self.encoder_pos = PosEmbeds(hidden_size, (resolution[0] // 4, resolution[1] // 4))

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, slot_size)
        )
        self.slot_attention = SlotAttentionBase(num_slots=num_slots, iters=num_iters, dim=slot_size, hidden_dim=slot_size*2)

        self.mlp_coords = nn.Sequential(
            nn.Linear(slot_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Sigmoid()
        )
        self.mlp_prop = nn.Sequential(
            nn.Linear(slot_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 19 - 3),
            # nn.Sigmoid()
        )
        self.quantizer = CoordQuantizer(nums=nums)
        
        self.thrs = [-1, 1, 0.5, 0.25, 0.125]
        self.smax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, inputs):
        x = self.encoder_cnn(inputs)
        _, pos = self.encoder_pos(x)
        x = spatial_flatten(x)
        pos = spatial_flatten(pos)
        x = x + pos
        x = x = self.mlp(self.layer_norm(x))
        x = self.slot_attention(x)
        

        props, coords, _ = self.quantizer(x)
        coords = x
        coords = self.mlp_coords(coords)
        
        # props = x
        props = self.mlp_prop(props)
        
        props[:, :, 0:2] = self.smax(props[:, :, 0:2].clone())
        props[:, :, 2:4] = self.smax(props[:, :, 2:4].clone())
        props[:, :, 4:7] = self.smax(props[:, :, 4:7].clone())
        props[:, :, 7:15] = self.smax(props[:, :, 7:15].clone())
        props[:, :, 15:] = self.sigmoid(props[:, :, 15:].clone()) 

        res = torch.cat([coords, props], dim=-1)
        return {
            'prediction': res,
            # 'log_likelihood': log_l
        }

    def step(self, batch, batch_idx):
        images = batch['image']
        targets = batch['target']
        result = self(images)
        hung_loss = hungarian_huber_loss(result['prediction'], targets) #, coord_scale=self.coord_scale,)
        loss = hung_loss #+ result['log_likelihood']*0.

        metrics = {
            'loss': loss,
            # 'log_likelihood': result['log_likelihood'],
            # 'qunatizer loss': quant_loss,
            # 'inner sim loss': sim_loss, 
            # 'comitment loss': com_loss,
            'hungarian huber loss': hung_loss,
           # 'coord entropy': coord_entr
            }
        ap_metrics = {}
        if batch_idx == 1:
            ap_metrics = {
                f'ap thr={thr}': average_precision_clevr(
                    result['prediction'].detach().cpu().numpy(), 
                    targets.detach().cpu().numpy(), 
                    thr
                    )
                for thr in self.thrs
            }

        return metrics, ap_metrics

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        sch = self.lr_schedulers()

        metrics, ap_metrics = self.step(batch, batch_idx)
        self.log_dict({f'train/{key}': value for key, value in metrics.items()}, on_step=False, on_epoch=True)
        if batch_idx == 1:
            self.log_dict({f'val/{key}': value for key, value in ap_metrics.items()}, on_step=False, on_epoch=True)

        optimizer.zero_grad()
        self.manual_backward(metrics['loss'])
        optimizer.step()
        #self.manual_optimizer_step(optimizer)
        sch.step()
        self.log('lr', sch.get_last_lr()[0], on_step=False, on_epoch=True)

        return metrics['loss']

    def validation_step(self, batch, batch_idx):
        metrics, ap_metrics = self.step(batch, batch_idx)
        self.log_dict({f'val/{key}': value for key, value in metrics.items()}, on_step=False, on_epoch=True)
        if batch_idx == 1:
            self.log_dict({f'val/{key}': value for key, value in ap_metrics.items()}, on_step=False, on_epoch=True)
        return metrics['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams["lr"], total_steps=self.trainer.estimated_stepping_batches, pct_start=0.05)
        return [optimizer], [scheduler]

