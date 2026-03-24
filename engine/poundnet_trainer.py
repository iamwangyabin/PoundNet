import lightning as L
import torch
from torch.nn import functional as F

from utils.network_factory import get_model
from utils.validate import validate


def generate_mapping(base_number):
    mapping = {}
    for index in range(base_number * 2):
        if index % 2 == 0:
            mapping[index] = index // 2
        else:
            mapping[index] = base_number + (index - 1) // 2
    return mapping


class Trainer_PoundNet(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_gts = []
        self.validation_step_outputs_preds = []
        self.mapping = generate_mapping(len(opt.datasets.train.multicalss_names))

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x, return_binary=True)
        logits = output["logits"]
        b_logits = output["b_logits"]

        loss = logits.new_tensor(0.0)

        if self.opt.train.a != 0:
            cls_y = y // 2
            logits_groups = torch.chunk(logits, 2 * self.opt.model.PROMPT_NUM_TEXT, dim=1)
            for logits_group in logits_groups:
                loss = loss + self.opt.train.a * F.cross_entropy(logits_group, cls_y)

        if self.opt.train.b != 0:
            new_y = torch.tensor(
                [self.mapping[label.item()] for label in y],
                dtype=torch.long,
                device=y.device,
            )
            loss = loss + self.opt.train.b * F.cross_entropy(logits, new_y)

        if self.opt.train.c != 0:
            loss = loss + self.opt.train.c * F.cross_entropy(b_logits, y % 2)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.forward_binary(x)["logits"]
        self.validation_step_outputs_preds.append(F.softmax(logits, 1)[:, 1].detach())
        self.validation_step_outputs_gts.append(y.detach())

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(torch.float32).flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(torch.float32).cpu().numpy()
        acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)
        self.log("val_acc_epoch", acc, logger=True, sync_dist=True)
        self.log("val_ap_epoch", ap, logger=True, sync_dist=True)
        self.log("val_racc_epoch", r_acc, logger=True, sync_dist=True)
        self.log("val_facc_epoch", f_acc, logger=True, sync_dist=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()

    def configure_optimizers(self):
        parameters = filter(lambda parameter: parameter.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(parameters)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]
