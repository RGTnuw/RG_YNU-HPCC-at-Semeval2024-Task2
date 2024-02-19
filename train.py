import argparse
import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from dataloader import Recorddataset
from evaluate import eval
from model import Model

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.072, contrast_mode='all', base_temperature=0.072):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        features = features.view(features.shape[0], features.shape[1], -1)
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--ptlm", default='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', type=str)
    parser.add_argument("--lmn", default="deberta", type=str, help="name of the language model")
    parser.add_argument('--data', type=str, help='data dir')
    parser.add_argument("--epoch", default=40, type=int)
    parser.add_argument("--eval_every", default=10, type=int)
    parser.add_argument("--prompt", default=0, choices=[0, 1], type=int)
    parser.add_argument("--mode", default='trn', choices=['mix', 'trn'])
    parser.add_argument("--from_check_point", default=False, type=bool)
    parser.add_argument("--tokenizer_dir", default=None, type=str, help='the tokenizer check point dir')
    parser.add_argument("--model_dir", default=None, type=str, help='the model check point dir')
    parser.add_argument("--seed", default=621, type=int)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    # 加载数据集
    if args.mode == 'trn':
        trn_dataset = Recorddataset(args, args.data, "train")
    else:
        trn_dataset = Recorddataset(args, args.data, "trn&dev")
    dev_dataset = Recorddataset(args, args.data, "dev")

    trn_loader = DataLoader(trn_dataset, batch_size=4, shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=False, drop_last=False)

    seed_val = args.seed
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    output_dir = "./result/{}_prompt{}_mode{}_epoch{}_eval{}/".format(args.ptlm, args.prompt, args.mode, args.epoch,
                                                                          args.eval_every)
    os.makedirs(output_dir, exist_ok=True)
    epochs = args.epoch
    num_total_steps = len(trn_loader) * epochs
    num_warmup_steps = len(trn_loader) * int(args.epoch / 8)

    model = Model(args, args.ptlm, args.from_check_point, args.tokenizer_dir, args.model_dir)
    model.to(device)
    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=5e-6, correct_bias=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_total_steps)
    # 训练和评估循环
    best_val, best_val_epoch = 0, 0
    best_recall, best_precision = 0, 0
    scl_loss_fct = SupConLoss()
    for epoch in range(epochs):
        total_loss = 0
        for iter, (sent, label) in enumerate(
                tqdm(trn_loader, desc=f'epoch: {epoch + 1}/{epochs}')): 
            label = label.to(device)
            output = model(sent, label, device)
            pred = torch.argmax(output[1], dim=-1)
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(output.logits.view(-1, 2), label.view(-1))
            features = output[1]
            scl_loss = scl_loss_fct(features, label)
            total_loss = 0.701 * ce_loss + 0.299 * scl_loss
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if iter % args.eval_every == 0 and iter != 0:
                with torch.no_grad():
                    l, f, p, r = eval(model, dev_loader, device, print_on_screen=False)
                print(
                    f"The Validation result at epoch {epoch + 1} iter {iter}: val_loss: {l}, val_f1: {f}, val_precision: {p}, val_recall: {r}")
                if f > best_val:
                    best_val_epoch = epoch + 1
                    best_val = f
                    best_precision = p
                    best_recall = r
                    model.save_model(output_dir)
                    print("total_loss_per_epoch: ", total_loss, "best_val", best_val, 'best_r', best_recall, 'best_p',
              best_precision, "best_val_epoch", best_val_epoch)
                    
    print("total_loss_per_epoch: ", total_loss, "best_val", best_val, 'best_r', best_recall, 'best_p',
              best_precision, "best_val_epoch", best_val_epoch)

if __name__ == '__main__':
    main()
    