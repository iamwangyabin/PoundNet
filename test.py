import io
import sys
import os
import csv
import argparse
import hydra
import torch.utils.data
import pickle

import data
import utils
from utils.util import load_config_with_cli
from utils.network_factory import get_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    model = get_model(conf)
    eval(conf.resume.target)(model, conf.resume.path)
    model.cuda()
    model.eval()

    all_results = []
    save_raw_results = {}

    for sub_data in conf.datasets.source:
        data_root = sub_data.data_root
        for sub_set in sub_data.sub_sets:
            dataset = eval(sub_data.target)(sub_data.data_root, conf.datasets.trsf,subset=sub_set, split=sub_data.split)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=conf.datasets.batch_size,
                                                      num_workers=conf.datasets.loader_workers, shuffle=True)

            result = eval(conf.eval_pipeline)(model, data_loader)

            ap = result['ap']
            auc = result['auc']
            f1 = result['f1']
            r_acc0 = result['r_acc0']
            f_acc0 = result['f_acc0']
            acc0 = result['acc0']
            num_real = result['num_real']
            num_fake = result['num_fake']

            print(f"{sub_data.benchmark_name} {sub_set}")
            print(f"AP: {ap:.4f},\tF1: {f1:.4f},\tAUC: {auc:.4f},\tACC: {acc0:.4f},\tR_ACC: {r_acc0:.4f},\tF_ACC: {f_acc0:.4f}")
            all_results.append([sub_data.benchmark_name, sub_set, ap, auc, f1, r_acc0, f_acc0, acc0, num_real, num_fake])
            save_raw_results[f"{sub_data.benchmark_name} {sub_set}"] = result


    columns = ['dataset', 'sub_set', 'ap', 'auc', 'f1', 'r_acc0', 'f_acc0', 'acc0', 'num_real', 'num_fake']
    with open(conf.test_name+'_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(columns)
        for values in all_results:
            writer.writerow(values)
    with open(conf.test_name + '.pkl', 'wb') as file:
        pickle.dump(save_raw_results, file)
