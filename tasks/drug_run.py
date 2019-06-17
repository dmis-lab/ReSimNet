import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import csv
import os
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics import precision_score, roc_auc_score

from datetime import datetime
from torch.autograd import Variable
from models.root.utils import *


LOGGER = logging.getLogger(__name__)


def prob_to_class(prob):
    return np.array([float(p >= 0.5) for p in prob])


def run_bi(model, loader, dataset, args, metric, train=False):
    total_step = 0.0
    stats = {'loss':[]}
    tar_set = []
    pred_set = []
    kk_tar_set = []
    kk_pred_set = []
    ku_tar_set = []
    ku_pred_set = []
    uu_tar_set = []
    uu_pred_set = []
    start_time = datetime.now()

    for d_idx, (d1, d1_r, d1_l, d2, d2_r, d2_l, score) in enumerate(loader):

        # Split for KK/KU/UU sets
        kk_idx = np.argwhere([a in dataset.known and b in dataset.known
                              for a, b in zip(d1, d2)]).flatten()
        ku_idx = np.argwhere([(a in dataset.known) != (b in dataset.known)
                              for a, b in zip(d1, d2)]).flatten()
        uu_idx = np.argwhere([a not in dataset.known and b not in dataset.known
                              for a, b in zip(d1, d2)]).flatten()
        assert len(kk_idx) + len(ku_idx) + len(uu_idx) == len(d1)

        # Grad zero + mode change
        model.optimizer.zero_grad()
        if train: model.train(train)
        else: model.eval()

        # Get outputs
        outputs, embed1, embed2 = model(d1_r.cuda(), d1_l, d2_r.cuda(), d2_l,
                                        None, None)
        loss = model.get_loss(outputs, score.cuda())
        stats['loss'] += [loss.data[0]]
        total_step += 1.0

        # Metrics for binary classification
        tmp_tar = score.data.cpu().numpy()
        tmp_pred = outputs.data.cpu().numpy()
        # tmp_pred = np.array([float(p >= 0.5) for p in tmp_pred[:]])
        # print(tmp_tar[:5], tmp_pred[:5])

        # Accumulate for final evaluation
        tar_set += list(tmp_tar[:])
        pred_set += list(tmp_pred[:])
        kk_tar_set += list(tmp_tar[kk_idx])
        kk_pred_set += list(tmp_pred[kk_idx])
        ku_tar_set += list(tmp_tar[ku_idx])
        ku_pred_set += list(tmp_pred[ku_idx])
        uu_tar_set += list(tmp_tar[uu_idx])
        uu_pred_set += list(tmp_pred[uu_idx])

        # Calculate current f1 scores
        f1 = metric(list(tmp_tar[:]), list(prob_to_class(tmp_pred[:])))
        f1_kk = metric(list(tmp_tar[kk_idx]), list(prob_to_class(tmp_pred[kk_idx])))
        f1_ku = metric(list(tmp_tar[ku_idx]), list(prob_to_class(tmp_pred[ku_idx])))
        f1_uu = metric(list(tmp_tar[uu_idx]), list(prob_to_class(tmp_pred[uu_idx])))

        # For binary classification, report f1
        _, _, f1, _ = f1
        _, _, f1_kk, _ = f1_kk
        _, _, f1_ku, _ = f1_ku
        _, _, f1_uu, _ = f1_uu

        # Optimize model
        if train and not args.save_embed:
            loss.backward()
            nn.utils.clip_grad_norm(model.get_model_params()[1],
                    args.grad_max_norm)
            model.optimizer.step()

        # Print for print step or at last
        if d_idx % args.print_step == 0 or d_idx == (len(loader) - 1):
            et = int((datetime.now() - start_time).total_seconds())
            _progress = (
                '{}/{} | Loss: {:.3f} | Total F1: {:.3f} | '.format(
                d_idx + 1, len(loader), loss.data[0], f1) +
                'KK: {:.3f} KU: {:.3f} UU: {:.3f} | '.format(
                f1_kk, f1_ku, f1_uu) +
                '{:2d}:{:2d}:{:2d}'.format(
                et//3600, et%3600//60, et%60))
            LOGGER.debug(_progress)

    if args.top_only:
    # if False:
        tar_sets = [tar_set, kk_tar_set, ku_tar_set, uu_tar_set]
        pred_sets = [pred_set, kk_pred_set, ku_pred_set, uu_pred_set]
        messages = ['Total', 'KK', 'KU', 'UU']
        top_criterion = 0.10
        top_k = 100

        for tar, pred, msg in zip(tar_sets, pred_sets, messages):
            sorted_target = sorted(tar[:], reverse=True)
            # top_cut = sorted_target[int(len(sorted_target) * top_criterion)]
            top_cut = 0.9

            sorted_pred, my_target = (list(t) for t in zip(*sorted(
                                      zip(pred[:], tar[:]), reverse=True)))
            precision = sum(k >= top_cut for k in my_target[:top_k]) / top_k
            LOGGER.info('{} cut: {:.3f}, P@{}: {:.2f}, '.format(
                        msg, top_cut, top_k, precision) +
                        'Pred Mean@100: {:.3f}, Tar Mean@100: {:.3f}'.format(
                        sum(sorted_pred[:top_k])/top_k,
                        sum(my_target[:top_k])/top_k))

    def sort_and_slice(list1, list2):
        list2, list1 = (list(t) for t in zip(*sorted(
                        zip(list2, list1), reverse=True)))
        list1 = list1[:len(list1)//100] + list1[-len(list1)//100:]
        # list1 = list1[-len(list1)//100:]
        list2 = list2[:len(list2)//100] + list2[-len(list2)//100:]
        # list2 = list2[-len(list2)//100:]
        assert len(list1) == len(list2)
        return list1, list2

    if args.top_only:
    # if False:
        tar_set, pred_set = sort_and_slice(tar_set, pred_set)
        kk_tar_set, kk_pred_set = sort_and_slice(kk_tar_set, kk_pred_set)
        ku_tar_set, ku_pred_set = sort_and_slice(ku_tar_set, ku_pred_set)
        uu_tar_set, uu_pred_set = sort_and_slice(uu_tar_set, uu_pred_set)

    # Calculate acuumulated f1 scores
    f1 = metric(tar_set, prob_to_class(pred_set))
    f1_kk = metric(kk_tar_set, prob_to_class(kk_pred_set))
    f1_ku = metric(ku_tar_set, prob_to_class(ku_pred_set))
    f1_uu = metric(uu_tar_set, prob_to_class(uu_pred_set))
    pr, rc, f1, _ = f1
    pr_kk, rc_kk, f1_kk, _ = f1_kk
    pr_ku, rc_ku, f1_ku, _ = f1_ku
    pr_uu, rc_uu, f1_uu, _ = f1_uu

    # TODO add spearman correlation

    # End of an epoch
    et = (datetime.now() - start_time).total_seconds()
    LOGGER.info('Results (Loss/F1/KK/KU/UU): {:.3f}\t'.format(
        sum(stats['loss'])/len(stats['loss'])) +
        '[{:.3f}\t{:.3f}\t{:.3f}]\t[{:.3f}\t{:.3f}\t{:.3f}]\t'.format(
        pr, rc, f1, pr_kk, rc_kk, f1_kk) +
        '[{:.3f}\t{:.3f}\t{:.3f}]\t[{:.3f}\t{:.3f}\t{:.3f}]\t'.format(
        pr_ku, rc_ku, f1_ku, pr_uu, rc_uu, f1_uu) +
        'count: {}/{}/{}/{}'.format(
        len(pred_set), len(kk_pred_set), len(ku_pred_set), len(uu_pred_set)))

    return f1_ku


def element(d):
    return [d[k] for k in range(0,len(d))]


def run_reg(model, loader, dataset, args, metric, train=False):
    total_step = 0.0
    stats = {'loss':[]}
    tar_set = []
    pred_set = []
    kk_tar_set = []
    kk_pred_set = []
    ku_tar_set = []
    ku_pred_set = []
    uu_tar_set = []
    uu_pred_set = []
    start_time = datetime.now()

    for d_idx, d in enumerate(loader):
        if args.rep_idx == 4:
            d1, d1_r, d1_a, d1_l, d2, d2_r, d2_a, d2_l, score = element(d)
        else:
            d1, d1_r, d1_l, d2, d2_r, d2_l, score = element(d)

        # Split for KK/KU/UU sets
        kk_idx = np.argwhere([a in dataset.known and b in dataset.known
                              for a, b in zip(d1, d2)]).flatten()
        ku_idx = np.argwhere([(a in dataset.known) != (b in dataset.known)
                              for a, b in zip(d1, d2)]).flatten()
        uu_idx = np.argwhere([a not in dataset.known and b not in dataset.known
                              for a, b in zip(d1, d2)]).flatten()
        assert len(kk_idx) + len(ku_idx) + len(uu_idx) == len(d1)

        # Grad zero + mode change
        model.optimizer.zero_grad()
        if train: model.train(train)
        else: model.eval()

        # Get outputs
        if args.rep_idx == 4:
            outputs, embed1, embed2 = model(d1_r.cuda(), d1_l,
                                            d2_r.cuda(), d2_r,
                                            d1_a.cuda(), d2_a.cuda())
        else:
            outputs, embed1, embed2 = model(d1_r.cuda(), d1_l,
                                            d2_r.cuda(), d2_l,
                                            None, None)
        loss = model.get_loss(outputs, score.cuda())
        stats['loss'] += [loss.data[0]]
        total_step += 1.0

        # Metrics for regression
        tmp_tar = score.data.cpu().numpy()
        tmp_pred = outputs.data.cpu().numpy()
        # print(tmp_tar[:10])

        # Accumulate for final evaluation
        tar_set += list(tmp_tar[:])
        pred_set += list(tmp_pred[:])
        kk_tar_set += list(tmp_tar[kk_idx])
        kk_pred_set += list(tmp_pred[kk_idx])
        ku_tar_set += list(tmp_tar[ku_idx])
        ku_pred_set += list(tmp_pred[ku_idx])
        uu_tar_set += list(tmp_tar[uu_idx])
        uu_pred_set += list(tmp_pred[uu_idx])

        # Calculate current f1 scores
        f1 = metric(list(tmp_tar[:]), list(tmp_pred[:]))
        f1_kk = metric(list(tmp_tar[kk_idx]), list(tmp_pred[kk_idx]))
        f1_ku = metric(list(tmp_tar[ku_idx]), list(tmp_pred[ku_idx]))
        f1_uu = metric(list(tmp_tar[uu_idx]), list(tmp_pred[uu_idx]))
        f1 = f1[0][1]
        f1_kk = f1_kk[0][1]
        f1_ku = f1_ku[0][1]
        f1_uu = f1_uu[0][1]

        # Optimize model
        if train and not args.save_embed:
            loss.backward()
            nn.utils.clip_grad_norm(model.get_model_params()[1],
                    args.grad_max_norm)
            model.optimizer.step()

        # Print for print step or at last
        if d_idx % args.print_step == 0 or d_idx == (len(loader) - 1):
            et = int((datetime.now() - start_time).total_seconds())
            _progress = (
                '{}/{} | Loss: {:.3f} | Total Corr: {:.3f} | '.format(
                d_idx + 1, len(loader), loss.data[0], f1) +
                'KK: {:.3f} KU: {:.3f} UU: {:.3f} | '.format(
                f1_kk, f1_ku, f1_uu) +
                '{:2d}:{:2d}:{:2d}'.format(
                et//3600, et%3600//60, et%60))
            LOGGER.debug(_progress)

    # if args.top_only:
    # # if False:
    #     tar_sets = [tar_set, kk_tar_set, ku_tar_set, uu_tar_set]
    #     pred_sets = [pred_set, kk_pred_set, ku_pred_set, uu_pred_set]
    #     messages = ['Total', 'KK', 'KU', 'UU']
    #     top_criterion = 0.10
    #     top_k = 100
    #
    #     for tar, pred, msg in zip(tar_sets, pred_sets, messages):
    #         sorted_target = sorted(tar[:], reverse=True)
    #         # top_cut = sorted_target[int(len(sorted_target) * top_criterion)]
    #         top_cut = 0.9
    #
    #         sorted_pred, my_target = (list(t) for t in zip(*sorted(
    #                                   zip(pred[:], tar[:]), reverse=True)))
    #         precision = sum(k >= top_cut for k in my_target[:top_k]) / top_k
    #         LOGGER.info('{} cut: {:.3f}, P@{}: {:.2f}, '.format(
    #                     msg, top_cut, top_k, precision) +
    #                     'Pred Mean@100: {:.3f}, Tar Mean@100: {:.3f}'.format(
    #                     sum(sorted_pred[:top_k])/top_k,
    #                     sum(my_target[:top_k])/top_k))
    #
    # def sort_and_slice(list1, list2):
    #     list2, list1 = (list(t) for t in zip(*sorted(
    #                     zip(list2, list1), reverse=True)))
    #     list1 = list1[:len(list1)//100] + list1[-len(list1)//100:]
    #     # list1 = list1[-len(list1)//100:]
    #     list2 = list2[:len(list2)//100] + list2[-len(list2)//100:]
    #     # list2 = list2[-len(list2)//100:]
    #     assert len(list1) == len(list2)
    #     return list1, list2
    #
    # if args.top_only:
    # # if False:
    #     tar_set, pred_set = sort_and_slice(tar_set, pred_set)
    #     kk_tar_set, kk_pred_set = sort_and_slice(kk_tar_set, kk_pred_set)
    #     ku_tar_set, ku_pred_set = sort_and_slice(ku_tar_set, ku_pred_set)
    #     uu_tar_set, uu_pred_set = sort_and_slice(uu_tar_set, uu_pred_set)

    # Calculate acuumulated f1 scores
    f1 = metric(tar_set, pred_set)
    f1_kk = metric(kk_tar_set, kk_pred_set)
    f1_ku = metric(ku_tar_set, ku_pred_set)
    f1_uu = metric(uu_tar_set, uu_pred_set)

    # Trun into correlation
    f1 = f1[0][1]
    f1_kk = f1_kk[0][1]
    f1_ku = f1_ku[0][1]
    f1_uu = f1_uu[0][1]

    # End of an epoch
    et = (datetime.now() - start_time).total_seconds()
    LOGGER.info('Results (Loss/F1/KK/KU/UU): {:.4f}\t'.format(
        sum(stats['loss'])/len(stats['loss'])) +
        '[{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}] '.format(
        f1, f1_kk, f1_ku, f1_uu) +
        'count: {}/{}/{}/{}'.format(
        len(pred_set), len(kk_pred_set), len(ku_pred_set), len(uu_pred_set)))


    corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(pred_set, tar_set)
    LOGGER.info('[TOTAL\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}] '.format(
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5))

    corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(kk_pred_set, kk_tar_set)
    LOGGER.info('[KK\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}] '.format(
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5))

    corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(ku_pred_set, ku_tar_set)
    LOGGER.info('[KU\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}] '.format(
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5))

    corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(uu_pred_set, uu_tar_set)
    LOGGER.info('[UU\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}] '.format(
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5))

    return f1_ku

def precision_at_k(y_pred, y_true, k):
    list_of_tuple = [(x, y) for x, y in zip(y_pred, y_true)]
    sorted_list_of_tuple = sorted(list_of_tuple, key=lambda tup: tup[0], reverse=True)
    topk = sorted_list_of_tuple[:int(len(sorted_list_of_tuple) * k)]
    topk_true = [x[1] for x in topk]
    topk_pred = [x[0] for x in topk]
    #print(topk)
    #print(topk_true)
    #print(topk_pred)
    precisionk = precision_score([1 if x > 0.9 else 0 for x in topk_true],
                                 [1 if x > -1.0 else 0 for x in topk_pred], labels=[0,1], pos_label=1)
    # print([1 if x > 90.0 else 0 for x in topk_true])
    # print([1 if x > 90.0 else 0 for x in topk_pred])
    # print(precisionk)
    return precisionk

def mse_at_k(y_pred, y_true, k):
    list_of_tuple = [(x, y) for x, y in zip(y_pred, y_true)]
    sorted_list_of_tuple = sorted(list_of_tuple, key=lambda tup: tup[0], reverse=True)
    topk = sorted_list_of_tuple[:int(len(sorted_list_of_tuple) * k)]
    topk_true = [x[1] for x in topk]
    topk_pred = [x[0] for x in topk]

    msek = np.square(np.subtract(topk_pred, topk_true)).mean()
    return msek

def evaluation(y_pred, y_true):
    # print(y_pred)
    # print(y_true)
    # print(pearsonr(np.ravel(y_pred), y_true))
    corr = pearsonr(np.ravel(y_pred), y_true)[0]
    # mse = np.square(np.subtract(y_pred, y_true)).mean()
    msetotal = mse_at_k(y_pred, y_true, 1.0)
    mse1 = mse_at_k(y_pred, y_true, 0.01)
    mse2 = mse_at_k(y_pred, y_true, 0.02)
    mse5 = mse_at_k(y_pred, y_true, 0.05)

    auroc = float('nan')
    if len([x for x in y_true if x > 0.9]) > 0:
        auroc = roc_auc_score([1 if x > 0.9 else 0 for x in y_true], y_pred)
    precision1 = precision_at_k(y_pred, y_true, 0.01)
    precision2 = precision_at_k(y_pred, y_true, 0.02)
    precision5 = precision_at_k(y_pred, y_true, 0.05)
    #print(auroc, precision1, precision2, precision5)
    return (corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5)


# Outputs response embeddings for a given dictionary
def save_embed(model, dictionary, dataset, args, drug_file):
    model.eval()
    key2vec = {}
    known_cnt = 0

    # Iterate drug dictionary
    for idx, item in enumerate(dictionary.items()):
        drug, rep = [item[k] for k in range(0,len(item))]
        if args.embed_d == 1:
            d1_r = rep[args.rep_idx]
            d1_k = drug in dataset.known
            d1_l = len(d1_r)
        else:
            d1_r = rep[0]
            d1_k = rep[1]
            d1_l = len(d1_r)

        # For string data (smiles/inchikey)
        if args.rep_idx == 0 or args.rep_idx == 1:
            d1_r = list(map(lambda x: dataset.char2idx[x]
                        if x in dataset.char2idx
                        else dataset.char2idx[dataset.UNK], d1_r))
            d1_l = len(d1_r)

        # Real valued for mol2vec
        if args.rep_idx != 3:
            d1_r = Variable(torch.LongTensor(d1_r)).cuda()
        else:
            d1_r = Variable(torch.FloatTensor(d1_r)).cuda()
        d1_l = torch.LongTensor(np.array([d1_l]))
        d1_r = d1_r.unsqueeze(0)
        d1_l = d1_l.unsqueeze(0)

        # Run model amd save embed
        _, embed1, embed2 = model(d1_r, d1_l, d1_r, d1_l, None, None)
        assert embed1.data.tolist() == embed2.data.tolist()
        """
        known = False
        for pert_id, _ in dataset.drugs.items():
            if drug == pert_id:
                known = True
                known_cnt += 1
                break
        """
        key2vec[drug] = [embed1.squeeze().data.tolist(), d1_k]

        # Print progress
        if idx % args.print_step == 0 or idx == len(dictionary) - 1:
            _progress = '{}/{} saving drug embeddings..'.format(
                idx + 1, len(dictionary))
            LOGGER.info(_progress)

    # Save embed as pickle
    pickle.dump(key2vec, open('{}/embed/{}.{}.pkl'.format(
                args.checkpoint_dir, drug_file, args.model_name), 'wb'),
                protocol=2)
    LOGGER.info('{}/{} number of known drugs.'.format(known_cnt, len(key2vec)))


# Outputs pred vs label scores given a dataloader
def save_prediction(model, loader, dataset, args):
    model.eval()
    csv_writer = csv.writer(open(args.checkpoint_dir + 'pred_' +
                                 args.model_name + '.csv', 'w'))
    csv_writer.writerow(['pert1', 'pert1_known', 'pert2', 'pert2_known',
                         'prediction', 'target'])

    for d_idx, (d1, d1_r, d1_l, d2, d2_r, d2_l, score) in enumerate(loader):

        # Run model for getting predictions
        outputs, _, _ = model(d1_r.cuda(), d1_l, d2_r.cuda(), d2_l, None, None)
        predictions = outputs.data.cpu().numpy()
        targets = score.data.tolist()

        for a1, a2, a3, a4 in zip(d1, d2, predictions, targets):
            csv_writer.writerow([a1, a1 in dataset.known,
                                 a2, a2 in dataset.known, a3, a4])

        # Print progress
        if d_idx % args.print_step == 0 or d_idx == len(loader) - 1:
            _progress = '{}/{} saving drug predictions..'.format(
                d_idx + 1, len(loader))
            LOGGER.info(_progress)

# Outputs pred vs label scores given a dataloader
def perform_ensemble(model, loader, dataset, args):
    model.eval()
    tar_set = []
    pred_set = []
    kk_tar_set = []
    kk_pred_set = []
    ku_tar_set = []
    ku_pred_set = []
    uu_tar_set = []
    uu_pred_set = []

    for d_idx, (d1, d1_r, d1_l, d2, d2_r, d2_l, score) in enumerate(loader):
        # Run model for getting predictions
        outputs, _, _ = model(d1_r.cuda(), d1_l, d2_r.cuda(), d2_l, None, None)

        # Split for KK/KU/UU sets
        kk_idx = np.argwhere([a in dataset.known and b in dataset.known
                              for a, b in zip(d1, d2)]).flatten()
        ku_idx = np.argwhere([(a in dataset.known) != (b in dataset.known)
                              for a, b in zip(d1, d2)]).flatten()
        uu_idx = np.argwhere([a not in dataset.known and b not in dataset.known
                              for a, b in zip(d1, d2)]).flatten()
        assert len(kk_idx) + len(ku_idx) + len(uu_idx) == len(d1)

        # Metrics for regression
        tmp_tar = score.data.cpu().numpy()
        tmp_pred = outputs.data.cpu().numpy()

        # Accumulate for final evaluation
        tar_set += list(tmp_tar[:])
        pred_set += list(tmp_pred[:])
        kk_tar_set += list(tmp_tar[kk_idx])
        kk_pred_set += list(tmp_pred[kk_idx])
        ku_tar_set += list(tmp_tar[ku_idx])
        ku_pred_set += list(tmp_pred[ku_idx])
        uu_tar_set += list(tmp_tar[uu_idx])
        uu_pred_set += list(tmp_pred[uu_idx])

    corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(pred_set, tar_set)
    print('[TOTAL\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}] '.format(
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5))

    corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(kk_pred_set, kk_tar_set)
    print('[KK\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}] '.format(
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5))

    corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(ku_pred_set, ku_tar_set)
    print('[KU\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}] '.format(
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5))

    corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5 = evaluation(uu_pred_set, uu_tar_set)
    print('[UU\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}\t{:.5f}] '.format(
        corr, msetotal, mse1, mse2, mse5, auroc, precision1, precision2, precision5))

    return pred_set, tar_set, kk_pred_set, kk_tar_set, ku_pred_set, ku_tar_set, uu_pred_set, uu_tar_set

# Outputs pred scores for new pair dataset
def save_pair_score(model, pair_dir, fp_dir, dataset, args):
    model.eval()
    drug2rep = pickle.load(open(fp_dir, 'rb'))

    folder_name = args.checkpoint_dir + 'save_pair_score/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for subdir, _, files in os.walk(pair_dir):
        for file_ in sorted(files):

            df = pd.read_csv(os.path.join(subdir, file_), sep=",")
            #print(df)
            LOGGER.info('save_pair_score processing {}...'.format(file_))

            csv_writer = csv.writer(open(folder_name + file_ + '_' +
                                         args.model_name + '.csv', 'w'))
            csv_writer.writerow(['drug1', 'drug2', 'prediction', 'jaccard'])

            batch = []
            for row_idx, row in df.iterrows():
                drug1 = row['id1']
                drug1_r = drug2rep[drug1][0]
                drug1_r = [float(value) for value in list(drug1_r)]

                drug2 = row['id2']
                drug2_r = drug2rep[drug2][0]
                drug2_r = [float(value) for value in list(drug2_r)]

                example = [drug1, drug1_r, len(drug1_r),
                           drug2, drug2_r, len(drug2_r), 0]
                batch.append(example)

                if len(batch) == 1024:
                    inputs = dataset.collate_fn(batch)
                    outputs, _, _ = model(inputs[1].cuda(), inputs[2], inputs[4].cuda(), inputs[5], None, None)
                    predictions = outputs.data.cpu().numpy()

                    for example, pred in zip(batch, predictions):
                        from scipy.spatial import distance
                        def jaccard(a, b):
                           return 1-distance.jaccard(a, b)
                        jac = jaccard(example[1], example[4])

                        csv_writer.writerow([example[0], example[3], pred, jac])
                        print(example[0], example[3], pred, jac)

                    batch = []

                # Print progress
                if row_idx % 5000 == 0 or row_idx == len(df) - 1:
                    _progress = '{}/{} saving unknwon predictions..'.format(
                        row_idx + 1, len(df))
                    LOGGER.info(_progress)

            if len(batch) > 0:
                inputs = dataset.collate_fn(batch)
                outputs, _, _ = model(inputs[1].cuda(), inputs[2], inputs[4].cuda(), inputs[5], None, None)
                predictions = outputs.data.cpu().numpy()

                for example, pred in zip(batch, predictions):
                    from scipy.spatial import distance
                    def jaccard(a, b):
                       return 1-distance.jaccard(a, b)
                    jac = jaccard(example[1], example[4])
                    csv_writer.writerow([example[0], example[3], pred, jac])


def save_pair_score_for_zinc(model, pair_dir, example_dir, dataset, args):
    print("\n=============================================================")
    print("SAVE PAIR SCORE FOR ZINC")
    print("=============================================================")

    model.eval()
    df_example = pd.read_csv(example_dir, sep=",")
    print(df_example)

    folder_name = args.checkpoint_dir + 'save_pair_score_for_zinc/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for subdir, _, files in os.walk(pair_dir):
        for file_ in sorted(files):

            df_zinc = pd.read_csv(os.path.join(subdir, file_), sep=",")
            LOGGER.info('save_pair_score processing {}...'.format(file_))
            csv_writer = csv.writer(open(folder_name + file_ + '_' +
                                         args.model_name + '.csv', 'w'))
            csv_writer.writerow(['pair1', 'pair2', 'prediction'])

            batch = []
            for row_idx, row in df_zinc.iterrows():
                drug1 = row['zinc_id']
                drug1_r = row['fingerprint']
                drug1_r = [float(value) for value in list(drug1_r)]

                for row_idex, row in df_example.iterrows():
                    try:
                        drug2 = row['pair']
                        drug2_r =row['fp']
                        drug2_r = [float(value) for value in list(drug2_r)]
                        #print(drug1, drug1_r, len(drug1_r), drug2, drug2_r, len(drug2_r))

                        example = [drug1, drug1_r, len(drug1_r),
                                   drug2, drug2_r, len(drug2_r), 0]
                        batch.append(example)
                    except KeyError:
                        continue

                    if len(batch) == 4096:
                        inputs = dataset.collate_fn(batch)
                        outputs, _, _ = model(inputs[1].cuda(), inputs[2], inputs[4].cuda(), inputs[5], None, None)
                        predictions = outputs.data.cpu().numpy()

                        for example, pred in zip(batch, predictions):
                            if pred > 0.9:
                                csv_writer.writerow([example[0], example[3], pred])

                        batch = []

                # Print progress
                if row_idx % 1000 == 0 or row_idx == len(df_zinc) - 1:
                    _progress = '{}/{} saving zinc predictions..'.format(
                        row_idx + 1, len(df_zinc))
                    LOGGER.info(_progress)

            if len(batch) > 0:
                inputs = dataset.collate_fn(batch)
                outputs, _, _ = model(inputs[1].cuda(), inputs[2], inputs[4].cuda(), inputs[5], None, None)
                predictions = outputs.data.cpu().numpy()

                for example, pred in zip(batch, predictions):
                    if pred > 0.9:
                        csv_writer.writerow([example[0], example[3], pred])
