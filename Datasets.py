# -*- coding: utf-8 -*-
import pickle
from pathlib import Path
import numpy as np
import re


LABEL_ID = 0
ARG_ID = 2
PRED_ID = 1
PROP_ID = 3
POS_ID = 4
MODE_ID = 5


def load_pickle(_path):
    with _path.open('rb') as f:
        data = pickle.load(f)
    return data


def get_datasets(type):
    list_path = Path("../../data/NTC_dataset/listed_{}.pkl".format(type))
    data = load_pickle(list_path)
    label = np.array([data[i][LABEL_ID] for i in range(len(data))])
    text = np.array([(data[i][PRED_ID], data[i][ARG_ID]) for i in range(len(data))])
    prop = np.array([data[i][PROP_ID] for i in range(len(data))])
    vocab = np.array([data[i][ARG_ID] for i in range(len(data))])
    return label, text, prop, vocab


def get_datasets_in_sentences(type, with_bccwj=False, with_bert=False):
    # list_path = Path("../../data/NTC_dataset/listed_{}_rework_181130.pkl".format(type))
    # list_path = Path("../../data/NTC_dataset/listed_{}_190105.pkl".format(type))
    if with_bccwj:
        type += "_bccwj"
        if with_bert:
            type += "_bert"
        if "train2" in type:
            type = type.replace("train2", "train")
        list_path = Path("../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc/listed_{}.pkl".format(type))
    else:
        list_path = Path("../data/NTC_dataset/listed_{}.pkl".format(type))
    data = load_pickle(list_path)

    label, labels = [], []
    arg, args = [], []
    pred, preds = [], []
    prop, props = [], []
    word_pos, word_poses = [], []
    ku_pos, ku_poses = [], []
    mode, modes = [], []

    count = 0
    bef_count = 0
    for i in range(len(data)):
        if data[i][6] != bef_count:
            labels.append(label)
            args.append(arg)
            preds.append(pred)
            props.append(prop)
            word_poses.append(word_pos)
            ku_poses.append(ku_pos)
            modes.append(mode)

            label = [data[i][LABEL_ID]]
            arg = [data[i][ARG_ID]]
            pred = [data[i][PRED_ID]]
            prop = [data[i][PROP_ID]]
            word_pos = [data[i][POS_ID][0]]
            ku_pos = [data[i][POS_ID][1]]
            mode = [data[i][MODE_ID]]

            bef_count = data[i][6]
            count += 1
        else:
            label.append(data[i][LABEL_ID])
            arg.append(data[i][ARG_ID])
            pred.append(data[i][PRED_ID])
            prop.append(data[i][PROP_ID])
            word_pos.append(data[i][POS_ID][0])
            ku_pos.append(data[i][POS_ID][1])
            mode.append(data[i][MODE_ID])

    labels.append(label)
    args.append(arg)
    preds.append(pred)
    # preds.append([data[count][PRED_ID]] * len(args[-1]))
    props.append(prop)
    word_poses.append(word_pos)
    ku_poses.append(ku_pos)
    modes.append(mode)

    # labels = np.array([np.array(line) for line in labels])
    # args = np.array([np.array(line) for line in args])
    # preds = np.array([np.array(line) for line in preds])
    # props = np.array([np.array(line) for line in props])
    # word_poses = np.array([np.array(line) for line in word_poses])
    # ku_poses = np.array([np.array(line) for line in ku_poses])
    # modes = np.array([np.array(line) for line in modes])

    vocab = [data[i][ARG_ID] for i in range(len(data))]
    word_pos_id = [data[i][POS_ID][0] for i in range(len(data))]
    ku_pos_id = [data[i][POS_ID][1] for i in range(len(data))]
    modes_id = [data[i][MODE_ID] for i in range(len(data))]

    return labels, args, preds, props, vocab, word_poses, ku_poses, modes, word_pos_id, ku_pos_id, modes_id


def get_sentences_nodep(type):
    # list_path = Path("../../data/NTC_dataset/listed_{}_rework_181130.pkl".format(type))
    # list_path = Path("../../data/NTC_dataset/listed_{}_190105.pkl".format(type))
    list_path = Path("../data/NTC_dataset/listed_{}.pkl".format(type))
    data = load_pickle(list_path)

    label, labels = [], []
    arg, args = [], []
    pred, preds = [], []
    prop, props = [], []
    word_pos, word_poses = [], []
    ku_pos, ku_poses = [], []
    mode, modes = [], []

    count = 0
    for i in range(len(data)):
        if data[i][6] == count + 1:
            labels.append(label)
            args.append(arg)
            preds.append(pred)
            props.append(prop)
            word_poses.append(word_pos)
            ku_poses.append(ku_pos)
            modes.append(mode)

            label = [data[i][LABEL_ID]]
            arg = [data[i][ARG_ID]]
            pred = [data[i][PRED_ID]]
            prop = [data[i][PROP_ID]]
            word_pos = [data[i][POS_ID][0]]
            ku_pos = [data[i][POS_ID][1]]
            mode = [data[i][MODE_ID]]
            if data[i][PROP_ID] == "dep" and data[i][LABEL_ID] != 3:
                label = [4]
            elif data[i][PROP_ID] != "dep":
                label = [data[i][LABEL_ID]]

            count += 1
        else:
            arg.append(data[i][ARG_ID])
            pred.append(data[i][PRED_ID])
            prop.append(data[i][PROP_ID])
            word_pos.append(data[i][POS_ID][0])
            ku_pos.append(data[i][POS_ID][1])
            mode.append(data[i][MODE_ID])
            if data[i][PROP_ID] == "dep" and data[i][LABEL_ID] != 3:
                label.append(4)
            elif data[i][PROP_ID] != "dep":
                label.append(data[i][LABEL_ID])

    labels.append(label)
    args.append(arg)
    preds.append([data[count][PRED_ID]] * len(args[-1]))
    props.append(prop)
    word_poses.append(word_pos)
    ku_poses.append(ku_pos)
    modes.append(mode)

    labels = np.array([np.array(line) for line in labels])
    args = np.array([np.array(line) for line in args])
    preds = np.array([np.array(line) for line in preds])
    props = np.array([np.array(line) for line in props])
    word_poses = np.array([np.array(line) for line in word_poses])
    ku_poses = np.array([np.array(line) for line in ku_poses])
    modes = np.array([np.array(line) for line in modes])

    vocab = np.array([data[i][ARG_ID] for i in range(len(data))])
    word_pos_id = np.array([data[i][POS_ID][0] for i in range(len(data))])
    ku_pos_id = np.array([data[i][POS_ID][1] for i in range(len(data))])
    modes_id = np.array([data[i][MODE_ID] for i in range(len(data))])

    return labels, args, preds, props, vocab, word_poses, ku_poses, modes, word_pos_id, ku_pos_id, modes_id


def get_sentences_nozero(type):
    # list_path = Path("../../data/NTC_dataset/listed_{}_rework_181130.pkl".format(type))
    # list_path = Path("../../data/NTC_dataset/listed_{}_190105.pkl".format(type))
    list_path = Path("../data/NTC_dataset/listed_{}.pkl".format(type))
    data = load_pickle(list_path)

    label, labels = [], []
    arg, args = [], []
    pred, preds = [], []
    prop, props = [], []
    word_pos, word_poses = [], []
    ku_pos, ku_poses = [], []
    mode, modes = [], []

    count = 0
    for i in range(len(data)):
        if data[i][6] == count + 1:
            labels.append(label)
            args.append(arg)
            preds.append(pred)
            props.append(prop)
            word_poses.append(word_pos)
            ku_poses.append(ku_pos)
            modes.append(mode)

            label = [data[i][LABEL_ID]]
            arg = [data[i][ARG_ID]]
            pred = [data[i][PRED_ID]]
            prop = [data[i][PROP_ID]]
            word_pos = [data[i][POS_ID][0]]
            ku_pos = [data[i][POS_ID][1]]
            mode = [data[i][MODE_ID]]
            if data[i][PROP_ID] == "zero" and data[i][LABEL_ID] != 3:
                label = [4]
            elif data[i][PROP_ID] != "zero":
                label = [data[i][LABEL_ID]]

            count += 1
        else:
            arg.append(data[i][ARG_ID])
            pred.append(data[i][PRED_ID])
            prop.append(data[i][PROP_ID])
            word_pos.append(data[i][POS_ID][0])
            ku_pos.append(data[i][POS_ID][1])
            mode.append(data[i][MODE_ID])
            if data[i][PROP_ID] == "zero" and data[i][LABEL_ID] != 3:
                label.append(4)
            elif data[i][PROP_ID] != "zero":
                label.append(data[i][LABEL_ID])

    labels.append(label)
    args.append(arg)
    preds.append([data[count][PRED_ID]] * len(args[-1]))
    props.append(prop)
    word_poses.append(word_pos)
    ku_poses.append(ku_pos)
    modes.append(mode)

    labels = np.array([np.array(line) for line in labels])
    args = np.array([np.array(line) for line in args])
    preds = np.array([np.array(line) for line in preds])
    props = np.array([np.array(line) for line in props])
    word_poses = np.array([np.array(line) for line in word_poses])
    ku_poses = np.array([np.array(line) for line in ku_poses])
    modes = np.array([np.array(line) for line in modes])

    vocab = np.array([data[i][ARG_ID] for i in range(len(data))])
    word_pos_id = np.array([data[i][POS_ID][0] for i in range(len(data))])
    ku_pos_id = np.array([data[i][POS_ID][1] for i in range(len(data))])
    modes_id = np.array([data[i][MODE_ID] for i in range(len(data))])

    return labels, args, preds, props, vocab, word_poses, ku_poses, modes, word_pos_id, ku_pos_id, modes_id


def get_datasets_in_sentences_test(type, with_bccwj=False, with_bert=False):
    # list_path = Path("../../data/NTC_dataset/listed_{}_rework_181130.txt".format(type))
    # list_path = Path("../../data/NTC_dataset/listed_{}_190105.txt".format(type))
    if with_bccwj:
        type += "_bccwj"
        if with_bert:
            type += "_bert"
        if "train2" in type:
            type = type.replace("train2", "train")
        list_path = Path("../data/BCCWJ-DepParaPAS/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/BCCWJ-DepParaPAS-3.3.0_1.2.0_20160301/ntc/listed_{}.txt".format(type))
    else:
        list_path = Path("../data/NTC_dataset/listed_{}.txt".format(type))
    with list_path.open('r', encoding="utf-8") as f:
        data = f.readlines()
    ret = []
    for line in data:
        line = line.strip().split(", ")
        line[4] = line[4][1:]
        line[5] = line[5][:-1]
        if len(line) == 9:
            ret.append([int(line[0]), line[1], line[2], line[3], (int(line[4]), int(line[5])), line[6], int(line[7]), int(line[8])])
        else:
            print(line)
    data = ret

    label, labels = [], []
    arg, args = [], []
    pred, preds = [], []
    prop, props = [], []
    word_pos, word_poses = [], []
    ku_pos, ku_poses = [], []
    mode, modes = [], []

    count = 0
    bef_count = 0
    for i in range(len(data)):
        if data[i][6] != bef_count:
            labels.append(label)
            args.append(arg)
            preds.append(pred)
            props.append(prop)
            word_poses.append(word_pos)
            ku_poses.append(ku_pos)
            modes.append(mode)

            label = [data[i][LABEL_ID]]
            arg = [data[i][ARG_ID]]
            pred = [data[i][PRED_ID]]
            prop = [data[i][PROP_ID]]
            word_pos = [data[i][POS_ID][0]]
            ku_pos = [data[i][POS_ID][1]]
            mode = [data[i][MODE_ID]]

            bef_count = data[i][6]
            count += 1
        else:
            label.append(data[i][LABEL_ID])
            arg.append(data[i][ARG_ID])
            pred.append(data[i][PRED_ID])
            prop.append(data[i][PROP_ID])
            word_pos.append(data[i][POS_ID][0])
            ku_pos.append(data[i][POS_ID][1])
            mode.append(data[i][MODE_ID])

    labels.append(label)
    args.append(arg)
    preds.append([data[count][PRED_ID]] * len(args[-1]))
    props.append(prop)
    word_poses.append(word_pos)
    ku_poses.append(ku_pos)
    modes.append(mode)

    labels = np.array([np.array(line) for line in labels])
    args = np.array([np.array(line) for line in args])
    preds = np.array([np.array(line) for line in preds])
    props = np.array([np.array(line) for line in props])
    word_poses = np.array([np.array(line) for line in word_poses])
    ku_poses = np.array([np.array(line) for line in ku_poses])
    modes = np.array([np.array(line) for line in modes])

    vocab = np.array([data[i][ARG_ID] for i in range(len(data))])
    word_pos_id = np.array([data[i][POS_ID][0] for i in range(len(data))])
    ku_pos_id = np.array([data[i][POS_ID][1] for i in range(len(data))])
    modes_id = np.array([data[i][MODE_ID] for i in range(len(data))])

    return labels, args, preds, props, vocab, word_poses, ku_poses, modes, word_pos_id, ku_pos_id, modes_id


def get_sentences_nodep_test(type):
    # list_path = Path("../../data/NTC_dataset/listed_{}_rework_181130.txt".format(type))
    # list_path = Path("../../data/NTC_dataset/listed_{}_190105.txt".format(type))
    list_path = Path("../data/NTC_dataset/listed_{}.txt".format(type))
    with list_path.open('r', encoding="utf-8") as f:
        data = f.readlines()
    ret = []
    for line in data:
        line = [re.sub("\(|\)|\,", '', item) for item in line.strip().split()]
        if len(line) == 9:
            ret.append([int(line[0]), line[1], line[2], line[3], (int(line[4]), int(line[5])), line[6], int(line[7]), int(line[8])])
    data = ret

    label, labels = [], []
    arg, args = [], []
    pred, preds = [], []
    prop, props = [], []
    word_pos, word_poses = [], []
    ku_pos, ku_poses = [], []
    mode, modes = [], []

    count = 0
    for i in range(len(data)):
        if data[i][6] == count + 1:
            labels.append(label)
            args.append(arg)
            preds.append(pred)
            props.append(prop)
            word_poses.append(word_pos)
            ku_poses.append(ku_pos)
            modes.append(mode)

            label = [data[i][LABEL_ID]]
            arg = [data[i][ARG_ID]]
            pred = [data[i][PRED_ID]]
            prop = [data[i][PROP_ID]]
            word_pos = [data[i][POS_ID][0]]
            ku_pos = [data[i][POS_ID][1]]
            mode = [data[i][MODE_ID]]

            if data[i][PROP_ID] == "dep" and data[i][LABEL_ID] != 3:
                label = [4]
            elif data[i][PROP_ID] != "dep":
                label = [data[i][LABEL_ID]]


            count += 1
        else:
            arg.append(data[i][ARG_ID])
            pred.append(data[i][PRED_ID])
            prop.append(data[i][PROP_ID])
            word_pos.append(data[i][POS_ID][0])
            ku_pos.append(data[i][POS_ID][1])
            mode.append(data[i][MODE_ID])
            if data[i][PROP_ID] == "dep" and data[i][LABEL_ID] != 3:
                label.append(4)
            elif data[i][PROP_ID] != "dep":
                label.append(data[i][LABEL_ID])

    labels.append(label)
    args.append(arg)
    preds.append([data[count][PRED_ID]] * len(args[-1]))
    props.append(prop)
    word_poses.append(word_pos)
    ku_poses.append(ku_pos)
    modes.append(mode)

    labels = np.array([np.array(line) for line in labels])
    args = np.array([np.array(line) for line in args])
    preds = np.array([np.array(line) for line in preds])
    props = np.array([np.array(line) for line in props])
    word_poses = np.array([np.array(line) for line in word_poses])
    ku_poses = np.array([np.array(line) for line in ku_poses])
    modes = np.array([np.array(line) for line in modes])

    vocab = np.array([data[i][ARG_ID] for i in range(len(data))])
    word_pos_id = np.array([data[i][POS_ID][0] for i in range(len(data))])
    ku_pos_id = np.array([data[i][POS_ID][1] for i in range(len(data))])
    modes_id = np.array([data[i][MODE_ID] for i in range(len(data))])

    return labels, args, preds, props, vocab, word_poses, ku_poses, modes, word_pos_id, ku_pos_id, modes_id


def get_sentences_nozero_test(type):
    # list_path = Path("../../data/NTC_dataset/listed_{}_rework_181130.txt".format(type))
    # list_path = Path("../../data/NTC_dataset/listed_{}_190105.txt".format(type))
    list_path = Path("../data/NTC_dataset/listed_{}.txt".format(type))
    with list_path.open('r', encoding="utf-8") as f:
        data = f.readlines()
    ret = []
    for line in data:
        line = [re.sub("\(|\)|\,", '', item) for item in line.strip().split()]
        if len(line) == 9:
            ret.append([int(line[0]), line[1], line[2], line[3], (int(line[4]), int(line[5])), line[6], int(line[7]), int(line[8])])
    data = ret

    label, labels = [], []
    arg, args = [], []
    pred, preds = [], []
    prop, props = [], []
    word_pos, word_poses = [], []
    ku_pos, ku_poses = [], []
    mode, modes = [], []

    count = 0
    for i in range(len(data)):
        if data[i][6] == count + 1:
            labels.append(label)
            args.append(arg)
            preds.append(pred)
            props.append(prop)
            word_poses.append(word_pos)
            ku_poses.append(ku_pos)
            modes.append(mode)

            label = [data[i][LABEL_ID]]
            arg = [data[i][ARG_ID]]
            pred = [data[i][PRED_ID]]
            prop = [data[i][PROP_ID]]
            word_pos = [data[i][POS_ID][0]]
            ku_pos = [data[i][POS_ID][1]]
            mode = [data[i][MODE_ID]]

            if data[i][PROP_ID] == "zero" and data[i][LABEL_ID] != 3:
                label = [4]
            elif data[i][PROP_ID] != "zero":
                label = [data[i][LABEL_ID]]

            count += 1
        else:
            arg.append(data[i][ARG_ID])
            pred.append(data[i][PRED_ID])
            prop.append(data[i][PROP_ID])
            word_pos.append(data[i][POS_ID][0])
            ku_pos.append(data[i][POS_ID][1])
            mode.append(data[i][MODE_ID])
            if data[i][PROP_ID] == "zero" and data[i][LABEL_ID] != 3:
                label.append(4)
            elif data[i][PROP_ID] != "zero":
                label.append(data[i][LABEL_ID])

    labels.append(label)
    args.append(arg)
    preds.append([data[count][PRED_ID]] * len(args[-1]))
    props.append(prop)
    word_poses.append(word_pos)
    ku_poses.append(ku_pos)
    modes.append(mode)

    labels = np.array([np.array(line) for line in labels])
    args = np.array([np.array(line) for line in args])
    preds = np.array([np.array(line) for line in preds])
    props = np.array([np.array(line) for line in props])
    word_poses = np.array([np.array(line) for line in word_poses])
    ku_poses = np.array([np.array(line) for line in ku_poses])
    modes = np.array([np.array(line) for line in modes])

    vocab = np.array([data[i][ARG_ID] for i in range(len(data))])
    word_pos_id = np.array([data[i][POS_ID][0] for i in range(len(data))])
    ku_pos_id = np.array([data[i][POS_ID][1] for i in range(len(data))])
    modes_id = np.array([data[i][MODE_ID] for i in range(len(data))])

    return labels, args, preds, props, vocab, word_poses, ku_poses, modes, word_pos_id, ku_pos_id, modes_id


def get_one_hot(target_vector, classes):
    return np.eye(classes)[target_vector]


def get_max_index(target_vector):
    return np.argmax(target_vector, axis=1)
