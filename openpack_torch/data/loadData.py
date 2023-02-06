import os
import shutil
from pathlib import Path

import openpack_toolkit as optk
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf, open_dict
from openpack_toolkit import OPENPACK_OPERATIONS


def ObtenerDataSet(
    cfg: DictConfig,
    nameActivitiesFile: str,
    nameDataset: str = "default",
    classes: optk.ActSet = OPENPACK_OPERATIONS,
):
    data_dir = cfg.datadir
    data_dir_freq = cfg.data_dir_freq
    # create training dir
    training_dir = os.path.join(data_dir, "training")
    if not os.path.isdir(training_dir):
        os.mkdir(training_dir)

    # create dataset training
    trainDir = os.path.join(training_dir, nameDataset)
    if not os.path.isdir(trainDir):
        os.mkdir(trainDir)

    # create validation dir
    validation_dir = os.path.join(data_dir, "validation")
    if not os.path.isdir(validation_dir):
        os.mkdir(validation_dir)

    # create dataset in validation
    valDir = os.path.join(validation_dir, nameDataset)
    if not os.path.isdir(valDir):
        os.mkdir(valDir)

    # create test dir
    test_dir = os.path.join(data_dir, "test")
    if not os.path.isdir(test_dir):
        os.mkdir(test_dir)

    # create dataset in test
    testDir = os.path.join(test_dir, nameDataset)
    if not os.path.isdir(testDir):
        os.mkdir(testDir)

    # create submission dir
    submission_dir = os.path.join(data_dir, "submission")
    if not os.path.isdir(submission_dir):
        os.mkdir(submission_dir)

    # create dataset in submission
    submissionDir = os.path.join(submission_dir, nameDataset)
    if not os.path.isdir(submissionDir):
        os.mkdir(submissionDir)

    split = cfg.dataset.split

    # TRAIN DATA
    pathTrain = f"{data_dir_freq}/{cfg.sampling}/train/"

    for dirpath, dirnames, filenames in os.walk(pathTrain):
        for filename in filenames:
            if ".jpg" in filename.lower():
                if not os.path.exists(os.path.join(trainDir, filename)):
                    shutil.copy(os.path.join(dirpath, filename), trainDir)

    labelsTrain = pd.read_csv(
        f"{data_dir_freq}/{cfg.sampling}/train/{nameActivitiesFile}")

    # VALIDATION DATA
    pathVal = f"{data_dir_freq}/{cfg.sampling}/val/"

    for dirpath, dirnames, filenames in os.walk(pathVal):
        for filename in filenames:
            if ".jpg" in filename.lower():
                if not os.path.exists(os.path.join(valDir, filename)):
                    shutil.copy(os.path.join(dirpath, filename), valDir)
    labelsVal = pd.read_csv(
        f"{data_dir_freq}/{cfg.sampling}/val/{nameActivitiesFile}")

    # TEST DATA
    pathTest = f"{data_dir_freq}/{cfg.sampling}/test/"

    for dirpath, dirnames, filenames in os.walk(pathTest):
        for filename in filenames:
            if ".jpg" in filename.lower():
                if not os.path.exists(os.path.join(testDir, filename)):
                    shutil.copy(os.path.join(dirpath, filename), testDir)
    labelsTest = pd.read_csv(
        f"{data_dir_freq}/{cfg.sampling}/test/{nameActivitiesFile}")

    # SUBMISSION DATA
    pathSubmission = f"{data_dir_freq}/{cfg.sampling}/submission/"

    for dirpath, dirnames, filenames in os.walk(pathSubmission):
        for filename in filenames:
            if ".jpg" in filename.lower():
                if not os.path.exists(os.path.join(submissionDir, filename)):
                    shutil.copy(os.path.join(dirpath, filename), submissionDir)
    labelsSubmission = pd.read_csv(
        f"{data_dir_freq}/{cfg.sampling}/submission/{nameActivitiesFile}")

    return trainDir, labelsTrain, valDir, labelsVal, testDir, labelsTest, submissionDir, labelsSubmission
