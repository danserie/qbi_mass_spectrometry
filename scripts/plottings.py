#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:14:52 2019

@author: zqwu
"""
import numpy as np
import matplotlib.pyplot as plt
from scripts.parse_data import amino_acid_MWs, generate_mz_chart
    
  
def plot_samples(inputs, trainer):
  labels, preds, IDs = trainer.predict(inputs)
  data = trainer.load_data_from_files(inputs)
  for sample_ID in inputs:
    assert sample_ID in IDs
    y = data[sample_ID][2]
    ind = IDs.index(sample_ID)
    assert np.allclose(y, labels[ind].reshape(y.shape))
    y_pred = preds[ind].reshape(y.shape)
    
    mz_chart = generate_mz_chart(data[sample_ID][0],
                                 data[sample_ID][1][0],
                                 data[sample_ID][1][1])
    
    bars = [(mz_chart[i, j, k], y[i, j, k]) for i, j, k in zip(*np.where(y > 0.01))]
    bars_pred = [(mz_chart[i, j, k], y_pred[i, j, k]) \
                  for i, j, k in zip(*np.where(y_pred > 0.01))]
    bars = np.array(bars)
    bars_pred = np.array(bars_pred)
    plt.clf()
    fig, ax = plt.subplots()
    ax.bar(bars_pred[:, 0], bars_pred[:, 1], width=1., label="pred", color='b')
    ax.bar(bars[:, 0], -bars[:, 1], width=2, label="ground truth", color='r')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.set_xlabel('m/z')
    ax.set_ylabel('ratio')
    plt.legend()
    plt.savefig(str(sample_ID) + '.png', dpi=300)
