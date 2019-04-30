#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:14:52 2019

@author: zqwu
"""
import numpy as np
import matplotlib.pyplot as plt
from scripts.parse_data import amino_acid_MWs, generate_mz_chart, calculate_precursor_MZ, reverse_amino_acid_coding

neutral_losses = {0: 0, 1: 17, 2: 18, 3: 34, 4: 35}

def generate_text_string(pair, pred):
  pred = pred.reshape(pair[2].shape)
  lines = []
  name = reverse_amino_acid_coding(pair[0], pair[1][1], pair[1][0])
  lines.append('Name: ' + name)
  lines.append('LibID: -1')

  mz = calculate_precursor_MZ(pair[0], pair[1][0], pair[1][1]) 
  lines.append('MW: %.2f' % (mz * pair[1][1]))
  lines.append('PrecursorMZ: %.2f' % mz)
  lines.append('Status: Normal')
  lines.append('FullName: X.%s.X/%d (HCD)' % (name[:-2], pair[1][1]))
  lines.append('Comment: predictions')

  peak_positions = np.where(pred > 0.01)
  mz_chart = generate_mz_chart(pair[0], pair[1][0], pair[1][1])

  mzs = [mz_chart[d1, d2, d3] for d1, d2, d3 in zip(*peak_positions)]
  signals = [pred[d1, d2, d3]*20000. for d1, d2, d3 in zip(*peak_positions)]
  signs = []
  for d1, d2, d3 in zip(*peak_positions):
    if d2 < 4:
      ion_type = 'b'
      charge = d2 + 1
      position = d1 + 1
    else:
      ion_type = 'y'
      charge = d2 - 4 + 1
      position = pair[0].shape[0] - d1 - 1
    neutral_loss = neutral_losses[d3]
    sign = ion_type + str(position)
    if neutral_loss > 0:
      sign += '-' + str(neutral_loss)
    if charge > 1:
      sign += '^' + str(charge)
    signs.append(sign)

  lines.append('NumPeaks: %d' % len(mzs))
  order = np.argsort(mzs)
  mzs = np.array(mzs)[order]
  signals = np.array(signals)[order]
  signs = np.array(signs)[order]

  for mz, signal, sign in zip(mzs, signals, signs):
    lines.append('%.2f\t%.2f\t%s' % (mz, signal, sign))
  return lines    
  
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
