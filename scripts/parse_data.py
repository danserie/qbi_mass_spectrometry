import csv
import re
import pickle
import numpy as np
from collections import Counter

beginning_modifier_regex = r"^[a-z]\[\d+\]"
middle_modifier_regex = r"[A-Z]\[\d+\]"
position_regex = r"^[by](?P<position>\d+)"
charge_regex = r"\^(?P<charge>[\d])+"
neutral_loss_regex = r"[by][\d]+-(?P<neutral_loss>[\d]+)"
neutral_gain_regex = r"[by][\d]+[+](?P<neutral_loss>[\d]+)"
delta_regex = r"\/(?P<delta>-?[0-9]\d*\.*\d*)$"
indicator_codes = "by"
csv_output_rows = [
    'acetyl',
    'name',
    'charge',
    'precursor',
    'mz',
    'intensity',
    'ion',
    'position',
    'neutral_loss',
    'ion_charge',
    'delta',
]

beginning_modifier_replacements = {
    "n[43]": 1
}

amino_acid_modifier_replacements = {
    "C[160]": "!",
    "M[147]": "@",
    "Q[129]": "E", #deamidation of asparagine
    "N[115]": "D", #deamidation of glutamine
    "Q[110]": "#",
#    "Q[111]": "#",
    "S[167]": "$",
    "T[181]": "%",
    "Y[243]": "^",  
    "M[0]":  "",
    "M[42]": "n[43]",
    "M[173]": "n[43]M",
    "M[189]": "n[43]@"
}

amino_acid_codes = "ACDEFGHIKLMNPQRSTVWY"
amino_acid_modifiers = "!@#$%^"
amino_acid_modified_codes = amino_acid_codes+amino_acid_modifiers

amino_acid_MWs = np.array([ 71.0371, 103.0092, 115.0269, 129.0426, 147.0684, #ACDEF
                            57.0215, 137.0589, 113.0841, 128.0950, 113.0841, #GHIKL
                           131.0405, 114.0429,  97.0528, 128.0586, 156.1011, #MNPQR
                            87.0320, 101.0477,  99.0684, 186.0793, 163.0633, #STVWY
                           160.0306, 147.0354, 110.0480, 166.9984, 181.0140, 243.0297]) #!@#$%^

neutral_loss_choices = [0, 17, 18, 34, 35]
neutral_loss_mass = {
    0: 0.,
    17: 17.0265,
    18: 18.0106,
    34: 34.0531,
    35: 35.0371}

def one_hot_encode(values, code):
  result = []
  for letter in values:
    letter_encoding = [1 if code_letter == letter else 0 for code_letter in code]
    try:
      assert (sum(letter_encoding) == 1)
    except AssertionError:
      raise ValueError("One hot encoding failed - unexpected letter in this name %s" % str(values))
    result.append(letter_encoding)
  return result


def reverse_one_hot_encode(vectors, code=amino_acid_modified_codes):
  letters = []
  for vector in vectors:
    i = np.argmax(vector)  # get the index of the item which is 1
    letters.append(code[i])
  return "".join(letters)

# reverse_one_hot_encode(one_hot_encode(values, code), code) should be values.

def amino_acid_name_parse(name_str):
  """
  :param name_str: name string with beginning charge, amino acids/modifiers, and charge.
  :return: has_beginning_charge, name, charge
  """
  name, charge = name_str.split("/")
  charge = int(charge)

  for modifier, modifier_code in amino_acid_modifier_replacements.items():
    # Replace modifiers with our special codes
    name = name.replace(modifier, modifier_code)

  beginning_modifier_search = re.search(beginning_modifier_regex, name)

  flag = 0
  if beginning_modifier_search:
    beginning_modifier_match = beginning_modifier_search.group(0)
    if beginning_modifier_match == "n[43]":
      flag = beginning_modifier_replacements[beginning_modifier_match]
      name = name.replace("n[43]", "")
    elif beginning_modifier_match == "n[44]":
      #print("n[44] encountered in: %s" % name_str)
      return None
    else:
      print("Error parsing name: %s" % name_str)
      return None

  if len(set(name) - set(amino_acid_modified_codes)) > 0:
    print("Unidentified token in %s" % name_str)
    return None
  name_one_hot_encoded = one_hot_encode(name, amino_acid_modified_codes)
  return flag, np.array(name_one_hot_encoded), charge


def reverse_amino_acid_coding(vectors, charge, has_beginning_modifier=False):
  """
  Reverse one hot encoding and character substitutions for amino acids.
  :param vectors: one hot encoded vectors
  :param has_beginning_modifier: If it should have the n[43] in front
  :return: original amino acid name string
  """
  letters = reverse_one_hot_encode(vectors, amino_acid_modified_codes)
  if has_beginning_modifier:
    letters = "n[43]"+letters

  for modifier, code in amino_acid_modifier_replacements.items():
    if len(code) == 1 and code in amino_acid_modifiers:
      letters = letters.replace(code, modifier)

  letters += "/{}".format(charge)
  return letters

def generate_mz_chart(seq, flag, charge):
  mzs = -np.ones((seq.shape[0]-1, 8, 5))
  mass = (seq * np.expand_dims(amino_acid_MWs, 0)).sum(1)
  b_masses = np.cumsum(mass) + 1.0078
  if flag:
    b_masses += 42.0106
  y_masses = np.cumsum(np.flip(mass, 0)) + 18.0106 + 1.0078
  
  for i, ion_charge in enumerate(range(1, 5)):
    for j, neutral_loss in enumerate([0, 17, 18, 34, 35]):
      row = (b_masses[:-1] - neutral_loss_mass[neutral_loss] + 1.0078*(ion_charge - 1))/ion_charge
      mzs[:, i, j] = row
  for i, ion_charge in enumerate(range(1, 5)):
    for j, neutral_loss in enumerate([0, 17, 18, 34, 35]):
      row = np.flip((y_masses[:-1] - neutral_loss_mass[neutral_loss] + 1.0078*(ion_charge - 1))/ion_charge)
      mzs[:, i+4, j] = row
  return mzs

def calculate_precursor_MZ(seq, flag, charge):
  mass = (seq * np.expand_dims(amino_acid_MWs, 0)).sum(1)
  full_mass = mass.sum() + 18.0106 + 1.0078 * charge
  if flag:
    full_mass += 42.0106
  return full_mass/charge
    
def read_all_tokens(file_name):
  f = open(file_name, 'r')
  tokens = {}
  current_chunk = []
  for line in f:
    if line.strip():
      if line.startswith('###'): continue
      current_chunk.append(line.strip())
    else:
      status = current_chunk[4].split()[1]
      if status!="Normal":
        print("NOT NORMAL")
        current_chunk = []
    
      name_str = current_chunk[0].split(" ")[1]
      n_ = name_str[:]
      while n_[0] != '/':
        if n_[1] != '[':
          token = n_[0]
          n_ = n_[1:]
        else:
          token = n_[:(n_.find(']') + 1)]
          n_ = n_[(n_.find(']') + 1):]
        if token in tokens:
          tokens[token] += 1
        else:
          tokens[token] = 1
      current_chunk =[]
  f.close()
  
  return tokens

def parse_chunk(current_chunk):
  status = current_chunk[4].split()[1]
  if status!="Normal":
    print("NOT NORMAL")
    return None

  name_str = current_chunk[0].split(" ")[1]
  res = amino_acid_name_parse(name_str)
  if not res is None:
    flag, name, charge = res
  else:
    return None
  
  # Check precursor mz
  precursor_mz_line = current_chunk[3]
  if not precursor_mz_line.startswith("PrecursorMZ: "):
    print("Chunk format error, missing precursorMZ line")
    return None
  precursor_mz = float(precursor_mz_line.split()[1])
  assert np.abs(calculate_precursor_MZ(name, flag, charge) - precursor_mz) < 0.1

  num_peaks_line = current_chunk[7]
  if not num_peaks_line.startswith("NumPeaks: "):
    print("Chunk format error, missing num peaks line")
    return None
  num_peaks = int(num_peaks_line.split()[1])
  if not len(current_chunk[8:]) == num_peaks:
    print("Chunk format error, num peaks unmatched")
    return None
  
  mzs = []
  intensities = []
  ions = []
  positions = []
  ion_charges = []
  neutral_losses = []
  deltas = []

  mass = (name * np.expand_dims(amino_acid_MWs, 0)).sum(1)
  # Ion masses references
  b_masses = np.cumsum(mass) + 1.0078
  if flag:
    b_masses += 42.0106
  y_masses = np.cumsum(np.flip(mass)) + 18.0106 + 1.0078

  for data in current_chunk[8:]:
    mz, intensity, ion_data = data.split()
    ion_data = ion_data.split(",")[0]  # Only care about first item
    if ion_data[0] not in indicator_codes:
      continue  # Skip this row
    if "i" in ion_data:
      continue
    
    # b or y
    ion_indicator_code = indicator_codes.index(ion_data[0])
    
    # charge
    charge_search = re.search(charge_regex, ion_data)
    if charge_search:
      ion_charge = int(charge_search.group("charge"))
    else:
      ion_charge = 1

    # Neutral loss
    neutral_loss_search = re.search(neutral_loss_regex, ion_data)
    neutral_gain_search = re.search(neutral_gain_regex, ion_data)
    if neutral_loss_search:
      neutral_loss = int(neutral_loss_search.group("neutral_loss"))
    elif neutral_gain_search:
      neutral_loss = -int(neutral_gain_search.group("neutral_loss"))
    else:
      neutral_loss = 0
      
    # position
    if ion_indicator_code == 0: # b ion
      position = np.argmin(np.abs((b_masses - neutral_loss + ion_charge - 1)/ion_charge - float(mz))) + 1
      assert np.abs((b_masses[position-1]- neutral_loss + ion_charge - 1)/ion_charge - float(mz)) < 0.2
    elif ion_indicator_code == 1: # y ion
      position = np.argmin(np.abs((y_masses - neutral_loss + ion_charge - 1)/ion_charge - float(mz))) + 1
      assert np.abs((y_masses[position-1]- neutral_loss + ion_charge - 1)/ion_charge - float(mz)) < 0.2

    position_search = re.search(position_regex, ion_data)
    position2 = int(position_search.group("position"))
    if position != position2:
      if not name_str.startswith('M[42]') and not name_str.startswith('M[0]'): # Common cause
        print("Position difference in sample %s\n%s" % (name_str, data))

    delta = float(ion_data.split("/")[1])
    delta = round(delta, 4)

    mzs.append(round(float(mz), 4))
    intensities.append(round(float(intensity), 4))
    ions.append(ion_indicator_code)
    positions.append(position)
    ion_charges.append(ion_charge)
    neutral_losses.append(neutral_loss)
    deltas.append(delta)

  assert len(mzs)==len(intensities)==len(ions)==len(neutral_losses)==len(ion_charges)==len(deltas)
  output = {
      'acetyl': flag,
      'name': name,
      'charge': charge,
      'precursor': precursor_mz,
      'mz': mzs,
      'intensity': intensities,
      'ion': ions,
      'position': positions,
      'neutral_loss': neutral_losses,
      'ion_charge': ion_charges,
      'delta': deltas,
      'raw': current_chunk
  }
  return output

def neutral_loss_type(n_loss):
  if n_loss == 0:
    return 0
  elif n_loss == 17:
    return 1 # loss of ammonia
  elif n_loss == 18:
    return 2 # loss of water
  elif n_loss == 35:
    return 3 # loss of ammonia + water
  elif n_loss == 36:
    return 4 # loss of 2 water
  else:
    return None
  

def make_table(output):
  peptide_length = output['name'].shape[0]
  n_charges = 4
  table = np.zeros((peptide_length-1, n_charges*2, 5))
  for i in range(len(output['mz'])):
    d3 = neutral_loss_type(output['neutral_loss'][i])
    charge = output['ion_charge'][i]
    if charge > n_charges or d3 is None:
      continue
    if output['ion'][i] == 0:
      d1 = output['position'][i] - 1
      d2 = charge - 1
    else:
      d1 = peptide_length - 1 - output['position'][i]
      d2 = charge - 1 + n_charges
    table[d1, d2, d3] += output['intensity'][i]
  table = table/table.sum()
  return table

if __name__ == "__main__":
  output_file = "/hd/trainingData/massiveKB.pkl"
  spec_data_file = open("/hd/trainingData/massiveKB_noSynPurged.sptxt", "r")
  j = 0
 
  samples = []
  current_chunk = []
  for line in spec_data_file:
    if line.strip():
      if line.startswith('###'): continue
      current_chunk.append(line.strip())
    else:
      sample = parse_chunk(current_chunk)
      if not sample is None:
        samples.append(sample)
      current_chunk = []
      if len(samples) > 0 and len(samples) % 10000 == 0:
        print("Finished %d samples" % len(samples))
        with open(output_file + str(j), 'wb') as f:
          pickle.dump(samples, f)
        j += 1
        samples = []

  if len(current_chunk) > 5:
    sample = parse_chunk(current_chunk)
    if not sample is None:
      samples.append(sample)
  with open(output_file + str(j), 'wb') as f:
    pickle.dump(samples, f)
    
  ################################################################
  output_file = "/hd/trainingData/massiveKB.pkl"
  pairs = {}
  raws = {}
  for i in range(50):
    pairs[i] = []
    raws[i] = []
  j = 171
  for f_n in range(j+1):
    f_name = output_file + str(f_n)
    dat = pickle.load(open(f_name, 'rb'))
    for d in dat:
      X = d['name']
      X_meta = (d['acetyl'], d['charge'])
      y = make_table(d)
      if not np.all(y == y):
        continue
      pairs[X.shape[0]].append((X, X_meta, y))
      raws[X.shape[0]].append(d['raw'])

  total_pairs = []
  all_raws = []
  for i in range(50):
    total_pairs.extend(pairs[i])
    all_raws.extend(raws[i])
  
  j = 0
  metadata = {}
  to_be_saved = {}
  raw_saved = {}
  for i, p in enumerate(total_pairs):
    if len(to_be_saved) >= 400:
      np.save('/hd/data/massiveKB%d' % j, to_be_saved)
      np.save('/hd/data/massiveKB_raw%d' % j, raw_saved)
      to_be_saved = {}
      raw_saved = {}
      j += 1
      
    to_be_saved[i] = p
    raw_saved[i] = all_raws[i]
    metadata[i] = ['/hd/data/massiveKB%d.npy' % j, p[0].shape[0]]

  np.save('/hd/data/massiveKB%d' % j, to_be_saved)
  np.save('/hd/data/massiveKB_raw%d' % j, raw_saved)
  
  np.save('/hd/data/massiveKB_meta', metadata)

