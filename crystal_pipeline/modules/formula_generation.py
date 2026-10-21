import csv
from pathlib import Path
from typing import Optional, Dict, Union, Literal, List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml
from transformers import GPT2Config, GPT2Model
from pymatgen.core.composition import Composition

import json
from collections import defaultdict
from transformers import set_seed
from .common import save_json

import numpy as np


RAW_ELECTRON_CONFIG = {
  #          Z 1s 2s 2p 3s 3p 4s  3d 4p 5s  4d 5p 6s  4f  5d 6p vs vp  vd  vf
  "START": [0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "H0":   [  1, 1, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # H
  "H1+":  [  1, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "H1-":  [  1, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0],
  "Li0":  [  3, 2, 1, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Li
  "Li1+": [  3, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Be0":  [  4, 2, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Be
  "Be2+": [  4, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "B0":   [  5, 2, 2, 1, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 1,  0,  0], # B
  "B3+":  [  5, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "C0":   [  6, 2, 2, 2, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 2,  0,  0], # C
  "C1+":  [  6, 2, 2, 1, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 1,  0,  0],
  "C2+":  [  6, 2, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0],
  "C3+":  [  6, 2, 1, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0],
  "C4+":  [  6, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "C4-":  [  6, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0],
  "C3-":  [  6, 2, 2, 5, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 5,  0,  0],
  "C2-":  [  6, 2, 2, 4, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 4,  0,  0],
  "C1-":  [  6, 2, 2, 3, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 3,  0,  0],
  "N5+":  [  7, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0], # N
  "N3+":  [  7, 2, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0],
  "N3-":  [  7, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0],
  "O2-":  [  8, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0], # O
  "F1-":  [  9, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0], # F
  "Na0":  [ 11, 2, 2, 6, 1, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Na
  "Na1+": [ 11, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Mg0":  [ 12, 2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Mg
  "Mg2+": [ 12, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Al0":  [ 13, 2, 2, 6, 2, 1, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 1,  0,  0], # Al
  "Al3+": [ 13, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Si0":  [ 14, 2, 2, 6, 2, 2, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 2,  0,  0], # Si
  "Si4+": [ 14, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Si4-": [ 14, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0],
  "P0":   [ 15, 2, 2, 6, 2, 3, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 3,  0,  0], # P
  "P5+":  [ 15, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "P3+":  [ 15, 2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0],
  "P1+":  [ 15, 2, 2, 6, 2, 2, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 2,  0,  0],
  "P3-":  [ 15, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0],
  "S0":   [ 16, 2, 2, 6, 2, 4, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 4,  0,  0], # S
  "S2-":  [ 16, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0],
  "S2+":  [ 16, 2, 2, 6, 2, 2, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 2,  0,  0],
  "S4+":  [ 16, 2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0],
  "S6+":  [ 16, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Cl1-": [ 17, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0], # Cl
  "Cl1+": [ 17, 2, 2, 6, 2, 4, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 4,  0,  0],
  "Cl5+": [ 17, 2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0],
  "Cl7+": [ 17, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "K0":   [ 19, 2, 2, 6, 2, 6, 1,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # K
  "K1+":  [ 19, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Ca0":  [ 20, 2, 2, 6, 2, 6, 2,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Ca
  "Ca2+": [ 20, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Sc0":  [ 21, 2, 2, 6, 2, 6, 2,  1, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  1,  0], # Sc
  "Sc3+": [ 21, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Ti0":  [ 22, 2, 2, 6, 2, 6, 2,  2, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  2,  0], # Ti
  "Ti4+": [ 22, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Ti3+": [ 22, 2, 2, 6, 2, 6, 0,  1, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  1,  0],
  "Ti2+": [ 22, 2, 2, 6, 2, 6, 0,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  2,  0],
  "V0":   [ 23, 2, 2, 6, 2, 6, 2,  3, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  3,  0], # V
  "V2+":  [ 23, 2, 2, 6, 2, 6, 0,  3, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  3,  0],
  "V3+":  [ 23, 2, 2, 6, 2, 6, 0,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  2,  0],
  "V4+":  [ 23, 2, 2, 6, 2, 6, 0,  1, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  1,  0],
  "V5+":  [ 23, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Cr0":  [ 24, 2, 2, 6, 2, 6, 1,  5, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  5,  0], # Cr
  "Cr2+": [ 24, 2, 2, 6, 2, 6, 0,  4, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  4,  0],
  "Cr3+": [ 24, 2, 2, 6, 2, 6, 0,  3, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  3,  0],
  "Cr4+": [ 24, 2, 2, 6, 2, 6, 0,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  2,  0],
  "Cr6+": [ 24, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Mn0":  [ 25, 2, 2, 6, 2, 6, 2,  5, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  5,  0], # Mn
  "Mn2+": [ 25, 2, 2, 6, 2, 6, 0,  5, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  5,  0],
  "Mn3+": [ 25, 2, 2, 6, 2, 6, 0,  4, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  4,  0],
  "Mn4+": [ 25, 2, 2, 6, 2, 6, 0,  3, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  3,  0],
  "Mn5+": [ 25, 2, 2, 6, 2, 6, 0,  2, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  2,  0],
  "Mn6+": [ 25, 2, 2, 6, 2, 6, 0,  1, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  1,  0],
  "Mn7+": [ 25, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Fe0":  [ 26, 2, 2, 6, 2, 6, 2,  6, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  6,  0], # Fe
  "Fe2+": [ 26, 2, 2, 6, 2, 6, 0,  6, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  6,  0],
  "Fe3+": [ 26, 2, 2, 6, 2, 6, 0,  5, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  5,  0],
  "Fe4+": [ 26, 2, 2, 6, 2, 6, 0,  4, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  4,  0],
  "Co0":  [ 27, 2, 2, 6, 2, 6, 2,  7, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  7,  0], # Co
  "Co2+": [ 27, 2, 2, 6, 2, 6, 0,  7, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  7,  0],
  "Co3+": [ 27, 2, 2, 6, 2, 6, 0,  6, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  6,  0],
  "Co4+": [ 27, 2, 2, 6, 2, 6, 0,  5, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  5,  0],
  "Ni0":  [ 28, 2, 2, 6, 2, 6, 2,  8, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  8,  0], # Ni
  "Ni2+": [ 28, 2, 2, 6, 2, 6, 0,  8, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  8,  0],
  "Ni3+": [ 28, 2, 2, 6, 2, 6, 0,  7, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  7,  0],
  "Ni4+": [ 28, 2, 2, 6, 2, 6, 0,  6, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  6,  0],
  "Cu0":  [ 29, 2, 2, 6, 2, 6, 1, 10, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0, 10,  0], # Cu
  "Cu1+": [ 29, 2, 2, 6, 2, 6, 0, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Cu2+": [ 29, 2, 2, 6, 2, 6, 0,  9, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  9,  0], # Zn
  "Cu3+": [ 29, 2, 2, 6, 2, 6, 0,  8, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  8,  0],
  "Cu4+": [ 29, 2, 2, 6, 2, 6, 0,  7, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  7,  0],
  "Zn0":  [ 30, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0], # Zn
  "Zn2+": [ 30, 2, 2, 6, 2, 6, 0, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Ga0":  [ 31, 2, 2, 6, 2, 6, 2, 10, 1, 0,  0, 0, 0,  0,  0, 0, 2, 1, 10,  0], # Ga
  "Ga3+": [ 31, 2, 2, 6, 2, 6, 0, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Ge0":  [ 32, 2, 2, 6, 2, 6, 2, 10, 2, 0,  0, 0, 0,  0,  0, 0, 2, 2, 10,  0], # Ge
  "Ge2+": [ 32, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0],
  "Ge4+": [ 32, 2, 2, 6, 2, 6, 0, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Ge4-": [ 32, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 2, 6, 10,  0],
  "As0":  [ 33, 2, 2, 6, 2, 6, 2, 10, 3, 0,  0, 0, 0,  0,  0, 0, 2, 3, 10,  0], # As
  "As5+": [ 33, 2, 2, 6, 2, 6, 0, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "As3+": [ 33, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0],
  "As3-": [ 33, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 2, 6, 10,  0],
  "Se0":  [ 34, 2, 2, 6, 2, 6, 2, 10, 4, 0,  0, 0, 0,  0,  0, 0, 2, 4, 10,  0], # Se
  "Se2+": [ 34, 2, 2, 6, 2, 6, 2, 10, 2, 0,  0, 0, 0,  0,  0, 0, 2, 2, 10,  0],
  "Se4+": [ 34, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0],
  "Se6+": [ 34, 2, 2, 6, 2, 6, 0, 10, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Se2-": [ 34, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 2, 6, 10,  0],
  "Br5+": [ 35, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0], # Br
  "Br1+": [ 35, 2, 2, 6, 2, 6, 2, 10, 4, 0,  0, 0, 0,  0,  0, 0, 2, 4, 10,  0],
  "Br1-": [ 35, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 2, 6, 10,  0],
  "Rb0":  [ 37, 2, 2, 6, 2, 6, 2, 10, 6, 1,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Rb
  "Rb1+": [ 37, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Sr0":  [ 38, 2, 2, 6, 2, 6, 2, 10, 6, 2,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Sr
  "Sr2+": [ 38, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Y0":   [ 39, 2, 2, 6, 2, 6, 2, 10, 6, 2,  1, 0, 0,  0,  0, 0, 2, 0,  1,  0], # Y
  "Y3+":  [ 39, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Zr0":  [ 40, 2, 2, 6, 2, 6, 2, 10, 6, 2,  2, 0, 0,  0,  0, 0, 2, 0,  2,  0], # Zr
  "Zr2+": [ 40, 2, 2, 6, 2, 6, 2, 10, 6, 0,  2, 0, 0,  0,  0, 0, 0, 0,  2,  0],
  "Zr3+": [ 40, 2, 2, 6, 2, 6, 2, 10, 6, 0,  1, 0, 0,  0,  0, 0, 0, 0,  1,  0],
  "Zr4+": [ 40, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0], # Zr
  "Nb0":  [ 41, 2, 2, 6, 2, 6, 2, 10, 6, 1,  4, 0, 0,  0,  0, 0, 1, 0,  4,  0], # Nb
  "Nb3+": [ 41, 2, 2, 6, 2, 6, 2, 10, 6, 0,  2, 0, 0,  0,  0, 0, 0, 0,  2,  0],
  "Nb4+": [ 41, 2, 2, 6, 2, 6, 2, 10, 6, 0,  1, 0, 0,  0,  0, 0, 0, 0,  1,  0],
  "Nb5+": [ 41, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Mo0":  [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 1,  5, 0, 0,  0,  0, 0, 1, 0,  5,  0], # Mo
  "Mo2+": [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 0,  4, 0, 0,  0,  0, 0, 0, 0,  4,  0],
  "Mo3+": [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 0,  3, 0, 0,  0,  0, 0, 0, 0,  3,  0],
  "Mo4+": [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 0,  2, 0, 0,  0,  0, 0, 0, 0,  2,  0],
  "Mo5+": [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 0,  1, 0, 0,  0,  0, 0, 0, 0,  1,  0],
  "Mo6+": [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "Ru0":  [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 1,  7, 0, 0,  0,  0, 0, 1, 0,  7,  0], # Ru
  "Ru2+": [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 0,  6, 0, 0,  0,  0, 0, 0, 0,  6,  0],
  "Ru3+": [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 0,  5, 0, 0,  0,  0, 0, 0, 0,  5,  0],
  "Ru4+": [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 0,  4, 0, 0,  0,  0, 0, 0, 0,  4,  0],
  "Ru5+": [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 0,  3, 0, 0,  0,  0, 0, 0, 0,  3,  0],
  "Rh0":  [ 45, 2, 2, 6, 2, 6, 2, 10, 6, 1,  8, 0, 0,  0,  0, 0, 1, 0,  8,  0], # Rh
  "Rh3+": [ 45, 2, 2, 6, 2, 6, 2, 10, 6, 0,  6, 0, 0,  0,  0, 0, 0, 0,  6,  0],
  "Rh4+": [ 45, 2, 2, 6, 2, 6, 2, 10, 6, 0,  5, 0, 0,  0,  0, 0, 0, 0,  5,  0],
  "Rh5+": [ 45, 2, 2, 6, 2, 6, 2, 10, 6, 0,  4, 0, 0,  0,  0, 0, 0, 0,  4,  0],
  "Pd0":  [ 46, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0], # Pd
  "Pd2+": [ 46, 2, 2, 6, 2, 6, 2, 10, 6, 0,  8, 0, 0,  0,  0, 0, 0, 0,  8,  0],
  "Pd4+": [ 46, 2, 2, 6, 2, 6, 2, 10, 6, 0,  6, 0, 0,  0,  0, 0, 0, 0,  6,  0],
  "Ag0":  [ 47, 2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0,  0,  0, 0, 1, 0, 10,  0], # Ag
  "Ag1+": [ 47, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Ag2+": [ 47, 2, 2, 6, 2, 6, 2, 10, 6, 0,  9, 0, 0,  0,  0, 0, 0, 0,  9,  0],
  "Ag3+": [ 47, 2, 2, 6, 2, 6, 2, 10, 6, 0,  8, 0, 0,  0,  0, 0, 0, 0,  8,  0],
  "Cd0":  [ 48, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 2, 0, 10,  0], # Cd
  "Cd2+": [ 48, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "In0":  [ 49, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0,  0,  0, 0, 2, 1, 10,  0], # In
  "In1+": [ 49, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 2, 0, 10,  0],
  "In3+": [ 49, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Sn0":  [ 50, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0,  0,  0, 0, 2, 2, 10,  0], # Sn
  "Sn2+": [ 50, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 2, 0, 10,  0],
  "Sn4+": [ 50, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Sb0":  [ 51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0,  0,  0, 0, 2, 3, 10,  0], # Sb
  "Sb3+": [ 51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 2, 0, 10,  0],
  "Sb5+": [ 51, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Sb3-": [ 51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 2, 6, 10,  0],
  "Te0":  [ 52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0,  0,  0, 0, 2, 4, 10,  0], # Te
  "Te6+": [ 52, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Te4+": [ 52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 2, 0, 10,  0], 
  "Te2-": [ 52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 2, 6, 10,  0],
  "I1-":   [ 53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 2, 6, 10,  0], # I
  "I1+":  [ 53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0,  0,  0, 0, 2, 4, 10,  0],
  "I5+":  [ 53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 2, 0, 10,  0],
  "I7+":  [ 53, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0],
  "Cs0":  [ 55, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1,  0,  0, 0, 1, 0,  0,  0], # Cs
  "Cs1+": [ 55, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 0, 0,  0,  0],
  "Ba0":  [ 56, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  0, 0, 2, 0,  0,  0], # Ba
  "Ba2+": [ 56, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 0, 0,  0,  0],
  "Hf0":  [ 72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  2, 0, 2, 0,  2, 14], # Hf
  "Hf2+": [ 72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  2, 0, 0, 0,  2, 14],
  "Hf3+": [ 72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  1, 0, 0, 0,  1, 14],
  "Hf4+": [ 72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  0, 0, 0, 0,  0, 14],
  "Ta0":  [ 73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  3, 0, 2, 0,  3, 14], # Ta
  "Ta3+": [ 73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  2, 0, 0, 0,  2, 14],
  "Ta4+": [ 73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  1, 0, 0, 0,  1, 14],
  "Ta5+": [ 73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  0, 0, 0, 0,  0, 14],
  "W0":   [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  4, 0, 2, 0,  4, 14], # W
  "W3+":  [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  3, 0, 0, 0,  3, 14],
  "W4+":  [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  2, 0, 0, 0,  2, 14],
  "W5+":  [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  1, 0, 0, 0,  1, 14],
  "W6+":  [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  0, 0, 0, 0,  0, 14],
  "Re0":  [ 75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  5, 0, 2, 0,  5, 14], # Re
  "Re4+": [ 75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  3, 0, 0, 0,  3, 14],
  "Re6+": [ 75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  1, 0, 0, 0,  1, 14],
  "Re7+": [ 75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  0, 0, 0, 0,  0, 14],
  "Os0":  [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  6, 0, 2, 0,  6, 14], # Os
  "Os4+": [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  4, 0, 0, 0,  4, 14],
  "Os5+": [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  3, 0, 0, 0,  3, 14],
  "Os6+": [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  2, 0, 0, 0,  2, 14],
  "Os8+": [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  0, 0, 0, 0,  0, 14],
  "Ir0":  [ 77, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  7, 0, 2, 0,  7, 14], # Ir
  "Ir3+": [ 77, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  6, 0, 0, 0,  6, 14],
  "Ir4+": [ 77, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  5, 0, 0, 0,  5, 14],
  "Pt0":  [ 78, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14,  9, 0, 1, 0,  9, 14], # Pt
  "Pt2+": [ 78, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  8, 0, 0, 0,  8, 14],
  "Pt4+": [ 78, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  6, 0, 0, 0,  6, 14],
  "Au0":  [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 1, 0, 10, 14], # Au
  "Au1+": [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14, 10, 0, 0, 0, 10, 14],
  "Au2+": [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  9, 0, 0, 0,  9, 14],
  "Au3+": [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  8, 0, 0, 0,  8, 14],
  "Au5+": [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14,  6, 0, 0, 0,  6, 14],
  "Hg0":  [ 80, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 2, 0, 10, 14], # Hg
  "Hg1+": [ 80, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 1, 0, 10, 14],
  "Hg2+": [ 80, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14, 10, 0, 0, 0, 10, 14],
  "Tl0":  [ 81, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 2, 1, 10, 14], # Tl
  "Tl1+": [ 81, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 2, 0, 10, 14],
  "Tl3+": [ 81, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14, 10, 0, 0, 0, 10, 14],
  "Pb0":  [ 82, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 2, 2, 10, 14], # Pb
  "Pb2+": [ 82, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 2, 0, 10, 14],
  "Pb4+": [ 82, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14, 10, 0, 0, 0, 10, 14],
  "Bi0":  [ 83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 2, 3, 10, 14], # Bi
  "Bi3+": [ 83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 2, 0, 10, 14],
  "Bi5+": [ 83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0, 14, 10, 0, 0, 0, 10, 14],
  "Bi3-": [ 83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 10, 14],
  "END":  [ 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
  "PAD":  [ 0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0],
}

def get_element_valence(symbol):
    if symbol.endswith("0"):
        return (symbol[:-1], 0)
    elif symbol.endswith("+"):
        return (symbol[:-2], int(symbol[-2]))
    elif symbol.endswith("-"):
        return (symbol[:-2], -int(symbol[-2]))
    else:
        return (symbol, 0)

# normalize entries (between 0.0 and 1.0)
config_max = np.array(list(RAW_ELECTRON_CONFIG.values())).max(axis=0).reshape(1, -1)
special_tokens = ["PAD", "START", "END"]
PURE_SPECIES = sorted([e for e in RAW_ELECTRON_CONFIG.keys() if e not in special_tokens])
PURE_ALLOY_SPECIES = [e for e in PURE_SPECIES if e.endswith("0")]
PURE_IONIC_SPECIES = [e for e in PURE_SPECIES if not e.endswith("0")]
ALLOY_SPECIES = special_tokens + PURE_ALLOY_SPECIES
IONIC_SPECIES = special_tokens + PURE_IONIC_SPECIES
ALLOY_ELECTRONIC_CONFIG = np.array([RAW_ELECTRON_CONFIG[e] for e in ALLOY_SPECIES]) / config_max
IONIC_ELECTRONIC_CONFIG = np.array([RAW_ELECTRON_CONFIG[e] for e in IONIC_SPECIES]) / config_max
IONIC_VALENCES = [0, 0, 0] + [get_element_valence(e)[1] for e in PURE_IONIC_SPECIES]


MAX_COUNT = 20
MAX_LENGTH = 10

class ElementList:
    def __init__(self, elements):
        self.elements = elements
        self.n_elements = len(elements)
        self.element_to_idx = {el: i for i, el in enumerate(elements)}
    
    def __getitem__(self, index):
        return self.elements[index]
    
    def __len__(self):
        return self.n_elements


class CompositionEmbedding(nn.Module):
    def __init__(self,
        dim_embedding: int,
        elements: ElementList, 
        use_electronic_config: bool,
        max_count: int,
        electronic_config: Optional[np.ndarray] = None
    ):
        super().__init__()
        self.elements = elements
        self.max_count = max_count
        self.element_embedding = nn.Embedding(len(elements), dim_embedding)
        self.count_embedding = nn.Embedding(self.max_count + 1, dim_embedding)
        self.use_electronic_config = use_electronic_config
        if use_electronic_config:
            self.register_buffer("electronic_config", torch.tensor(electronic_config).float())
            self.electronic_layer = nn.Linear(self.electronic_config.shape[1], dim_embedding)
    
    def get_electronic_embedding(self, element_tokens: torch.LongTensor):
        return F.embedding(element_tokens, self.electronic_layer(self.electronic_config))

    def forward(self, element_tokens: torch.LongTensor, count_tokens: torch.LongTensor):
        element_embedding = self.element_embedding(element_tokens)
        count_embedding = self.count_embedding(count_tokens)
        composition_embedding = element_embedding + count_embedding
        if self.use_electronic_config:
            composition_embedding += self.get_electronic_embedding(element_tokens)
        return composition_embedding


class GPT2ForChem(nn.Module):
    def __init__(self, 
        config: GPT2Config, 
        elements: ElementList, 
        electronic_config: np.ndarray, 
        max_count: int, 
        prop_types: Dict[str, Literal["continuous", "binary"]]={},
        verbose: bool=True
    ):
        super().__init__()
        self.model = GPT2Model(config)
        self.elements = elements
        self.embedding = CompositionEmbedding(
            dim_embedding=config.n_embd, 
            elements=elements, 
            use_electronic_config=True,
            max_count=max_count,
            electronic_config=electronic_config
        )
        self.prop_embedding = nn.ModuleDict()
        self.prop_types = prop_types
        for prop, prop_type in prop_types.items():
            if verbose:
                print(f"Building property embedding for {prop} with type {prop_type}")
            if prop_type == "continuous":
                self.prop_embedding[prop] = nn.Linear(1, config.n_embd)
            elif prop_type == "binary":
                self.prop_embedding[prop] = nn.Embedding(2, config.n_embd)
            else:
                raise ValueError(f"Invalid property type: {prop_type}")
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, 
        input_elements: torch.LongTensor, 
        input_counts: torch.LongTensor, 
        input_props: Dict[str, torch.Tensor], 
        attention_mask: Optional[torch.Tensor]=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embedding = self.embedding(input_elements, input_counts) # (B, L, n_embd)
        for prop, prop_value in input_props.items():
            if prop in self.prop_embedding:
                input_embedding[:, 0, :] += self.prop_embedding[prop](prop_value)
        last_hidden_state = self.model(inputs_embeds=input_embedding, attention_mask=attention_mask).last_hidden_state
        outputs = self.lm_head(last_hidden_state)
        elem_logits = outputs[:, :, :len(self.elements)]
        count_logits = outputs[:, :, len(self.elements):]
        return elem_logits, count_logits


def predict_sequence(
    model: GPT2ForChem, 
    device: str, 
    elements: ElementList, 
    prop_dict: Dict[str, Union[float, int]]={}, 
    n_samples: int=10
) -> List[List[Tuple[str, int]]]:
    """
    Autoregressively predict the next tokens from 'start_tokens'
    until we reach 'END' or hit max_steps.
    """
    model.eval()
    collections = []
    with torch.no_grad():
        for _ in range(n_samples):
            predicted = [("START", 0)]
            input_props = {}
            for prop, prop_value in prop_dict.items():
                if prop not in model.prop_types:
                    continue
                if model.prop_types[prop] == "continuous":
                    input_props[prop] = torch.tensor(prop_value).float().reshape(-1, 1).to(device)
                elif model.prop_types[prop] == "binary":
                    input_props[prop] = torch.tensor(prop_value).long().to(device)
                else:
                    raise ValueError(f"Invalid property type: {model.prop_types[prop]}")
            for __ in range(MAX_LENGTH):
                # input shape (1, L_current)
                input_elements, input_counts = zip(*[(elements.element_to_idx[el], ct) for el, ct in predicted])
                input_elements = torch.tensor(input_elements).unsqueeze(0).to(device)
                input_counts = torch.tensor(input_counts).unsqueeze(0).to(device)
                
                elem_logits, count_logits = model(input_elements, input_counts, input_props)
                
                # Get the final step's prediction
                elem_logits = torch.softmax(elem_logits[0, -1, :], dim=-1)  # (num_elements,)
                count_logits = torch.softmax(count_logits[0, -1, :], dim=-1)  # (MAX_COUNT + 1,)
                
                elem_idx = torch.multinomial(elem_logits, 1).item()
                count_idx = torch.multinomial(count_logits, 1).item()
                elem = elements[elem_idx]
                count = count_idx

                if elem == "END":
                    break

                if count > 0 and elem != "PAD" and elem != "START":
                    predicted.append((elem, count))
            
            collections.append(predicted[1:])
    return collections


def get_valence_error(sample):
    return sum([get_element_valence(el)[-1] * ct for el, ct in sample])


class FormulaGenerator:
    """Load the original ELM checkpoint and preserve sampled oxidation states."""
    def __init__(self, config):
        global MAX_COUNT, MAX_LENGTH
        self.config = config
        MAX_COUNT, MAX_LENGTH = config["max_count"], config["max_length"]
        self.device = config["device"]
        set_seed(config["seed"])
        kind = config["formula_type"]
        self.elements = ElementList({"ionic": IONIC_SPECIES, "alloy": ALLOY_SPECIES}[kind])
        electronic = {"ionic": IONIC_ELECTRONIC_CONFIG, "alloy": ALLOY_ELECTRONIC_CONFIG}[kind]
        architecture = config["architecture"]
        self.model = GPT2ForChem(
            GPT2Config(vocab_size=len(self.elements) + MAX_COUNT + 1,
                       n_positions=MAX_LENGTH, **architecture),
            self.elements, electronic, MAX_COUNT,
            prop_types=config["property_types"], verbose=False,
        ).to(self.device)
        self.model.load_state_dict(torch.load(config["checkpoint"], map_location=self.device, weights_only=True))
        self.model.eval()

    def generate(self):
        count = self.config["num_formulas"]
        seen, results = set(), []
        for attempt in range(self.config["max_sampling_attempts"]):
            sample = predict_sequence(self.model, self.device, self.elements,
                                      self.config["property_values"], n_samples=1)[0]
            if not sample or get_valence_error(sample) != 0:
                continue
            amounts, charges = defaultdict(int), defaultdict(set)
            for species, amount in sample:
                element, valence = get_element_valence(species)
                amounts[element] += amount
                charges[element].add(valence)
            # The route interface needs one formal oxidation state per element.
            if any(len(states) != 1 for states in charges.values()):
                continue
            comp = Composition(dict(amounts))
            formula = comp.reduced_formula
            if formula in seen:
                continue
            states = {element: next(iter(values)) for element, values in charges.items()}
            seen.add(formula)
            results.append({"formula": formula, "oxidation_states": states,
                            "elm_tokens": [[species, amount] for species, amount in sample]})
            if len(results) == count:
                return results
        raise ValueError(f"ELM produced {len(results)}/{count} usable unique formulas within the sampling limit")

    def close(self):
        del self.model
        torch.cuda.empty_cache()


def generate_formulas(config, output_file):
    generator = FormulaGenerator(config)
    try:
        formulas = generator.generate()
        save_json(output_file, {"formulas": formulas})
        return formulas
    finally:
        generator.close()
