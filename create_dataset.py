import numpy as np
from utils.parse_utils import BIWIParser, DpParser, create_dataset

# eth data.
# annot_file = 'data/ewap_dataset/seq_eth/obsmat.txt'
# npz_out_file = 'data/ewap_dataset/seq_eth/data.npz'
# parser = BIWIParser()

# DP data.
annot_file = 'data/dp_vehicle/train/*0.txt'
npz_out_file = 'data/dp_vehicle/train/data_4s_with_offset.npz'
parser = DpParser()

parser.load(annot_file)

obsvs, preds, times, batches = create_dataset(parser, n_past=5, n_next=40)

np.savez(npz_out_file, obsvs=obsvs, preds=preds, times=times, batches=batches)
print('dataset was created successfully and stored in:', npz_out_file)
