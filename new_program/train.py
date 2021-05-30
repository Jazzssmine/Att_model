import sys
sys.path.append("../../")
import os

from newsrec_utils import prepare_hparams
from nrms import NRMSModel
from mind_iterator import Iterator

epochs = 5
seed = 42
batch_size = 32

# Options: demo, small, large
MIND_type = 'small'

data_path = 'MIND/small'
train_news_file = os.path.join(data_path, 'train', r'news.tsv')
train_behaviors_file = os.path.join(data_path, 'train', r'behaviors.tsv')
valid_news_file = os.path.join(data_path, 'valid', r'news.tsv')
valid_behaviors_file = os.path.join(data_path, 'valid', r'behaviors.tsv')
wordEmb_file = os.path.join(data_path, "utils", "embedding.npy")
userDict_file = os.path.join(data_path, "utils", "uid2index.pkl")
wordDict_file = os.path.join(data_path, "utils", "word_dict.pkl")
yaml_file = os.path.join(data_path, "utils", r'nrms.yaml')

hparams = prepare_hparams(yaml_file,
                          wordEmb_file=wordEmb_file,
                          wordDict_file=wordDict_file,
                          userDict_file=userDict_file,
                          batch_size=batch_size,
                          epochs=epochs,
                          show_step=10)
print(hparams)

iterator = Iterator
model = NRMSModel(hparams, iterator, seed=seed)

model.run_eval(valid_news_file, valid_behaviors_file)