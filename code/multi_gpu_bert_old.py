from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import logging
import torch
import random
import os
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader

from sklearn.preprocessing import OneHotEncoder   #导入OneHotEncoder库
from transformers import BertTokenizer, AdamW, BertModel, BertPreTrainedModel, BertConfig, \
    get_linear_schedule_with_warmup, XLNetModel, XLNetTokenizer, XLNetConfig
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.utils import shuffle
import logging
logging.basicConfig(level=logging.DEBUG, filename="train.log",filemode='a')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda')
#device = torch.device('cpu')

#model_path = './chinese_wwm_pytorch/'
# model_path = '../chinese_xlnet_mid_pytorch/'
model_path = "./chinese_roberta_wwm_large_ext_pytorch/"
# model_path = "../MC-BERT/"
# model_path = "../ernie/"

bert_config = BertConfig.from_pretrained(model_path + 'bert_config.json', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained(model_path + 'vocab.txt', config=bert_config)

# xlnet_config = XLNetConfig.from_pretrained(model_path + 'config.json', output_hidden_states=True)
# tokenizer = XLNetTokenizer.from_pretrained(model_path + 'spiece.model', config=xlnet_config)

seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

modelId = 0

MAX_LEN = 512
epoch = 5
num_class = 10
learn_rate = 2e-5
train_batch_size = 32
valid_batch_size = 32
k_fold = 5
dropout = 0.2

file_path = './log/'
# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)

train = pd.read_csv('data/train.csv')
#semi = pd.read_csv('data/Semi-supervised_test.csv')
#train = pd.concat([train, semi], sort=False)
test = pd.read_csv('data/test_data.csv')
sub = pd.read_csv('data/submit_example.csv')

train_content = train['content'].values.astype(str)
test_content = test['content'].values.astype(str)


Onehot_mapping = {
    '财经': 0,
    '房产': 1,
    '家居': 2,
    '教育': 3,
    '科技': 4,
    '时尚': 5,
    '时政': 6,
    '游戏': 7,
    '娱乐': 8,
    '体育': 9
}
class_mapping = dict(zip(Onehot_mapping.values(), Onehot_mapping.keys()))

train_label = np.array(train['class_label'].map(Onehot_mapping))
test_label = np.zeros((len(test),num_class))

oof_train = np.zeros((len(train), num_class), dtype=np.float32)
oof_test = np.zeros((len(test), num_class), dtype=np.float32)

class BertForClass(nn.Module):
    def __init__(self, n_classes=num_class):
        super(BertForClass, self).__init__()
        self.model_name = 'BertForClass'
        self.bert_model = BertModel.from_pretrained(model_path, config=bert_config)
        self.dropout = nn.Dropout(p=dropout)
        self.multi_drop = 5
        self.multi_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(self.multi_drop)])
        self.classifier = nn.Linear(bert_config.hidden_size * 2, n_classes)

    def forward(self, input_ids, input_masks, segment_ids):
        sequence_output, pooler_output, hidden_states = self.bert_model(input_ids=input_ids, token_type_ids=segment_ids,
                                                                        attention_mask=input_masks)
        seq_avg = torch.mean(sequence_output, dim=1)
        concat_out = torch.cat((seq_avg, pooler_output), dim=1)
        logit = self.classifier(self.dropout(concat_out))
        '''
        for j, dropout in enumerate(self.multi_dropouts):
            if j == 0:
                logit = self.classifier(dropout(concat_out)) / self.multi_drop
            else:
                logit += self.classifier(dropout(concat_out)) / self.multi_drop
        '''
        return logit


class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='word_embeddings', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='word_embeddings'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]


class data_generator:
    def __init__(self, data, batch_size=16, max_length=MAX_LEN, shuffle=False):
        self.data = data
        self.batch_size = batch_size
        self.max_length = max_length
        self.shuffle = shuffle
        self.steps = len(self.data[0]) // self.batch_size
        if len(self.data[0]) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        c, y = self.data
        idxs = list(range(len(self.data[0])))
        if self.shuffle:
            np.random.shuffle(idxs)
        input_ids, input_masks, segment_ids, labels = [], [], [], []

        for index, i in enumerate(idxs):

            text = c[i]
            if len(text) > 512:
                text = text[:256] + text[-256:]
            input_id = tokenizer.encode(text, max_length=self.max_length,truncation='longest_first')
            input_mask = [1] * len(input_id)
            segment_id = [0] * len(input_id)
            padding_length = self.max_length - len(input_id)
            input_id += ([0] * padding_length)
            input_mask += ([0] * padding_length)
            segment_id += ([0] * padding_length)

            input_ids.append(input_id)
            input_masks.append(input_mask)
            segment_ids.append(segment_id)
            labels.append(y[i])
            if len(input_ids) == self.batch_size or i == idxs[-1]:
                yield input_ids, input_masks, segment_ids, labels
                input_ids, input_masks, segment_ids, labels = [], [], [], []


kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)

for fold, (train_index, valid_index) in enumerate(kf.split(train_content, train_label)):
    # if fold <= 3:
    #   continue
    print('\n\n------------fold:{}------------\n'.format(fold))
    c = train_content[train_index]
    y = train_label[train_index]

    val_c = train_content[valid_index]
    val_y = train_label[valid_index]

    train_D = data_generator([c, y], batch_size=train_batch_size, shuffle=True)
    val_D = data_generator([val_c, val_y], batch_size=valid_batch_size)

    model = BertForClass().to(device)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # model = BertForClass().to(device)
    # pgd = PGD(model)
    # K = 3
    loss_fn = nn.CrossEntropyLoss()  # BCEWithLogitsLoss就是把Sigmoid-BCELoss合成一步


    num_train_steps = int(len(train) / train_batch_size * epoch)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=learn_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train) / train_batch_size / 2),
        num_training_steps=num_train_steps
    )

    best_f1 = 0
    PATH = './models/model{}/bert_{}.pth'.format(modelId, fold)
    save_model_path = './models/model{}'.format(modelId)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    for e in range(epoch):
        print('\n------------epoch:{}------------'.format(e))
        model.train()
        acc = 0
        train_len = 0
        loss_num = 0
        tq = tqdm(train_D)

        for input_ids, input_masks, segment_ids, labels in tq:
            input_ids = torch.tensor(input_ids).to(device)
            input_masks = torch.tensor(input_masks).to(device)
            segment_ids = torch.tensor(segment_ids).to(device)
            label_t = torch.tensor(labels, dtype=torch.long).to(device)

            y_pred = model(input_ids, input_masks, segment_ids)

            loss = loss_fn(y_pred, label_t)
            loss.backward()
            '''
            pgd.backup_grad()
            # 对抗训练
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))  # 在embedding上添加对抗扰动, first attack时备份param.data
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()
                y_pred = model(input_ids, input_masks, segment_ids)

                loss_adv = loss_fn(y_pred, label_t)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            pgd.restore()  # 恢复embedding参数
            '''

            # 梯度下降，更新参数
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

            y_pred = np.argmax(y_pred.detach().to("cpu").numpy(), axis=1)
            acc += sum(y_pred == labels)
            loss_num += loss.item()
            train_len += len(labels)
            tq.set_postfix(fold=fold, epoch=e, loss=loss_num / train_len, acc=acc / train_len)

        model.eval()
        with torch.no_grad():
            y_p = []
            train_logit = None
            for input_ids, input_masks, segment_ids, labels in tqdm(val_D):
                input_ids = torch.tensor(input_ids).to(device)
                input_masks = torch.tensor(input_masks).to(device)
                segment_ids = torch.tensor(segment_ids).to(device)
                label_t = torch.tensor(labels, dtype=torch.long).to(device)

                y_pred = model(input_ids, input_masks, segment_ids)
                y_pred = y_pred.detach().to("cpu").numpy()
                if train_logit is None:
                    train_logit = y_pred
                else:
                    train_logit = np.vstack((train_logit, y_pred))

                y_pred = np.argmax(y_pred, axis=1)
                y_p += list(y_pred)

            f1 = f1_score(y_p, val_y, average="macro")
            print("best_f1:{}  f1:{}\n".format(best_f1, f1))
            if f1 >= best_f1:
                best_f1 = f1
                oof_train[valid_index] = np.array(train_logit)
                torch.save(model, PATH)

    test_D = data_generator([test_content, test_label], batch_size=valid_batch_size)
    model = torch.load(PATH).to(device)
    model.eval()
    with torch.no_grad():
        res = []
        pred_logit = None

        for input_ids, input_masks, segment_ids, labels in tqdm(test_D):
            input_ids = torch.tensor(input_ids).to(device)
            input_masks = torch.tensor(input_masks).to(device)
            segment_ids = torch.tensor(segment_ids).to(device)

            y_pred = model(input_ids, input_masks, segment_ids)
            y_pred = y_pred.detach().to("cpu")

            if pred_logit is None:
                pred_logit = y_pred
            else:
                pred_logit = np.vstack((pred_logit, y_pred))


    oof_test += np.array(pred_logit)

    optimizer.zero_grad()

    del model
    torch.cuda.empty_cache()

oof_test /= k_fold
save_result_path = './result'
if not os.path.exists(save_result_path):
    os.makedirs(save_result_path)



sub['class_label'] = np.argmax(oof_test,axis=1)
sub['class_label'] = sub['class_label'].map(class_mapping)

def rank_label(label):
    if label in ["财经","时政"]:
        return "高风险"
    elif label in ["房产", "科技"]:
        return "中风险"
    elif label in ["教育", "时尚", "游戏"]:
        return "低风险"
    elif label in ["家居", "体育", "娱乐"]:
        return "可公开"

sub['rank_label'] = sub['class_label'].apply(rank_label)
sub.to_csv("./result/result{}_original.csv".format(modelId), index=False)


from functools import partial
import numpy as np
import scipy as sp
from sklearn.metrics import f1_score
class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef*X_p
        ll = f1_score(y, np.argmax(X_p, axis=-1), average='macro')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1. for _ in range(len(set(y)))]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
        return f1_score(y, np.argmax(X_p, axis=-1), average='macro')

    def coefficients(self):
        return self.coef_['x']

op = OptimizedF1()
op.fit(oof_train,train_label)

oof_test_optimizer = op.coefficients()*oof_test

sub['class_label'] = np.argmax(oof_test_optimizer,axis=1)
sub['class_label'] = sub['class_label'].map(class_mapping)

def rank_label(label):
    if label in ["财经","时政"]:
        return "高风险"
    elif label in ["房产", "科技"]:
        return "中风险"
    elif label in ["教育", "时尚", "游戏"]:
        return "低风险"
    elif label in ["家居", "体育", "娱乐"]:
        return "可公开"

sub['rank_label'] = sub['class_label'].apply(rank_label)
sub.to_csv("./result/result{}_optimizer.csv".format(modelId), index=False)

np.save('./result/oof_train{}.npy'.format(modelId), oof_train)
np.save('./result/oof_test{}.npy'.format(modelId), oof_test)
np.save('./result/oof_test_optimizer{}.npy'.format(modelId), oof_test_optimizer)
