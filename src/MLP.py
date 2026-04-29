from gensim.models import KeyedVectors

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#加载二进制词向量文件
word2vec=KeyedVectors.load_word2vec_format(
    "Dataset/wiki_word2vec_50.bin",binary=True
)

def read_data(file_path):
    with open(file_path,"r",encoding="utf-8")as f:
        data=[]
        max_len=0
        for line in f:
            line=line.strip()
            parts=line.split()
            label=int(parts[0])
            words=parts[1:]
            vectors=[]
            for word in words:
                if word in word2vec:
                    vectors.append(word2vec[word])
                else:
                    vectors.append([0.0]*50)
            vectors=torch.tensor(vectors,dtype=torch.float)
            label=torch.tensor(label,dtype=torch.long)
            data.append((vectors,label))
            if vectors.shape[0]>max_len:
                max_len=vectors.shape[0]

    return data,max_len

def build_tensor(data,max_len):
    padded_vectors=[]
    padded_labels=[]
    true_lengths=[]
    for vectors,label in data:
        #记录每个句子的真实长度，后面做池化时需要忽略padding
        true_lengths.append(min(vectors.shape[0], max_len))
        if vectors.shape[0]<max_len:
            pad=torch.zeros(max_len-vectors.shape[0],50)
            vectors=torch.cat([vectors,pad],dim=0)
            padded_vectors.append(vectors)
            padded_labels.append(label)
        else:
            vectors = vectors[:max_len]
            padded_vectors.append(vectors)
            padded_labels.append(label)

    X=torch.stack(padded_vectors)
    y=torch.stack(padded_labels)
    true_lengths = torch.tensor(true_lengths, dtype=torch.long)

    return X,y,true_lengths

train_data, train_max_len = read_data("Dataset/train.txt")
val_data, _ = read_data("Dataset/validation.txt")
test_data, _ = read_data("Dataset/test.txt")

X_train, y_train, len_train = build_tensor(train_data, train_max_len)
X_val, y_val, len_val = build_tensor(val_data, train_max_len)
X_test, y_test, len_test = build_tensor(test_data, train_max_len)

batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = X_train.to(device)
y_train = y_train.to(device)
len_train = len_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
len_val = len_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
len_test = len_test.to(device)

train_loader = DataLoader(
    TensorDataset(X_train, y_train, len_train),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val, len_val),
    batch_size=batch_size,
    shuffle=False,
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test, len_test),
    batch_size=batch_size,
    shuffle=False,
)

class TextMLP(nn.Module):
    def __init__(self):
        super(TextMLP, self).__init__()
        #先对每个词向量做MLP映射，再对整句做池化，这样仍然不显式建模词序
        self.token_fc1 = nn.Linear(50, 128)
        self.token_fc2 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.cls_fc1 = nn.Linear(256, 128)
        self.cls_fc2 = nn.Linear(128, 2)

    def forward(self, x, lengths):
        x = F.relu(self.token_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.token_fc2(x))

        #构造mask，忽略padding位置
        mask = (torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(2)

        #对非padding部分做平均池化和最大池化，再拼接成句向量
        x_zero = x * mask.float()
        mean_pool = x_zero.sum(dim=1) / lengths.unsqueeze(1)

        x_masked = x.masked_fill(~mask, float("-inf"))
        max_pool = x_masked.max(dim=1).values

        x = torch.cat([mean_pool, max_pool], dim=1)
        x = self.dropout(F.relu(self.cls_fc1(x)))
        x = self.cls_fc2(x)
        return x

model=TextMLP().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练
min_val_loss=float("inf")
best_state=None
patience=5
wait=0

print(f"device={device}")

for epoch in range(50):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y, batch_len in train_loader:
        output = model(batch_x, batch_len)
        loss = criterion(output, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * batch_y.size(0)
        train_correct += (torch.argmax(output, dim=1) == batch_y).sum().item()
        train_total += batch_y.size(0)

    train_loss = train_loss_sum / train_total
    train_acc = train_correct / train_total

    model.eval()
    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_len in val_loader:
            val_output = model(batch_x, batch_len)
            val_loss = criterion(val_output, batch_y)
            val_pred = torch.argmax(val_output, dim=1)

            val_loss_sum += val_loss.item() * batch_y.size(0)
            val_correct += (val_pred == batch_y).sum().item()
            val_total += batch_y.size(0)

    val_loss_avg = val_loss_sum / val_total
    val_acc = val_correct / val_total

    print(
        f"epoch {epoch+1}, "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"val_loss={val_loss_avg:.4f}, val_acc={val_acc:.4f}"
    )

    if val_loss_avg < min_val_loss:
        min_val_loss = val_loss_avg
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if wait >= patience:
        print("early stopping")
        break

if best_state is not None:
    model.load_state_dict(best_state)

model.eval()
test_correct = 0
test_total = 0
tp = 0
fp = 0
fn = 0

with torch.no_grad():
    for batch_x, batch_y, batch_len in test_loader:
        test_output = model(batch_x, batch_len)
        test_pred = torch.argmax(test_output, dim=1)

        test_correct += (test_pred == batch_y).sum().item()
        test_total += batch_y.size(0)

        tp += ((test_pred == 1) & (batch_y == 1)).sum().item()
        fp += ((test_pred == 1) & (batch_y == 0)).sum().item()
        fn += ((test_pred == 0) & (batch_y == 1)).sum().item()

test_acc = test_correct / test_total
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"test_acc={test_acc:.4f}")
print(f"precision={precision:.4f}")
print(f"recall={recall:.4f}")
print(f"f1={f1:.4f}")
