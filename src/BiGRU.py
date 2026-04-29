from gensim.models import KeyedVectors
#BiGRU：按顺序读句子，同时结合正向和反向两个方向的信息来理解整句话。
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
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
            #把words中每一个词转为50维词向量
            vectors=[]
            for word in words:
                if word in word2vec:
                    vectors.append(word2vec[word])
                else:
                    vectors.append([0.0]*50)
            #一行文本->label+words->vectors
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
        #记录每个句子的真实长度，后面取最后一个有效时间步时要用
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

X_train, y_train,len_train = build_tensor(train_data, train_max_len)
X_val, y_val,len_val = build_tensor(val_data, train_max_len)
X_test, y_test,len_test = build_tensor(test_data, train_max_len)

batch_size = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = X_train.to(device)
y_train = y_train.to(device)
len_train = len_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
len_val=len_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
len_test=len_test.to(device)

train_loader = DataLoader(
    TensorDataset(X_train, y_train,len_train),
    batch_size=batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(X_val, y_val,len_val),
    batch_size=batch_size,
    shuffle=False,
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test,len_test),
    batch_size=batch_size,
    shuffle=False,
)

#BiGRU输入x.shape=[N,max_len,50]
#因为是双向GRU，所以最后每个句子的表示会同时包含正向和反向信息
class TextBiGRU(nn.Module):
    def __init__(self):
        super(TextBiGRU,self).__init__()
        self.rnn=nn.GRU(
            input_size=50,#每个时间步输入50维词向量
            hidden_size=128,#每个方向的隐藏状态维度都是128
            batch_first=True,
            num_layers=2,#堆叠两层GRU，提高表示能力
            dropout=0.3,#多层GRU之间使用dropout
            bidirectional=True,#双向GRU，同时从前往后和从后往前读句子
        )
        #注意力层：给不同时间步分配不同权重
        self.attn=nn.Linear(256,1)
        self.dropout=nn.Dropout(0.5)
        self.fc=nn.Linear(256,2)#注意力加权后得到256维句向量，再做二分类

    def forward(self, x,lengths):
        #pack后让GRU跳过padding位置，减少无效计算和干扰
        packed_x = pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, h_n = self.rnn(packed_x)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        #构造mask，忽略padding位置
        mask = torch.arange(output.size(1), device=output.device).unsqueeze(0) < lengths.unsqueeze(1)

        #对每个时间步计算注意力分数，只关注有效token
        attn_score = self.attn(output).squeeze(2)
        attn_score = attn_score.masked_fill(~mask, float("-inf"))
        attn_weight = torch.softmax(attn_score, dim=1).unsqueeze(2)

        #对所有有效时间步做加权求和，得到句子表示
        context = (output * attn_weight).sum(dim=1)
        x = self.dropout(context)
        x = self.fc(x)
        return x

model=TextBiGRU().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练
min_val_loss=float("inf")
best_state=None#保存最好的模型参数
patience=3
wait=0

print(f"device={device}")

for epoch in range(50):
    model.train()
    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y,batch_len in train_loader:
        output = model(batch_x,batch_len)
        loss = criterion(output, batch_y)

        optimizer.zero_grad()#把旧梯度清空
        loss.backward()#反向传播，计算梯度
        optimizer.step()#沿这些梯度下降的方向修改

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
        for batch_x, batch_y,batch_len in val_loader:
            val_output = model(batch_x,batch_len)
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
    for batch_x, batch_y,batch_len in test_loader:
        test_output = model(batch_x,batch_len)
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
