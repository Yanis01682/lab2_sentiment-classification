from gensim.models import KeyedVectors

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
            #words=["死囚","爱","刽子手"]
            #把words中每一个词转为50维词向量
            #要把words转为词向量，这样的话word2vec["爱"]=...
            vectors=[]
            for word in words:
                if word in word2vec:
                    vectors.append(word2vec[word])
                else:
                    vectors.append([0.0]*50)
            #一行文本->label+words->vectors
            #这个时候，一行的words转成了一个句子长度*50的vector
            #下一步是把vectors和label转成tensor
            vectors=torch.tensor(vectors,dtype=torch.float)
            label=torch.tensor(label,dtype=torch.long)
            data.append((vectors,label))
            if vectors.shape[0]>max_len:
                max_len=vectors.shape[0]

    return data,max_len


    #tensor的结构是[句子长度,50]
    #label原来是一个一维的数，转tensor之后就是一个0维tensor
    #train_data=[
    #(vector1,label1),
    #(vector2,label2),
    #...
    #]
    #其中vector1的shape可能是[34,50]，接下来要padding，把词向量的行补成一致的
    #padding
def build_tensor(data,max_len):
    padded_vectors=[]
    padded_labels=[]
    for vectors,label in data:
        if vectors.shape[0]<max_len:
            pad=torch.zeros(max_len-vectors.shape[0],50)
            vectors=torch.cat([vectors,pad],dim=0)
            padded_vectors.append(vectors)
            padded_labels.append(label)
        else:
            vectors = vectors[:max_len]
            padded_vectors.append(vectors)
            padded_labels.append(label)

    #pad之后，padded_vectors里面的每一个元素都是[max_len,50]的tensor
    #stack一下，X是沿一个新的维度把这N个形状相同的二维张量叠起来
    #X.shape=[N,mex_len,50]
    #第 0 维：第几个句子
    #第 1 维：句子里的第几个词位置
    #第 2 维：这个词向量的 50 个数
    X=torch.stack(padded_vectors)
    y=torch.stack(padded_labels)
    #X是训练输入
    #y是训练标签
    X=X.unsqueeze(1)
    #X.shape=[N,1,max_len,50]
    #加一个通道维，把它看成单通道输出

    return X,y

train_data, train_max_len = read_data("Dataset/train.txt")
val_data, _ = read_data("Dataset/validation.txt")
test_data, _ = read_data("Dataset/test.txt")

X_train, y_train = build_tensor(train_data, train_max_len)
X_val, y_val = build_tensor(val_data, train_max_len)
X_test, y_test = build_tensor(test_data, train_max_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

class TextCNN(nn.Module):#所有神经网络的基类，我定义的神经网络要继承它
    def __init__(self):#初始化函数，也就是创建这个模型对象时，自动执行的函数
        super(TextCNN,self).__init__()#先把父类 nn.Module 该初始化的东西初始化一下
        #池化层
        self.conv=nn.Conv2d( 
            in_channels=1,#通道只有一个
            out_channels=100,#想用100个卷积核，也就是最后提取100个特征
            kernel_size=(3,50)#卷积核高3宽50，高对应句子长度，宽对应词向量维度,也就是一次看3个词，每个词都看完整的50维词向量
        )
        #全连接层
        self.fc=nn.Linear(100,2)#输入100维特征，输出2维，对应二分类的两个类别
    #前向传播
    def forward(self,x):
        x=self.conv(x)#因为卷积核是 (3, 50)，宽度方向正好覆盖整个 50，所以最后那个宽度会变成 1，[N, 100, max_len-3+1, 1]
        x = F.relu(x)
        x = x.squeeze(3)#把最后那个一维的维度去掉，[N, 100, max_len-2]
        x = F.max_pool1d(x, x.size(2))#对每个卷积核得到的一整条序列做“最大池化”(从一串数里，只留下最重要的那个最大值)，[N, 100, 1]
        x = x.squeeze(2)#[N, 100]
        #全连接层
        x = self.fc(x)#[N, 2]，相当于做了一次y=Wx+b函数,x 是长度 100 的向量,W 是一个 2 x 100 的矩阵,输出 y 就是长度 2 的向量
        return x
    
model=TextCNN().to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()#看模型给两个类别各打了多少分，看真实标签是哪一类，算出loss函数
optimizer = optim.Adam(model.parameters(), lr=0.001)#根据loss算出来的梯度，去更新模型参数


# 训练
min_val_loss=float("inf")
best_state=None#保存最好的模型参数
patience=3
wait=0

print(f"device={device}")

for epoch in range(50):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()#把旧梯度清空
    loss.backward()#反向传播，计算梯度
    optimizer.step()#沿这些梯度下降的方向修改
    acc=(torch.argmax(output,dim=1)==y_train).float().mean()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
        val_pred = torch.argmax(val_output, dim=1)
        val_acc = (val_pred == y_val).float().mean()


    print(
        f"epoch {epoch+1}, "
        f"train_loss={loss.item():.4f}, train_acc={acc.item():.4f}, "
        f"val_loss={val_loss.item():.4f}, val_acc={val_acc.item():.4f}"
    )
    if val_loss.item() < min_val_loss:
        min_val_loss = val_loss.item()
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
with torch.no_grad():
    test_output = model(X_test)
    test_pred = torch.argmax(test_output, dim=1)
    test_acc = (test_pred == y_test).float().mean()
    print(f"test_acc={test_acc.item():.4f}")
