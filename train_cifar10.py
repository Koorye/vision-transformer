import random
import torch
from visdom import Visdom
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.datasets import CIFAR10
from torchvision import transforms as T
from tqdm import tqdm

from models.vit import VisionTransformer

seed = 999
depth = 2
n_heads = 12
n_classes = 10
n_epoches = 7
batch_size = 256

random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

trans = T.Compose([
    T.Resize(256),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, -.406], [0.229, 0.224, 0.225]),
])
train_set = CIFAR10(root='data', train=True, download=True, transform=trans)
val_set = CIFAR10(root='data', train=False, download=True, transform=trans)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

model = VisionTransformer(n_classes=n_classes,
                          depth=depth,
                          n_heads=n_heads,
                          proj_dropout=.1,
                          attn_dropout=.1,
                          dropout=.1)

optim = Adam(model.parameters(), lr=1e-3, betas=(.5,.999))
criterion = nn.CrossEntropyLoss()
viz = Visdom()
viz.line([0],[0], win='Train Loss', opts=dict(title='Train Loss'))
viz.line([0],[0], win='Val ACC', opts=dict(title='Val ACC'))

for epoch in range(1,n_epoches+1):

    model.train()
    total_loss = 0.

    for index, (data, target) in enumerate(tqdm(train_loader, desc=f'Epoch{epoch} Train')):
        data = data.to(device)
        target = target.to(device)

        scores = model(data)
        loss = criterion(scores, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total_loss += loss.item()
        viz.line([total_loss / (index+1)], [index+1], win='Train Loss', update='append')
    
    model.eval()
    n_correct = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f'Epoch{epoch} Val'):
            data = data.to(device)
            target = target.to(device)

            scores = model(data)
            pred = scores.argmax(dim=1)
            n_correct += torch.eq(pred, target).sum().item()

        viz.line([n_correct / len(val_loader)], [epoch], win='Val ACC', update='append')
    
    torch.save(model.state_dict(), f'weights/VIT_depth{depth}_{n_heads}heads_{n_classes}classes_epoch{epoch}_acc{n_correct / len(val_loader)}.pth')