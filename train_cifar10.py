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
depth = 6
n_heads = 8
n_classes = 10
n_epoches = 20
batch_size = 16
show_size = 8
# historical_weights = 'VIT_depth2_4heads_10classes_epoch1_acc21.29936305732484.pth'
historical_weights = None

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

random.seed(seed)
torch.manual_seed(seed)

device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Use {device}')

trans = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([.485, .456, .406], [.229, .224, .225]),
])
untrans = T.Compose([
    T.Normalize([-.485/.229, -.456/.224, -.406/.225],
                [1/.229, 1/.224, 1/.225]),
])

train_set = CIFAR10(root='data', train=True, download=True, transform=trans)
val_set = CIFAR10(root='data', train=False, download=True, transform=trans)

train_loader = DataLoader(train_set, batch_size=batch_size,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=batch_size,
                        shuffle=False, num_workers=0)

model = VisionTransformer(img_size=32,
                          patch_size=2,
                          n_classes=n_classes,
                          depth=depth,
                          n_heads=n_heads,
                          embed_dim=512,
                          mlp_ratio=1
                          ).to(device)
if historical_weights is not None:
    model.load_state_dict(torch.load(f'weights/{historical_weights}'))
    historical_epoch = int(historical_weights.split('_')[4][5:])
else:
    historical_epoch = 0

optim = Adam(model.parameters(), lr=1e-4, betas=(.5, .999))
criterion = nn.CrossEntropyLoss()
viz = Visdom()
viz.line([0], [0], win='Train Loss', opts=dict(title='Train Loss'))
viz.line([0], [0], win='Val ACC', opts=dict(title='Val ACC'))


for epoch in range(historical_epoch+1, n_epoches+1):

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
        viz.line([total_loss / (index+1)],
                 [(epoch-1) * len(train_loader) + index + 1],
                 win='Train Loss',
                 update='append')

    model.eval()
    n_correct = 0

    with torch.no_grad():
        for data, target in tqdm(val_loader, desc=f'Epoch{epoch} Val'):
            data = data.to(device)
            target = target.to(device)

            scores = model(data)
            pred = scores.argmax(dim=1)

            n_correct += torch.eq(pred, target).sum().item()

            pred_labels = [classes[int(idx)]
                           for idx in pred[:show_size].cpu().numpy().flatten()]
            pred_str = ''
            for label in pred_labels:
                pred_str += label + '<br><br>'

            viz.images(untrans(data[:show_size]), nrow=1, win='Predict Image',
                       opts=dict(title='Predict Image'))
            viz.text(pred_str, win='Predict', opts=dict(title='Predict'))

    acc = n_correct / len(val_set)
    viz.line([acc], [epoch],
            win='Val ACC', update='append')

    torch.save(model.state_dict(
    ), f'weights/VIT_depth{depth}_{n_heads}heads_{n_classes}classes_epoch{epoch}_acc{acc:.2%}.pth')
