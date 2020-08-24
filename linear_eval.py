import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from model import SimCLR
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torchvision.transforms as transforms

def evaluate(test_loader, model, classifier, device):
    correct = 0    
    total = 0
    model.eval()
    classifier.eval()
    with torch.no_grad():
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device) 
            h = model.get_hidden(test_x)
            logits = classifier(h)
              
            pred = torch.argmax(logits, dim=1)
            total += test_y.shape[0]
            correct += (pred == test_y).sum().item()
    total_acc = 100.0 * correct / total
    return total_acc
        
def main(args):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    ## Hyperparameters
    finetune = args.finetune if not args.baseline else True
    proj_dim = 512
    hidden_dim = args.hid_dim

    model = SimCLR(out_dim=256).to(device)
    
    if args.dataset == 'STL10':
        # Dataset
        train_dataset = datasets.STL10('./data', split='train', transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.STL10('./data', split='test', transform=transforms.ToTensor(), download=True)
    else: # cifar dataset
        img_transform = transforms.Compose([transforms.CenterCrop(96), transforms.ToTensor()])
        train_dataset = datasets.CIFAR10('./data', train=True, transform=img_transform, download=True)
        test_dataset = datasets.CIFAR10('./data', train=False, transform=img_transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_worker, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, num_workers=args.num_worker)

    n_classes = 10

    if not args.baseline:
        ckpt = torch.load(args.simclr_path)
        print("Load checkpoint trained for %d epochs. Loss is %f." %(ckpt["epoch"], ckpt["loss"]))
        model.load_state_dict(ckpt["model"])

    # Freeze encoder network
    if not finetune:
        for p in model.f.parameters():
            p.requires_grad = False

    # Add Linear Classifier.
    classifier = nn.Sequential(nn.Linear(proj_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_classes)).to(device)
    if finetune: 
        p = list(model.f.parameters())+list(classifier.parameters())
    else:
        p = classifier.parameters()
    optimizer = torch.optim.Adam(p, lr=3e-04) 

    target_acc = 0.0 # apply Early stopping
    token = 0 

    # Train Linear Classifier
    for epoch in range(50):
        start = time.time()
        for x, y in train_loader:
            classifier.train()
            x, y = x.to(device), y.to(device)
            h = model.get_hidden(x)
            logits = classifier(h) 
        
            optimizer.zero_grad()
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

        valid_acc = evaluate(test_loader, model, classifier, device)
        print("[Epoch %2d] Valid Loss %f. Time takes %s" %(epoch, valid_acc, time.time()-start))
        if (valid_acc > target_acc):
            target_acc = valid_acc
            token = 0 
        elif token >= 2:
            print("Early Stop at Epoch %d." %(epoch))
            break
        else:
            token += 1

    # Evaluate
    acc = evaluate(test_loader, model, classifier, device)
    print("[Eval] finetune {} Acc {:.3f}".format(finetune, target_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "SimCLR Linear Evaluation")

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128)

    parser.add_argument(
        '--simclr_path',
        type=str,
        default="model.ckpt")

    parser.add_argument(
        '--dataset',
        type=str,
        choices={"STL10", "cifar"},
        default="STL10")

    parser.add_argument(
        '--hid_dim',
        type=int,
        default=1024)

    parser.add_argument(
        '--num_worker',
        type=int,
        default=8)

    parser.add_argument(
        '--finetune',
        action='store_true')

    parser.add_argument(
        '--baseline',
        action='store_true')

    args = parser.parse_args()
    main(args)
