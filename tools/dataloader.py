# dataloader.py
from torch.utils.data import Dataset, DataLoader

def MNISTDataloader(Dataset):
    train_loader = DataLoader(Dataset['train'], batch_size=512, shuffle=True)
    test_loader = DataLoader(Dataset['test'], batch_size=512, shuffle=False)
    return train_loader, test_loader