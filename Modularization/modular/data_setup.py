""" 
Contains functionality for creating dataloaders
for image classification data.
"""
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders (
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose,
    batch_size: int, 
    num_workers: int = NUM_WORKERS
):
    """Creates training and testing dataloaders"""

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                      transform=transform, # transforms to perform on data (images)
                                      target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir, 
                                    transform=transform)
    class_names = train_data.classes


    train_dataloader = DataLoader(dataset=train_data, 
                              batch_size=batch_size, # how many samples per batch?
                              num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                              shuffle=True, 
                              pin_memory=True)

    test_dataloader = DataLoader(dataset=test_data, 
                                batch_size=batch_size, 
                                num_workers=num_workers,
                                pin_memory=True, 
                                shuffle=False) # don't usually need to shuffle testing data

    return train_dataloader, test_dataloader, class_names
    
    

