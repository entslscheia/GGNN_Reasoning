from torch.utils.data import DataLoader

class ABoxDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(ABoxDataloader, self).__init__(*args, **kwargs)
