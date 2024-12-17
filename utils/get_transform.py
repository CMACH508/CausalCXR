from torchvision import transforms

def mean_std_transform():
    return transforms.ToTensor()

def train_transform(mean, std, size):
    transform = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform

def test_transform(mean, std, size):
    transform = transforms.Compose([
        transforms.Resize([size, size]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform