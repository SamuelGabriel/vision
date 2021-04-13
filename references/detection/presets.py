import transforms as T
from torchvision import transforms


# Normlize like the pre-trained backbone (https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198)
def normalize(img, target):
    normalization_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
    return normalization_transform(img), target

class DetectionPresetTrain:
    def __init__(self, hflip_prob=0.5):
        trans = [T.ToTensor(), normalize]
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))

        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.Compose([T.ToTensor(),normalize])

    def __call__(self, img, target):
        return self.transforms(img, target)
