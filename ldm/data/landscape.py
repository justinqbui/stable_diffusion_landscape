import os
import numpy as np
from PIL import Image
import albumentations
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from taming.data.imagenet import retrieve


class ImagePathTopCrop(Dataset):
    '''
    custom class that crops such that we keep the top of the image
    based on the ImagePaths class from taming transformers
    '''
    def __init__(self, paths, size=None, labels=None):
        self.size = size

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.uint8)
        image = (image/127.5 - 1.0).astype(np.float32)

        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class LandscapeBase(Dataset):
    def __init__(self, config):
        self.config = config or OmegaConf.create()

        self.keep_orig_class_label = False
        self.process_images = True
        self._prepare()
        self._prepare_synset_to_human()
        self._prepare_idx_to_synset()
        self._prepare_human_to_integer_label()
        self._load()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def _prepare(self):
        raise NotImplementedError()

    def _prepare_synset_to_human(self):
        self.human_dict = os.path.join(self.root, "synset_human.txt")

    def _prepare_idx_to_synset(self):
        self.idx2syn = os.path.join(self.root, "index_synset.yaml")

    def _prepare_human_to_integer_label(self):
        self.human2int = os.path.join(self.root, "human_int_label.yaml")


    def _load(self):
        with open(self.txt_filelist, "r") as f:
            self.relpaths = f.read().splitlines()

        self.synsets = [p.split(os.sep)[0] for p in self.relpaths]
        self.abspaths = [os.path.join(self.datadir, p) for p in self.relpaths]

        unique_synsets = np.unique(self.synsets)
        class_dict = dict((synset, i) for i, synset in enumerate(unique_synsets))
        self.class_labels = [class_dict[s] for s in self.synsets]


        with open(self.human_dict, "r") as f:
            human_dict = f.read().splitlines()
            human_dict = dict(line.split(maxsplit=1) for line in human_dict)

        self.human_labels = [human_dict[s] for s in self.synsets]

        labels = {
            "relpath": np.array(self.relpaths),
            "synsets": np.array(self.synsets),
            "class_label": np.array(self.class_labels),
            "human_label": np.array(self.human_labels),
        }

        if self.process_images:
            self.size = retrieve(self.config, "size", default=256)
            self.data = ImagePathTopCrop(self.abspaths,
                                   labels=labels,
                                   size=self.size,
                                   )
        else:
            self.data = self.abspaths


class LandscapeTrain(LandscapeBase):
    NAME = "train"
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)
    def _prepare(self):
        # path must be in ~/.cache/autoencoders/landscape-ldm/train
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~" + os.sep + ".cache"))
        self.root = os.path.join(cachedir, "autoencoders" + os.sep + "landscape-ldm", self.NAME)

        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        # self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
        #                             default=True)

class LandscapeValidation(LandscapeBase):
    NAME = "val"
    def __init__(self, process_images=True, data_root=None, **kwargs):
        self.process_images = process_images
        self.data_root = data_root
        super().__init__(**kwargs)

    def _prepare(self):
        # path must be in ~/.cache/autoencoders/landscape-ldm/val
        cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~" + os.sep + ".cache"))
        self.root = os.path.join(cachedir, "autoencoders" + os.sep + "landscape-ldm", self.NAME)

        self.datadir = os.path.join(self.root, "data")
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        # self.random_crop = retrieve(self.config, "ImageNetTrain/random_crop",
        #                             default=True)