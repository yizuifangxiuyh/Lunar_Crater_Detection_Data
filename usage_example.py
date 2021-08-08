from __future__ import division
import numpy as np
from numpy import random
from PIL import Image
import numbers


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    def __init__(self, trans):
        self.transforms = trans

    def __call__(self, img, bboxes=None):
        for t in self.transforms:
            img, bboxes = t(img, bboxes)
        return img, bboxes


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        :param size: h,w
        :param interpolation:
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, bboxes=None):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img, bboxes
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                scale = self.size / w
                bboxes[:, :4] *= scale
                return img.resize((ow, oh), self.interpolation), bboxes
            else:
                oh = self.size
                ow = int(self.size * w / h)
                scale = self.size / h
                bboxes[:, :4] *= scale
                return img.resize((ow, oh), self.interpolation), bboxes
        else:
            w, h = img.size
            scale = np.array([self.size[::-1][0] / w, self.size[::-1][0] / h])
            # x, x
            bboxes[:, [0, 2]] *= scale[0]
            # y, y
            bboxes[:, [1, 3]] *= scale[1]
            return img.resize(self.size[::-1], self.interpolation), bboxes


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes=None):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            img_center = np.array(img.size[:2])[::-1] / 2
            # h, w
            img_center = np.hstack((img_center, img_center))
            bboxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - bboxes[:, [0, 2]])

            box_w = abs(bboxes[:, 0] - bboxes[:, 2])

            bboxes[:, 0] -= box_w
            bboxes[:, 2] += box_w
            return img.transpose(Image.FLIP_LEFT_RIGHT), bboxes
        return img, bboxes


class ToGray(object):
    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img, bboxes=None):
        """
        Args:
            img (PIL Image): Image to be converted to grayscale.

        Returns:
            PIL Image: Randomly grayscaled image.
        """
        if self.num_output_channels == 1:
            img = img.convert('L')
        elif self.num_output_channels == 3:
            img = img.convert('L')
            np_img = np.array(img, dtype=np.uint8)
            np_img = np.dstack([np_img, np_img, np_img])
            img = Image.fromarray(np_img, 'RGB')
        else:
            raise ValueError('num_output_channels should be either 1 or 3')

        return img, bboxes


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, bboxes=None):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor, bboxes


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, bboxes=None):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """

        i, j, h, w = self.get_params(img, self.size)

        # convert to integer rect x1,y1,x2,y2
        rect = np.array([j, i, j + w, i + h])
        # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
        overlap = jaccard_numpy(bboxes[:, :4], rect)
        # keep overlap with gt box IF center in sampled patch
        centers = (bboxes[:, :2] + bboxes[:, 2:4]) / 2.0
        # mask in all gt boxes that above and to the left of centers
        m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
        # mask in all gt boxes that under and to the right of centers
        m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
        # mask in that both m1 and m2 are true
        mask = m1 * m2
        # take only matching gt boxes
        current_bboxes = bboxes[mask, :].copy()
        # should we use the box left and top corner or the crop's
        current_bboxes[:, :2] = np.maximum(current_bboxes[:, :2], rect[:2])
        # adjust to crop (by substracting crop's left,top)
        current_bboxes[:, :2] -= rect[:2]

        current_bboxes[:, 2:4] = np.minimum(current_bboxes[:, 2:4], rect[2:])
        # adjust to crop (by substracting crop's left,top)
        current_bboxes[:, 2:4] -= rect[:2]

        return img.crop((j, i, j + w, i + h)), current_bboxes


if __name__ == "__main__":
    import glob
    import cv2
    import os

    def get_images_paths(name_pattern):
        """
        Get images paths of the dataset !
        :param name_pattern: prepared for glob.glob using
        :return: image paths
        """
        images_paths = glob.glob(name_pattern)
        return images_paths


    def vis_detections(im, dets):
        """Draw detected bounding boxes."""

        for box in dets:
            cv2.rectangle(im, tuple(box[0:2]), tuple(box[2:4]), (255, 0, 0), 2)
        return im


    def vis_circle(im, dets):
        """Draw detected bounding boxes."""

        for box in dets:
            cv2.circle(im, tuple((box[0:2] + box[2:4]) // 2), (box[2] - box[0]) // 2, (255, 0, 0), 2)
        return im


    crater_png_paths = get_images_paths(
        '../LRO_DATA/train/*_small.png')

    crater_png_paths.sort()
    transform_list = []
    transform_list.append(ToGray(1))
    transform_list.append(Resize([400, 400], Image.BICUBIC))
    transform_list.append(RandomCrop(200))
    transform_list.append(RandomHorizontalFlip())
    augment_test = Compose(transform_list)

    for crater_path in crater_png_paths:
        txt_file_name = crater_path.replace('.png', '.txt')
        if os.path.exists(txt_file_name):
            with open(txt_file_name, 'r') as txt_file:
                crater_info = np.loadtxt(txt_file, delimiter=',', skiprows=1, dtype=np.int32)
                if len(crater_info.shape) == 1:
                    crater_info = crater_info[np.newaxis, :]

        crater_bboxes = np.hstack((crater_info[:, 1:3],
                                   crater_info[:, 1:3] + crater_info[:, 3:5]))
        crater_arr = cv2.imread(crater_path)
        big_test = crater_arr.copy()
        vis_circle(big_test, crater_bboxes)
        cv2.imshow('3', big_test)

        crater_bboxes = crater_bboxes.astype(np.float32)
        crater_img = Image.open(crater_path).convert('L')
        img, bboxes = augment_test(crater_img, crater_bboxes)
        img = np.asarray(img)
        cv2.imshow('2', img)
        bboxes = bboxes.astype(np.int32)  # convert it into a numpy array
        bboxes = bboxes[bboxes[:, 2] - bboxes[:, 0] >= 8]
        bboxes = bboxes[bboxes[:, 3] - bboxes[:, 1] >= 8]

        img_test = img.copy()
        vis_circle(img_test, bboxes)
        cv2.imshow('1', img_test)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
