import argparse
import logging
import os
import os.path as osp
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import *
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
dir_img = osp.join("..", "unet_dataset", "images", "test")
dir_mask = osp.join("..", "unet_dataset", "labels", "test")

def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        # tf = transforms.Compose(
        #     [
        #         transforms.ToPILImage(),
        #         transforms.Resize(full_img.size[1]),
        #         transforms.ToTensor()
        #     ]
        # )

        # probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return (full_mask > out_threshold)


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)
    parser.add_argument('--model_type', type=str, default='unet',
                        help="Model which choosed.")

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files

def caculate_dice(y_true, y_pred, threshold = 0.5, smooth = 0.000001):
    y_true = (y_true > 0.5).astype(np.int_)
    y_pred = (y_pred > 0.5).astype(np.int_)
    return (2. * np.sum(y_true * y_pred)) / (np.sum(y_true) + np.sum(y_pred) + smooth)

def Evaluate(true_mask, pred_mask):
    """
    Get the DICE/IOU between each predicted mask and each true mask.

    Inputs:
        masks_true : array-like
            A 2D array of shape (image_height, image_width)
        masks_pred : array-like
            A 2D array of shape (image_height, image_width)

    Returns:
        array-like
            A 2D array of shape (n_true_masks, n_predicted_masks), where
            the element at position (i, j) denotes the dice between the `i`th true
            mask and the `j`th predicted mask.
    """
    assert true_mask.shape == pred_mask.shape, "Gt and pred must have same shape."
    height, width = true_mask.shape
    m_true = true_mask.copy().reshape(height * width).T
    m_pred = pred_mask.copy().reshape(height * width)
    TP = np.dot(m_pred, m_true)
    TN = np.dot(1 - m_pred, 1 - m_true)
    FP = np.dot(m_pred, 1 - m_true)
    FN = np.dot(1 - m_pred, m_true)
    dice = 2*TP/(2*TP+FP+FN)
    iou = TP/(TP+FP+FN)
    precision = TP/(TP+FP)
    recall = sensitivity = TP/(TP+FN)
    specificity = TN/(FP+TN)
    return dice, iou, precision, recall, specificity

# def Evaluate(true_mask, pred_mask):
#     """
#     Get the DICE/IOU between each predicted mask and each true mask.

#     Inputs:
#         masks_true : array-like
#             A 2D array of shape (image_height, image_width)
#         masks_pred : array-like
#             A 2D array of shape (image_height, image_width)

#     Returns:
#         array-like
#             A 2D array of shape (n_true_masks, n_predicted_masks), where
#             the element at position (i, j) denotes the dice between the `i`th true
#             mask and the `j`th predicted mask.
#     """
#     assert true_mask.shape == pred_mask.shape, "Gt and pred must have same shape."
#     masks_true = true_mask[np.newaxis, ...]
#     masks_pred = pred_mask[np.newaxis, ...]
#     n_true_masks, height, width = masks_true.shape
#     n_pred_masks = masks_pred.shape[0]
#     m_true = masks_true.copy().reshape(n_true_masks, height * width).T
#     m_pred = masks_pred.copy().reshape(n_pred_masks, height * width)
#     numerator = np.dot(m_pred, m_true)
#     denominator = m_pred.sum(1).reshape(-1, 1) + m_true.sum(0).reshape(1, -1)
#     dice = 2*numerator / denominator
#     iou = numerator / (denominator - numerator)
#     sensitivity = numerator / m_true.sum(0).reshape(1, -1)
#     specificity = numerator / m_pred.sum(1).reshape(-1, 1)
#     return  dice, iou, sensitivity, specificity

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = sorted(glob(osp.join(dir_img, "**", "*.npy"), recursive=True))

    nets = {
        "unet": UNet,
        "inunet": InUNet,
        "attunet": AttU_Net,
        "inattunet": InAttU_Net,
        "att2uneta": Att2U_NetA,
        "att2unetb": Att2U_NetB,
        "att2unetc": Att2U_NetC,
    }
    try:
        net_type = nets[args.model_type.lower()]
        net = net_type(n_channels=1, n_classes=1, bilinear=True)
    except KeyError:
        os._exit(0)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    n = len(in_files)
    test_res = []
    print("main_id\t\t\t\tdice\t\t\tiou\t\t\tprecision\t\tsensitivity/recall\tspecificity")
    TN = TP = FP = FN = 0
    for i, fn in enumerate(in_files):
        main_id = osp.basename(fn).split(".")[0]
        logging.info("\nPredicting image {} ...".format(fn))

        img = np.load(fn)

        pred_mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device).astype(np.int)
        true_mask = np.load(fn.replace("images", "labels"))
        dice, iou, precision, recall, specificity = Evaluate(true_mask, pred_mask)
        print(f"{i+1}/{n}-{main_id}:\t{dice}\t{iou}\t{precision}\t{recall}\t{specificity}")
        test_res.append((main_id, dice, iou, precision, recall, specificity))
        if dice == 0:
            if pred_mask.sum() == 0 and true_mask.sum() != 0:
                FN += 1
            elif pred_mask.sum() != 0 and true_mask.sum() == 0:
                FP += 1
            elif pred_mask.sum() == 0 and true_mask.sum() == 0:
                TN += 1
        else:
            TP += 1
        if args.viz:
            logging.info("Visualizing results for image {}, close to continue ...".format(fn))
            plot_img_and_mask(img, mask)
    
    test_res.sort(key = lambda x:x[0])
    ids, dice, iou, precision, recall, specificity = zip(*test_res)
    dice_max = np.max(dice)
    dice_min = np.min(dice)
    dice_mean = np.mean(dice)
    dice_var = np.var(dice)
    print("dice_max: {}, dice_min: {}, dice_mean: {}, dice_var: {}".format(dice_max, dice_min, dice_mean, dice_var))
    print(f"bump TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

    writer = pd.ExcelWriter(osp.join(osp.dirname(args.model), "test_res.xlsx"))

    res = {
        "id": ids,
        "dice": dice,
        "iou": iou,
        "precisions": precision,
        "sensitivity/recall": recall,
        "specificity": specificity,
    }
    res = pd.DataFrame(res)
    res.to_excel(writer, sheet_name="result", index=False)

    writer.close()
