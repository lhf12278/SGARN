import os
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import math
import numpy as np
import cv2

scene_directory = r'/data/Test/EXTRA'
test_file = 'hdr'

list = os.listdir(scene_directory)

for scene in range(len(list)):
    # Load the image
    # Read Expo times in scene
    expoTimes = ReadExpoTimes(os.path.join(scene_directory, list[scene], 'exposure.txt'))
    # Read Image in scene
    imgs = ReadImages(list_all_files_sorted(os.path.join(scene_directory, list[scene]) ,'.tif'))
    # inputs-process
    pre_img0 = LDR_to_HDR(imgs[0], expoTimes[0], 2.2)
    pre_img1 = LDR_to_HDR(imgs[1], expoTimes[1], 2.2)
    pre_img2 = LDR_to_HDR(imgs[2], expoTimes[2], 2.2)
    output0 = np.concatenate((imgs[0], pre_img0), 2)
    output1 = np.concatenate((imgs[1], pre_img1), 2)
    output2 = np.concatenate((imgs[2], pre_img2), 2)
    # label-process
    im1 = torch.Tensor(output0).to(device)
    im1 = torch.unsqueeze(im1, 0).permute(0, 3, 1, 2)

    im2 = torch.Tensor(output1).to(device)
    im2 = torch.unsqueeze(im2, 0).permute(0, 3, 1, 2)

    im3 = torch.Tensor(output2).to(device)
    im3 = torch.unsqueeze(im3, 0).permute(0, 3, 1, 2)

    # Load the pre-trained model
    model = BASENet().to(device)
    model.eval()
    model.load_state_dict(torch.load('../Model/{}.pkl'.format(test_file)))

    # Run
    with torch.no_grad():
        pre , fusion_gradient = model(im1, im2, im3)
    pre = torch.clamp(pre, 0., 1.)
    pre = pre.permute(0, 2, 3, 1)
    pre = pre.data[0].cpu().numpy()
    output=cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
    cv2.imwrite('../test_result/{}.hdr'.format(scene), output)
