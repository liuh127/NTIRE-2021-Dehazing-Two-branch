import torch
import time
import argparse
from model import fusion_refine,Discriminator
from train_dataset import dehaze_train_dataset
from test_dataset import dehaze_test_dataset
from val_dataset import dehaze_val_dataset
from torch.utils.data import DataLoader
import os
from torchvision.models import vgg16
from utils_test import to_psnr,to_ssim_skimage
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from perceptual import LossNetwork
from torchvision.utils import save_image as imwrite
from pytorch_msssim import msssim
# --- Parse hyper-parameters train --- #
parser = argparse.ArgumentParser(description='RCAN-Dehaze-teacher')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=20, type=int)
parser.add_argument('-train_epoch', help='Set the training epoch', default=10000, type=int)
parser.add_argument('--train_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--model_save_dir', type=str, default='./output_result')
parser.add_argument('--log_dir', type=str, default=None)
# --- Parse hyper-parameters test --- #
parser.add_argument('--test_dataset', type=str, default='')
parser.add_argument('--predict_result', type=str, default='./output_result/picture/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1,  type=int)
parser.add_argument('--vgg_model', default='', type=str, help='load trained model or not')
parser.add_argument('--imagenet_model', default='', type=str, help='load trained model or not')
parser.add_argument('--rcan_model', default='', type=str, help='load trained model or not')
args = parser.parse_args()

val_dataset = os.path.join(args.data_dir, 'NTIRE2021_Test_Hazy')
predict_result= args.predict_result
test_batch_size=args.test_batch_size

# --- output picture and check point --- #
if not os.path.exists(args.model_save_dir):
    os.makedirs(args.model_save_dir)
output_dir=os.path.join(args.model_save_dir,'')

# --- Gpu device --- #
device_ids = [Id for Id in range(torch.cuda.device_count())]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
MyEnsembleNet = fusion_refine(args.imagenet_model, args.rcan_model)
print('MyEnsembleNet parameters:', sum(param.numel() for param in MyEnsembleNet.parameters()))


val_dataset = dehaze_val_dataset(val_dataset)
val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0)

# --- Multi-GPU --- #
MyEnsembleNet = MyEnsembleNet.to(device)
MyEnsembleNet= torch.nn.DataParallel(MyEnsembleNet, device_ids=device_ids)


# --- Load the network weight --- #
try:
    MyEnsembleNet.load_state_dict(torch.load( 'best.pkl'))
    print('--- weight loaded ---')
except:
    print('--- no weight loaded ---')

# --- Strat testing --- #
with torch.no_grad():
    img_list = []
    time_list = []
    MyEnsembleNet.eval()
    imsave_dir = output_dir
    if not os.path.exists(imsave_dir):
        os.makedirs(imsave_dir)
    for batch_idx, hazy in enumerate(val_loader):
        # print(len(val_loader))
        start = time.time()
        hazy = hazy.to(device)
        
        img_tensor = MyEnsembleNet(hazy)

        end = time.time()
        time_list.append((end - start))
        img_list.append(img_tensor)

        imwrite(img_list[batch_idx], os.path.join(imsave_dir, str(batch_idx)+'.png'))
    time_cost = float(sum(time_list) / len(time_list))
    print('running time per image: ', time_cost)
                

# writer.close()








