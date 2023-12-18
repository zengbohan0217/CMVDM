import torch
from torch import nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image
from src.utils import *
from src.config_dec import *
from src.utils.misc import set_gpu


def pure_test(data_loader_labeled, dec, save_folder='./vis_digit_0818'):
    os.makedirs(save_folder, exist_ok=True)
    dec.eval().cuda()

    main_loader = data_loader_labeled
    for batch_idx, (images_gt, fmri_gt) in enumerate(main_loader):
        images_gt, fmri_gt = map(lambda x: x.cuda(), [images_gt, fmri_gt])
        with torch.no_grad():
            images_D = dec(fmri_gt)
        pred_image_tensors = images_D.cpu()
        gt_image_tensors = images_gt.cpu()
        
        for index, image_tensor in enumerate(pred_image_tensors):
            save_image(torch.cat([gt_image_tensors[index], image_tensor], dim=1), pjoin(save_folder, str(index) + '_gt_gen.png'))

def run_god_test(dec_path):
    
    # dataset / dataloader
    img_xfm_basic = transforms.Compose([transforms.Resize(size=112, interpolation=Image.BILINEAR), transforms.CenterCrop(112), transforms.ToTensor()])
    val_labeled_avg = KamitaniDataset_discussion(fmri_xfm=np.float32, subset_case=KamitaniDataset_discussion.TEST)
    val_labeled_avg = CustomDataset(val_labeled_avg, input_xfm=img_xfm_basic)
    data_loaders_labeled = {
        'test': data.DataLoader(val_labeled_avg, batch_size=min([24,16,48,50][-1], len(val_labeled_avg)), shuffle=False, num_workers=7, pin_memory=True),
    }
    
    # init fmri-decoder
    dec = make_model('BaseDecoder', 9919, 112, start_CHW=(64, 14, 14), n_conv_layers_ramp=3, n_chan=64, n_chan_output=3, depth_extractor=None)
    # Load pretrained encoder
    dec = nn.DataParallel(dec)
    assert os.path.isfile(dec_path)
    print('\t==> Loading checkpoint {}'.format(os.path.basename(dec_path)))
    dec.load_state_dict(torch.load(dec_path)['state_dict'])
    
    pure_test(data_loaders_labeled['test'], dec)

def depth_infer(fmri, 
                dec_path,
                save_folder='./vis_depth'):

    os.makedirs(save_folder, exist_ok=True)

     # init fmri-decoder
    dec = make_model('BaseDecoder', 4643, 112, start_CHW=(64, 14, 14), n_conv_layers_ramp=3, n_chan=64, n_chan_output=4, depth_extractor=None)
    # Load pretrained encoder
    dec = nn.DataParallel(dec)
    assert os.path.isfile(dec_path)
    print('\t==> Loading checkpoint {}'.format(os.path.basename(dec_path)))
    dec.load_state_dict(torch.load(dec_path)['state_dict'])

    dec.eval().cuda()
    with torch.no_grad():
        images_D = dec(fmri)
    





if __name__ == '__main__':
    run_god_test()