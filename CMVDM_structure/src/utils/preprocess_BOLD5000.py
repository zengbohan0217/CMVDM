import os
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

def canny_contour(images):
    ret = []
    for image in tqdm(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contour_img = image.copy()
        contour_img.fill(255)
        edges = cv2.Canny(gray, 100, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_img, contours, -1, (0,255,0),5)
        pil_image = Image.fromarray(contour_img)
        pil_image = pil_image.convert("RGB")

        ret.append(pil_image)

    return ret

def canny_edge(images):
    ret = []
    for image in tqdm(images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        pil_image = Image.fromarray(edges)
        pil_image = pil_image.convert("RGB")

        ret.append(pil_image)

    return ret


def list_get_all_index(list, value):
    return [i for i, v in enumerate(list) if v == value]

def identity(x):
    return x

def normalize(x, mean=None, std=None):
    mean = np.mean(x) if mean is None else mean
    std = np.std(x) if std is None else std
    return (x - mean) / (std * 1.0)

def get_stimuli_list(root, sub):
    sti_name = []
    path = os.path.join(root, 'Stimuli_Presentation_Lists', sub)
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        if not os.path.isdir(os.path.join(path, folder)):
            continue
        files = os.listdir(os.path.join(path, folder))
        files.sort()
        for file in files:
            if file.endswith('.txt'):
                sti_name += list(np.loadtxt(os.path.join(path, folder, file), dtype=str))

    sti_name_to_return = []
    for name in sti_name:
        if name.startswith('rep_'):
            name = name.replace('rep_', '', 1)
        sti_name_to_return.append(name)
    return sti_name_to_return


def create_BOLD5000_dataset(path='./data/BOLD5000', fmri_transform=identity,
            image_transform=identity, subjects = ['CSI1', 'CSI2', 'CSI3', 'CSI4'], include_nonavg_test=False):
    roi_list = ['EarlyVis', 'LOC', 'OPA', 'PPA', 'RSC']
    fmri_path = os.path.join(path, 'BOLD5000_GLMsingle_ROI_betas/py')
    img_path = os.path.join(path, 'BOLD5000_Stimuli')

    imgs_dict = np.load(os.path.join(img_path, 'Scene_Stimuli/Presented_Stimuli/img_dict.npy'),allow_pickle=True).item()
    repeated_imgs_list = np.loadtxt(os.path.join(img_path, 'Scene_Stimuli', 'repeated_stimuli_113_list.txt'), dtype=str)

    fmri_files = [f for f in os.listdir(fmri_path) if f.endswith('.npy')]
    fmri_files.sort()
    
    fmri_train_major = []
    fmri_test_major = []
    img_train_major = []
    img_test_major = []
    for sub in subjects:
        # load fmri
        fmri_data_sub = []
        for roi in roi_list:
            for npy in fmri_files:
                if npy.endswith('.npy') and sub in npy and roi in npy:
                    fmri_data_sub.append(np.load(os.path.join(fmri_path, npy)))
        fmri_data_sub = np.concatenate(fmri_data_sub, axis=-1) # concatenate all rois
        
      
        # load image
        img_files = get_stimuli_list(img_path, sub)
        img_data_sub = [imgs_dict[name] for name in img_files]
        
        # split train test
        test_idx = [list_get_all_index(img_files, img) for img in repeated_imgs_list]
        test_idx = [i for i in test_idx if len(i) > 0] # remove empy list for CSI4
        test_fmri = np.stack([fmri_data_sub[idx].mean(axis=0) for idx in test_idx])
        test_img = np.stack([img_data_sub[idx[0]] for idx in test_idx])
        
        test_idx_flatten = []
        for idx in test_idx:
            test_idx_flatten += idx # flatten
        if include_nonavg_test:
            test_fmri = np.concatenate([test_fmri, fmri_data_sub[test_idx_flatten]], axis=0)
            test_img = np.concatenate([test_img, np.stack([img_data_sub[idx] for idx in test_idx_flatten])], axis=0)

        train_idx = [i for i in range(len(img_files)) if i not in test_idx_flatten]
        train_img = np.stack([img_data_sub[idx] for idx in train_idx])
        train_fmri = fmri_data_sub[train_idx]

        fmri_train_major.append(train_fmri)
        fmri_test_major.append(test_fmri)
        img_train_major.append(train_img)
        img_test_major.append(test_img)
        break

    num_voxels = fmri_train_major[0].shape[-1]
    img_train = img_train_major[0]
    img_test = img_test_major[0]

    img_train = []
    for img in img_train_major[0]:
        img_train.append(Image.fromarray(img.astype(np.uint8)))

    img_test = []
    for img in img_test_major[0]:
        img_test.append(Image.fromarray(img.astype(np.uint8)))

    data_dict = {'num_voxels': num_voxels, 'fmri_train': fmri_train_major[0], 'fmri_test': fmri_test_major[0],
                 'img_train': img_train, 'img_test': img_test}
    np.save(os.path.join(path, 'preprocessed_bold5000_data_rgb.npy'), data_dict)

if __name__ == "__main__":
    create_BOLD5000_dataset(path='/path/to/Research/SiluetteExtraction/src/data')
    print("Done!")