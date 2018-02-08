import cv2
import numpy as np
import tensorflow as tf
import os 
import glob
import h5py



# Get the Image
def imread(path):
    img = cv2.imread(path)
    return img

def imsave(image, path, config):
    #checkimage(image)
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))

    # NOTE: because normial, we need mutlify 255 back    
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)

def checkimage(image):
    cv2.imshow("test",image)
    cv2.waitKey(0)

def modcrop(img, scale =3):
    """
        To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is grayscale
    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = int((h // scale) * scale)
        w = int((w // scale) * scale)
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = int( (h // scale) * scale)
        w = int(( w // scale) * scale)
        img = img[0:h, 0:w]
    return img

def checkpoint_dir(config):
    if config.is_train:
        return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
        return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")

def preprocess(path ,scale = 3):
    """
        Args:
            path: the image directory path
            scale: the image need to scale 
    """
    img = imread(path)

    label_ = modcrop(img, scale)
    
    input_ = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale, interpolation = cv2.INTER_CUBIC) # Resize by scaling factor

    kernel_size = (7, 7);
    sigma = 3.0;
    #input_ = cv2.GaussianBlur(input_, kernel_size, sigma);
    #checkimage(input_)

    return input_, label_

def prepare_data(dataset="Train",Input_img=""):
    """
        Args:
            dataset: choose train dataset or test dataset
            For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp',..., 't99.bmp']
    """
    
    # Defines list of data path lists for different folders of training data
    dataPaths = []
    
    # If mode is train, dataPaths from each folder in Train directory are
    # stored into a list which is then appended to dataPaths
    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), dataset) # Join the Train dir to current directory
        for root, dirs, files in os.walk(data_dir):
            if dirs != []:
                for folder in dirs:
                    dataFolderDir = os.path.join(data_dir, folder)             
                    data = glob.glob(os.path.join(dataFolderDir, "*.bmp")) # make set of all dataset file path
                    dataPaths.append(data)
    else:
        '''if Input_img !="":
            data = [os.path.join(os.getcwd(),Input_img)]'''
        
        data_dir = os.path.join(os.path.join(os.getcwd(), dataset), "Mode1")
        data = glob.glob(os.path.join(data_dir, "*.bmp")) # make set of all dataset file path
        dataPaths.append(data)
    print(dataPaths)
    return dataPaths

def load_data(is_train, train_mode, test_img):
    if is_train:
        data = prepare_data(dataset="Train")
    else:
        '''if test_img != "":
            return prepare_data(dataset="Test",Input_img=test_img)'''
        
        data = prepare_data(dataset="Test")
    return data

def make_sub_data(data, config):
    """
        Make the sub_data set
        Args:
            data : the set of all file path 
            config : the all flags
    """
    sub_input_sequence = []
    sub_label_sequence = []
    for lsts in data:
        for i in range(1, len(lsts)-1):
            
            # Performs resize of 3 neighbouring images using bicubic
            # Labels are generated for current frame image
            input_, label_, = preprocess(lsts[i], config.scale) # do bicbuic
            input_prev, _ = preprocess(lsts[i-1], config.scale)
            input_next, _ = preprocess(lsts[i+1], config.scale)
            
            if len(input_.shape) == 3: # is color
                h, w, c = input_.shape
            else:
                h, w = input_.shape # is grayscale
            
            # Returns test data if we are not training
            if not config.is_train:
                
                input_data = []
                if config.train_mode == 0:
                    inputPrev_ = imread(lsts[0])/255.0
                    input_ = imread(lsts[1])/255.0
                    inputNext_ = imread(lsts[2])/255.0
                    input_data = np.dstack((input_, inputNext_))
                              
                sub_input_sequence.append(input_data)
                return sub_input_sequence, sub_label_sequence
    
            # NOTE: make subimage of LR and HR
            # Input 
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):
    
                    sub_input = input_[x: x + config.image_size, y: y + config.image_size] # 17 * 17
                    sub_input_prev = input_prev[x: x + config.image_size, y: y + config.image_size]
                    sub_input_next = input_next[x: x + config.image_size, y: y + config.image_size]
                    
                    # Reshape the subinput and sublabel
                    sub_input = sub_input.reshape([config.image_size, config.image_size, config.c_dim])
                    sub_input_prev = sub_input_prev.reshape([config.image_size, config.image_size, config.c_dim])
                    sub_input_next = sub_input_next.reshape([config.image_size, config.image_size, config.c_dim])
                    
                    # Normialize
                    sub_input =  sub_input / 255.0
                    sub_input_prev = sub_input_prev / 255.0
                    sub_input_next = sub_input_next / 255.0
                    
                    if config.train_mode == 0:
                        
                        # Prepares one frame pair if train_mode == 0
                        sub_curr_prev = np.dstack((sub_input, sub_input_prev))
                        
                        # Prepares sub_input_data of dimension [l x w x 2*c_dim]
                        sub_input_data = np.array(sub_curr_prev)
                    elif config.train_mode == 1:
                        sub_curr_prev_next = np.dstack((sub_input, sub_input_prev, sub_input_next))
                        sub_input_data = np.array(sub_curr_prev_next)
                    else:
                        
                        # Prepares subframe tensors curr-prev frames and curr-next frames
                        # Each tensor is of dimension h x w x (2*c_dim)
                        sub_curr_prev = np.dstack((sub_input, sub_input_prev))
                        sub_curr_next = np.dstack((sub_input, sub_input_next))
                        
                        # Prepares sub_input_data of dimension [2 x l x w x 2*c_dim]
                        sub_input_data = np.array([sub_curr_prev, sub_curr_next])
                    
                    # Add to sequence
                    sub_input_sequence.append(sub_input_data)
    
            # Label (the time of scale)
            if config.train_mode != 0:
                for x in range(0, h * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                    for y in range(0, w * config.scale - config.image_size * config.scale + 1, config.stride * config.scale):
                        
                        # Sets label to piece from original image if train_mode is 1 or 2
                        sub_label = label_[x: x + config.image_size * config.scale, y: y + config.image_size * config.scale] # 17r * 17r
                        sub_label = sub_label.reshape([config.image_size * config.scale , config.image_size * config.scale, config.c_dim])
                        
                        # Normialize
                        sub_label =  sub_label / 255.0
                        
                        # Add to sequence
                        sub_label_sequence.append(sub_label)
            else:
                for x in range(0, h - config.image_size + 1, config.stride):
                    for y in range(0, w - config.image_size + 1, config.stride):
                        
                        # Sets target image to label if train mode is 0
                        sub_label = input_[x: x + config.image_size, y: y + config.image_size]
                        sub_label = sub_label.reshape([config.image_size , config.image_size, config.c_dim])
                        
                        # Normialize
                        sub_label =  sub_label / 255.0
                        
                        # Add to sequence
                        sub_label_sequence.append(sub_label)

    return sub_input_sequence, sub_label_sequence


def read_data(path):
    """
        Read h5 format data file

        Args:
            path: file path of desired file
            data: '.h5' file format that contains  input values
            label: '.h5' file format that contains label values 
    """
    with h5py.File(path, 'r') as hf:
        input_ = np.array(hf.get('input'))
        label_ = np.array(hf.get('label'))
        return input_, label_

def make_data_hf(input_, label_, config):
    """
        Make input data as h5 file format
        Depending on "is_train" (flag value), savepath would be change.
    """
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir + '/test.h5')

    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('input', data=input_)
        hf.create_dataset('label', data=label_)

def input_setup(config):
    """
        Read image files and make their sub-images and saved them as a h5 file format
    """

    # Load data path, if is_train False, get test data
    data = load_data(config.is_train, config.train_mode, None)

    # Make sub_input and sub_label, if is_train false more return nx, ny
    sub_input_sequence, sub_label_sequence = make_sub_data(data, config)
    
    # Make list to numpy array. With this transform
    # training mode = 0 => [?, size, size, 6]
    # training mode = 2 => [?, 2, size, size, 3]
    # training mode = 1 => [? size, size, 9]
    arrinput = np.asarray(sub_input_sequence) 
    
    # [?, size , size, 3]
    arrlabel = np.asarray(sub_label_sequence) 
    
    print('Input data shape: ', arrinput.shape)
    print('Labels data shape: ', arrlabel.shape)
    make_data_hf(arrinput, arrlabel, config)

