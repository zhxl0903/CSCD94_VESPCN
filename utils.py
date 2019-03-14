import cv2
import numpy as np
import tensorflow as tf
import os 
import glob
import h5py


def imread(path):

    """
        This method reads an image given path.

        Input:
        path: path of image to be read

        Returns:
        img: read image in numpy array
    """

    img = cv2.imread(path)
    return img

def imsave(image, path, config):

    """
        This method saves image given image, path, and config.

        Inputs:
        image: image to be saved
        path: path to save image in config.result_dir
        config: FLAG from main.py

    """

    # checkimage(image)
    # Checks the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(),config.result_dir)):
        os.makedirs(os.path.join(os.getcwd(),config.result_dir))

    # NOTE: because normalization, we need to multiply 255 back
    cv2.imwrite(os.path.join(os.getcwd(),path),image * 255.)

def checkimage(image):

    """
        This method generates a test image given image.

        Input:
        image: image to be displayed
    """

    cv2.imshow("test",image)
    cv2.waitKey(0)

def modcrop(img, scale =3):

    """
        This method applies modcrop to img.
        To scale down and up the original image, first thing to
        do is to have no remainder while scaling operation.

        Inputs:
        img: image to apply modcrop
        scale: downscale scale

        Returns:
        img after applying modecrop
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

    """
        This method generates the checkpoint directory based on conifg.is_train.

        Inputs:
        config: FLAG from main.py

        Returns:
        path: path of checkpoint directory
    """
    if config.is_train:
        return os.path.join('./{}'.format(config.checkpoint_dir), "train.h5")
    else:
        return os.path.join('./{}'.format(config.checkpoint_dir), "test.h5")


def preprocess(path ,scale = 3):

    """
        This method prepares labels and downscaled image given path
        of image and scale. Modcrop is used on the image label to ensure
        length and width of image is divisible by scale.

        Inputs:
        path: the image directory path
        scale: scale to downscale

        Outputs:
        input_: downscaled version of image
        label_: label after applying moderop
    """
    img = imread(path)
    
    # Crops image to ensure length and width of img is divisble by 
    # scale for resizing by scale
    label_ = modcrop(img, scale)
    
    # Resize by scaling factor
    input_ = cv2.resize(label_,None,fx = 1.0/scale ,fy = 1.0/scale,
                        interpolation = cv2.INTER_CUBIC) 

    #kernel_size = (7, 7);
    #sigma = 3.0;
    #input_ = cv2.GaussianBlur(input_, kernel_size, sigma);
    #checkimage(input_)

    return input_, label_


def prepare_data(train_mode, dataset="Train"):

    """
        This method prepares data paths of frames used for data preparation.
        Data paths in each sequence are sorted based on number in
        the frame's file name.

        Inputs:
        train_mode: 0-6
        dataset: prepares data paths for training dataset preparation iff dataset = Train

        Returns:
        dataPaths: list of lists of data paths for the sequences to be used for data preparation
    """
    
    # Defines list of data path lists for different folders of training data
    dataPaths = []
    
    # If mode is train, dataPaths from each folder in Train directory are
    # stored into a list which is then appended to dataPaths
    # Join the Train dir to current directory
    if dataset == "Train":
        data_dir = os.path.join(os.getcwd(), dataset) 
        for root, dirs, files in os.walk(data_dir):
            if dirs != []:
                for folder in dirs:
                    dataFolderDir = os.path.join(data_dir, folder)        
                    
                    # make set of all dataset file path
                    data = glob.glob(os.path.join(dataFolderDir, "*.png"))
                    
                    # Sorts by number in file name
                    data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                        os.path.basename(f)))))
                    dataPaths.append(data)
    else:

        if train_mode == 0:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset),
                                    "Mode0")
            
            # make set of all dataset file path
            data = glob.glob(os.path.join(data_dir, "*.png"))
            
            # Sorts by number in file name
            data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                       os.path.basename(f)))))

            dataPaths.append(data)
        elif train_mode == 1:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset),
                                    "Mode1")
            
            # make set of all dataset file path
            data = glob.glob(os.path.join(data_dir, "*.png"))
            
            # Sorts by number in file name
            data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                       os.path.basename(f)))))

            dataPaths.append(data)
        elif train_mode == 3:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset),
                                    "Mode3")
            
            # make set of all dataset file path
            data = glob.glob(os.path.join(data_dir, "*.png"))
            
            # Sorts by number in file name
            data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                       os.path.basename(f)))))

            dataPaths.append(data)
        elif train_mode == 2:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset),
                                    "Mode2")
            
            # make set of all dataset file path
            data = glob.glob(os.path.join(data_dir, "*.png"))
            
            # Sorts by number in file name
            data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                       os.path.basename(f)))))

            dataPaths.append(data)
        elif train_mode == 4:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset),
                                    "Mode4")
            
            # make set of all dataset file path
            data = glob.glob(os.path.join(data_dir, "*.png"))
            
            # Sorts by number in file name
            data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                       os.path.basename(f)))))

            dataPaths.append(data)
        elif train_mode == 5:
            
            # Prepares testing data paths for mode 5
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset),
                                    "Mode5")
            for root, dirs, files in os.walk(data_dir):
                if dirs:
                    for folder in dirs:
                        dataFolderDir = os.path.join(data_dir, folder)        
                        
                        # make set of all dataset file path
                        data = glob.glob(os.path.join(dataFolderDir, "*.png"))
                        
                        # Sorts by number in file name
                        data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                           os.path.basename(f)))))
                        dataPaths.append(data)
        elif train_mode == 6:
            data_dir = os.path.join(os.path.join(os.getcwd(), dataset),
                                    "Mode6")
            for root, dirs, files in os.walk(data_dir):
                if dirs:
                    for folder in dirs:
                        dataFolderDir = os.path.join(data_dir, folder)        
                        
                        # make set of all dataset file path
                        data = glob.glob(os.path.join(dataFolderDir, "*.png"))
                        
                        # Sorts by number in file name
                        data.sort(key=lambda f: int(''.join(filter(str.isdigit,
                                                            os.path.basename(f)))))
                        dataPaths.append(data)

    print(dataPaths)
    return dataPaths

def load_data(is_train, train_mode):

    """
        This method prepares file paths for data to be used for data preparation
        given is_train and train_mode.

        Inputs:
        is_train: Gets file paths for training data iff is_train=True
        train_mode: train mode

        Returns:
        data: list containing prepared file paths
    """

    if is_train:
        data = prepare_data(train_mode = train_mode, dataset="Train")
    else:
        '''if test_img != "":
            return prepare_data(dataset="Test",Input_img=test_img)'''
        
        data = prepare_data(train_mode = train_mode, dataset="Test")
    return data


def make_sub_data(data, config):

    """
        This method makes sub_data set based on train mode in config.
        If config.is_train = False, then no labels are prepared.

        Inputs:
            data : list of lists of all data paths for the frames from different sequences
            config : FLAGS from main.py

        Returns:
            sub_input_sequence: list of prepared sub_input or testing data
            sub_label_sequence: list of prepared labels if config.is_train=True ([] otherwise)
            dataPaths: dataPaths of target frames for which each image set in sub_input_sequence is prepared
    """
    sub_input_sequence = []
    sub_label_sequence = []
    
    dataPaths = []

    # Returns test data if is_train=False
    if not config.is_train:
        for lsts in data:
            
            # Sets default upper bound for image processing loop for list lsts
            ubound = len(lsts) - 1
            lbound = 0
             
            if config.train_mode == 1 or config.train_mode == 3 or \
                config.train_mode == 6:
                ubound = len(lsts)
            elif config.train_mode == 2 or config.train_mode == 5:
                ubound = len(lsts) - 1
                lbound = 1
             
            # Inits dataset list for mode 5
            dataSet = []

            # Loops over all images in lsts
            for i in range(lbound, ubound):
                print('Processing image at: ' + lsts[i])
                dataPaths.append(lsts[i])
                 
                input_data = []
                if config.train_mode == 0:
                        
                    # Prepares test frame set using frame i and frame i+1
                    input_ = imread(lsts[i]) / 255.0
                    inputNext_ = imread(lsts[i+1]) / 255.0
                    input_data = np.dstack((input_, inputNext_))
                elif config.train_mode == 1 or config.train_mode == 3 \
                        or config.train_mode == 4 or config.train_mode == 6:
                        
                    # Prepares test frame set using frame i
                    input_ = imread(lsts[i]) / 255.0
                    input_data = input_
                elif config.train_mode == 2 or config.train_mode == 5:
                     
                    # Prepares test frame set using frame i, frame i+1
                    # frame i-1
                    input_ = imread(lsts[i]) / 255.0
                    inputNext_ = imread(lsts[i+1]) / 255.0
                    inputPrev_ = imread(lsts[i-1]) / 255.0
                     
                    input_data = np.dstack((input_, inputPrev_, inputNext_))
                 
                if config.train_mode != 5 and config.train_mode != 6:
                    sub_input_sequence.append(input_data)
                else:
                    dataSet.append(input_data)
            if config.train_mode == 5 or config.train_mode == 6:
                sub_input_sequence.append(dataSet)
        return sub_input_sequence, sub_label_sequence, dataPaths

    # Prepares training data if is_train
    for lsts in data:
        
        # Sets default upper bound for image processing loop for list lsts
        ubound = len(lsts) - 1
        lbound = 1

        # Sets bound to loop over all images in data if train mode is 1
        if config.train_mode == 1 or config.train_mode == 4:
            ubound = len(lsts)
            lbound = 0
            
        for i in range(lbound, ubound):
            
            # Performs resize of 3 neighbouring images using bicubic
            # Labels are generated for current frame image
            # do bicbuic downscaling
            print('Processing image at: ' + lsts[i]) 
            input_, label_, = preprocess(lsts[i], config.scale) 
            
            if config.train_mode == 0 or config.train_mode == 2:
                input_prev, _ = preprocess(lsts[i-1], config.scale)
                input_next, _ = preprocess(lsts[i+1], config.scale)

            if len(input_.shape) == 3:

                # is color
                h, w, c = input_.shape
            else:

                # is grayscale
                h, w = input_.shape
                
            # NOTE: make subimage of LR and HR
            # Input 
            for x in range(0, h - config.image_size + 1, config.stride):
                for y in range(0, w - config.image_size + 1, config.stride):

                    # Retrieves sub frames from prev and next frames if
                    # train mode is 0 or 2
                    if config.train_mode == 0 or config.train_mode == 2:
                        sub_input_prev = input_prev[x: x + config.image_size,
                                                    y: y + config.image_size]
                        sub_input_next = input_next[x: x + config.image_size,
                                                    y: y + config.image_size]
    
                        # Reshapes subframes from prev and next frames if
                        # train_mode is 0 or 2
                        sub_input_prev = \
                            sub_input_prev.reshape([config.image_size,
                                                    config.image_size,
                                                    config.c_dim])
                        sub_input_next = \
                            sub_input_next.reshape([config.image_size,
                                                    config.image_size,
                                                    config.c_dim])
                        
                        # Normalizes sub input frames
                        sub_input_prev = sub_input_prev / 255.0
                        sub_input_next = sub_input_next / 255.0
                    
                    # Retrieves sub frames from current frame
                    sub_input = input_[x: x + config.image_size,
                                       y: y + config.image_size]
                    
                    # Reshapes subinput
                    sub_input = sub_input.reshape([config.image_size,
                                                   config.image_size,
                                                   config.c_dim])
                    # Normializes sub_input
                    sub_input = sub_input / 255.0

                    if config.train_mode == 0:
                        
                        # Prepares one frame pair if train_mode == 0
                        sub_curr_prev = np.dstack((sub_input, sub_input_prev))
                        
                        # Prepares sub_input_data of shape (h , w , 2*c_dim)
                        sub_input_data = np.array(sub_curr_prev)
                        del sub_curr_prev
                        del sub_input
                    elif config.train_mode == 1 or config.train_mode == 4:
                        
                        # Obtains sub images for training subpixel convnet
                        sub_input_data = np.array(sub_input)
                        del sub_input
                    else:
                        
                        # Prepares subframe tensors curr-prev frames and 
                        # curr-next frames
                        # Each tensor is of shape (h, w, (3*c_dim))
                        sub_curr_prev_next = np.dstack((sub_input,
                                                        sub_input_prev,
                                                        sub_input_next))

                        # Prepares sub_input_data of shape (2, l, w, 2*c_dim)
                        sub_input_data = np.array(sub_curr_prev_next)
                        del sub_input
                        del sub_input_next
                        del sub_input_prev
                        del sub_curr_prev_next
                    
                    # Adds to sequence
                    sub_input_sequence.append(sub_input_data)
                    del sub_input_data
    
            # Prepares labels based on train_mode
            if config.train_mode != 0:
                for x in range(0,
                               h * config.scale -
                               config.image_size * config.scale + 1,
                               config.stride * config.scale):
                    for y in range(0,
                                   w * config.scale -
                                   config.image_size * config.scale + 1,
                                   config.stride * config.scale):
                        
                        # Sets label to piece from original image 
                        # if train_mode is 1 or 2
                        sub_label = label_[x: x + config.image_size 
                                           * config.scale,
                                           y: y + config.image_size 
                                           * config.scale]
                        sub_label = sub_label.reshape([config.image_size 
                                                       * config.scale,
                                                       config.image_size 
                                                       * config.scale,
                                                       config.c_dim])
                        
                        # Normalizes sub_label
                        sub_label = sub_label / 255.0
                        
                        # Adds to sequence
                        sub_label_sequence.append(sub_label)
                        del sub_label
            else:
                for x in range(0, h - config.image_size + 1,
                               config.stride):
                    for y in range(0, w - config.image_size + 1,
                                   config.stride):
                        
                        # Sets target image to label if train mode is 0
                        sub_label = input_[x: x + config.image_size,
                                           y: y + config.image_size]
                        sub_label = sub_label.reshape([config.image_size,
                                                       config.image_size,
                                                       config.c_dim])
                        
                        # Normalizes sub_label
                        sub_label = sub_label / 255.0
                        
                        # Adds to sequence
                        sub_label_sequence.append(sub_label)
                        del sub_label

    return sub_input_sequence, sub_label_sequence, dataPaths


def read_data(path, config):

    """
        Reads h5 format data file from path. Mode 5 and 6 requires
        additional loading of image paths and number of image sets.

        Inputs:
        path: file path of h5 file
        config: config from FLAGS of main.py

        Returns:
        input_: loaded inputs
        label_: loaded labels
        dataPaths: loaded dataPaths

    """
    dataPaths = []
    input_ = []

    # Prepares path of image path file and number of image sets file
    fileDir = os.path.split(path)[0]
    numImgSetsDir = os.path.join(fileDir, 'numImgSets.txt')
    pathsDir = os.path.join(fileDir, 'paths.txt')

    # Loads image paths and number of image sets from files if train mode is 5 or 6
    if config.train_mode == 5 or config.train_mode == 6:
        
        print('Loading number of image sets and image paths...')
        
        # Gets number of image sets from file
        f = open(numImgSetsDir, 'r')
        numImgSets = int((f.readline()).strip('\n'))
        f.close()
        
        # Loads image paths from file
        f = open(pathsDir, 'r')
        paths = f.readlines()

        # Loads list of image paths
        for i in range(len(paths)):
            dataPaths.append(paths[i].strip('\n'))
        f.close()
        
    with h5py.File(path, 'r') as hf:
        
        if config.train_mode != 5 and config.train_mode != 6:

            # Loads input from hf file
            input_ = np.array(hf.get('input'))
        else:
            
            # Loads image sets from hf file and appends to input_ list
            for i in range(numImgSets):
                 
                print('Loading data set ' + str(i) + ' ...')
                input_.append(np.array(hf.get('input' + str(i))))
            
        label_ = np.array(hf.get('label'))

    return input_, label_, dataPaths


def make_data_hf(input_, label_, dataPaths, config):

    """
        This method makes training or testing as h5 file format. Mode 5 and Mode 6
        multi-dir modes require additional saves of dataPaths and number of image set
        numpy arrays in input_ list.

        Inputs:
        input_: inputs to be saved
        label_: labels to be saved
        dataPaths: dataPaths of data from which each sample in input_ is prepared
        config: config from FLAGS of main.py

    """
    
    # Check the check dir, if not, create one
    if not os.path.isdir(os.path.join(os.getcwd(), config.checkpoint_dir)):
        os.makedirs(os.path.join(os.getcwd(), config.checkpoint_dir))

    if config.is_train:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir 
                                + '/train.h5')
    else:
        savepath = os.path.join(os.getcwd(), config.checkpoint_dir 
                                + '/test.h5')
    
    print('Saving prepared data...')
    with h5py.File(savepath, 'w') as hf:

        # Creates single dataset for input_ and label_ if mode is not multi-dir
        if config.train_mode != 5 and config.train_mode != 6:
            hf.create_dataset('input', data=input_)
            hf.create_dataset('label', data=label_)
        else:
            
            # Saves each image set
            for i in range(len(input_)):       
                print('Saving image set ' + str(i) + ' ...')
                hf.create_dataset('input'+str(i), data=input_[i])
            
            hf.create_dataset('label', data=label_)
    
    # Saves dataPaths and number of imageSets
    if config.train_mode == 5 or config.train_mode == 6:
        
        print('Saving image paths and number of image sets...')
        
        # Saves image paths to file
        f = open(os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'paths.txt'), 'w')
        for i in range(len(dataPaths)):
            f.write(dataPaths[i]+'\n')
        f.close()

        # Saves number of image sets to file
        f = open(os.path.join(os.path.join(os.getcwd(), config.checkpoint_dir), 'numImgSets.txt'), 'w')
        f.write(str(len(input_))+'\n')
        f.close()    


def input_setup(config):

    """
       This method reads video sequences and prepares training or testing data,
       which are saved as h5 file format.

       Input:
       config: config from FLAGS of main.py

    """

    # Load data path, if is_train False, get test data
    print('Loading data to be prepared...')
    data = load_data(config.is_train, config.train_mode)

    # Make sub_input and sub_label, if is_train false more return nx, ny
    print('Preparing data...')
    sub_input_sequence, sub_label_sequence, dataPaths = make_sub_data(data, config)
    
    # Turn input data list to numpy array. With this transform
    # training mode = 0 => [?, size, size, 6] (training) or [?, h, w, 6] (testing)
    # training mode = 2 => [?, size, size, 9] (training) or [?, h, w, 9] (testing)
    # training mode = 1 => [? size, size, 3] (training) or [?, h, w, 3] (testing)
    # training mode = 3 or 4 =>[?, size, size, 3] (training) or [?, h, w, 3] (testing)
    # training mode = 5 (test only) => list of np arrays of shape: [?, size, size, 9]
    # training mode = 6 (test only) => list of np arrays of shape: [?, size, size, 3]
    if config.train_mode != 5 and config.train_mode != 6:
        arrinput = np.asarray(sub_input_sequence)
    else:
        
        # Prepares input data from array in sub_input_sequence list
        arrinput = []
        for i in range(len(sub_input_sequence)):
            arrinput.append(np.asarray(sub_input_sequence[i]))
    
    # [?, size , size, 3] (training) or [?, h, w, 3] (testing)
    arrlabel = np.asarray(sub_label_sequence) 
    
    # Prints shapes of prepared input data if training_mode != 5 or 6
    if config.train_mode != 5 and config.train_mode != 6:
        print('Input data shape: ', arrinput.shape)
        print('Labels data shape: ', arrlabel.shape)
    make_data_hf(arrinput, arrlabel, dataPaths, config)

def bilinear_sampler(x, v, resize=False, normalize=False, crop=None, out="CONSTANT"):
  """
    Args:
      x - Input tensor [N, H, W, C]
      v - Vector flow tensor [N, H, W, 2], tf.float32
      (optional)
      resize - Whether to resize v as same size as x
      normalize - Whether to normalize v from scale 1 to H (or W).
                  h : [-1, 1] -> [-H/2, H/2]
                  w : [-1, 1] -> [-W/2, W/2]
      crop - Setting the region to sample. 4-d list [h0, h1, w0, w1]
      out  - Handling out of boundary value.
             Zero value is used if out="CONSTANT".
             Boundary values are used if out="EDGE".
  """

  def _get_grid_array(N, H, W, h, w):
    N_i = tf.range(N)
    H_i = tf.range(h+1, h+H+1)
    W_i = tf.range(w+1, w+W+1)
    n, h, w, = tf.meshgrid(N_i, H_i, W_i, indexing='ij')
    n = tf.expand_dims(n, axis=3) # [N, H, W, 1]
    h = tf.expand_dims(h, axis=3) # [N, H, W, 1]
    w = tf.expand_dims(w, axis=3) # [N, H, W, 1]
    n = tf.cast(n, tf.float32) # [N, H, W, 1]
    h = tf.cast(h, tf.float32) # [N, H, W, 1]
    w = tf.cast(w, tf.float32) # [N, H, W, 1]

    return n, h, w

  shape = tf.shape(x) # TRY : Dynamic shape
  N = shape[0]
  if crop is None:
    H_ = H = shape[1]
    W_ = W = shape[2]
    h = w = 0
  else :
    H_ = shape[1]
    W_ = shape[2]
    H = crop[1] - crop[0]
    W = crop[3] - crop[2]
    h = crop[0]
    w = crop[2]

  if resize:
    if callable(resize) :
      v = resize(v, [H, W])
    else :
      v = tf.image.resize_bilinear(v, [H, W])

  if out == "CONSTANT":
    x = tf.pad(x,
      ((0,0), (1,1), (1,1), (0,0)), mode='CONSTANT')
  elif out == "EDGE":
    x = tf.pad(x,
      ((0,0), (1,1), (1,1), (0,0)), mode='REFLECT')

  vy, vx = tf.split(v, 2, axis=3)
  if normalize :
    vy *= tf.cast(H / 2, tf.float32)
    vx *= tf.cast(W / 2, tf.float32)

  n, h, w = _get_grid_array(N, H, W, h, w) # [N, H, W, 3]

  vx0 = tf.floor(vx)
  vy0 = tf.floor(vy)
  vx1 = vx0 + 1
  vy1 = vy0 + 1 # [N, H, W, 1]

  H_1 = tf.cast(H_+1, tf.float32)
  W_1 = tf.cast(W_+1, tf.float32)
  iy0 = tf.clip_by_value(vy0 + h, 0., H_1)
  iy1 = tf.clip_by_value(vy1 + h, 0., H_1)
  ix0 = tf.clip_by_value(vx0 + w, 0., W_1)
  ix1 = tf.clip_by_value(vx1 + w, 0., W_1)

  i00 = tf.concat([n, iy0, ix0], 3)
  i01 = tf.concat([n, iy1, ix0], 3)
  i10 = tf.concat([n, iy0, ix1], 3)
  i11 = tf.concat([n, iy1, ix1], 3) # [N, H, W, 3]
  i00 = tf.cast(i00, tf.int32)
  i01 = tf.cast(i01, tf.int32)
  i10 = tf.cast(i10, tf.int32)
  i11 = tf.cast(i11, tf.int32)

  x00 = tf.gather_nd(x, i00)
  x01 = tf.gather_nd(x, i01)
  x10 = tf.gather_nd(x, i10)
  x11 = tf.gather_nd(x, i11)
  w00 = tf.cast((vx1 - vx) * (vy1 - vy), tf.float32)
  w01 = tf.cast((vx1 - vx) * (vy - vy0), tf.float32)
  w10 = tf.cast((vx - vx0) * (vy1 - vy), tf.float32)
  w11 = tf.cast((vx - vx0) * (vy - vy0), tf.float32)
  output = tf.add_n([w00*x00, w01*x01, w10*x10, w11*x11])

  return output

