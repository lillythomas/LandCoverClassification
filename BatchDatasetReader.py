"""
Code ideas from https://github.com/Newmu/dcgan and tensorflow mnist dataset reader
"""
import numpy as np
import scipy.misc as misc


class BatchDatset:
    files = []
    images = []
    annotations = []
    image_filenames = []
    annotation_filenames = []
    image_basenames = []
    annotation_basenames = []    
    image_options = {}
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, records_list, image_options={}, infer_else=False):
        """
        Intialize a generic file reader with batching for list of files
        :param records_list: list of file records to read -
        sample record: {'image': f, 'annotation': annotation_file, 'filename': filename}
        :param image_options: A dictionary of options for modifying the output image
        Available options:
        resize = True/ False
        resize_size = #size of output image - does bilinear resize
        color=True/False
        """
        print("Initializing Batch Dataset Reader...")
        print(image_options)
        self.files = records_list
        self.files = sorted(self.files)    # SORT FOR INFERENCE PURPOSES ONLY. THIS IS THE SOLUTION TO BYPASS TF'S CRUDDY MEMORY ALLOCATION. SPLIT SORTED LIST OF FILES AND ITERATE IN 'BATCHES' UNTIL ALL FILES HAVE BEEN INFERRED UPON.
        self.files_restore = records_list
        self.files_restore = sorted(self.files_restore)
        self.image_options = image_options
        if infer_else == False:
            self._read_images()
        if infer_else == True:
            self._read_images_else()

    def _read_images(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        self.annotations = np.array(
            [np.expand_dims(self._transform(filename['annotation']), axis=3) for filename in self.files])
        print (self.images.shape)
        print (self.annotations.shape)

    def _read_images_else(self):
        self.__channels = True
        self.images = np.array([self._transform(filename['image']) for filename in self.files])
        self.__channels = False
        print (self.images.shape)        

    def _transform(self, filename):
        image = misc.imread(filename)
        if self.__channels and len(image.shape) < 3:  # make sure images are of shape(h,w,3)
            image = np.array([image for i in range(3)])

        if self.image_options.get("resize", False) and self.image_options["resize"]:
            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,
                                         [resize_size, resize_size], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.annotations

    def get_records_else(self):
        return self.images    
    
    def get_parts(self):
        parts = len(self.files)/10 + 1
        print("number of parts: %s" % parts)
        return parts
    
    def read_filenames(self):
        self.image_filenames = [filename['image'] for filename in self.files]
        self.annotation_filenames = [filename['annotation'] for filename in self.files]
        return self.image_filenames, self.annotation_filenames
    
    def read_filenames_else(self):
        self.image_filenames = [filename['image'] for filename in self.files]
        return self.image_filenames
    
    def read_filenames_inference_partitioned(self, part):
        if part == 0:
            file_start = 10 * part      # parts in range [1:infinity]
            file_end = file_start + 9 + 1
        else:
            file_start = 10 * part
            file_end = file_start + 10 + 1
        self.files = self.files[file_start:file_end]
        self.image_filenames = [filename['image'] for filename in self.files]
        self.annotation_filenames = [filename['annotation'] for filename in self.files]
        self.files = self.files_restore
        return self.image_filenames, self.annotation_filenames
    
    def read_filenames_inference_partitioned_else(self, part):
        if part == 0:
            file_start = 10 * part      # parts in range [1:infinity]
            file_end = file_start + 9 + 1
        else:
            file_start = 10 * part
            file_end = file_start + 10 + 1
        self.files = self.files[file_start:file_end]
        self.image_filenames = [filename['image'] for filename in self.files]
        self.files = self.files_restore
        return self.image_filenames    
    
    def read_filenames_inference_sequential(self, max_files):
        self.files = self.files[max_files:]
        self.image_filenames = [filename['image'] for filename in self.files]
        self.annotation_filenames = [filename['annotation'] for filename in self.files]
        return self.image_filenames, self.annotation_filenames

    def read_basenames(self):
        self.image_filenames, self.annotation_filenames = self.read_filenames()
        print("image filenames: %s" % self.image_filenames)
        print("annotation filenames: %s" % self.annotation_filenames)
        self.image_basenames = [filename.split("/")[-1][:-4] for filename in self.image_filenames]
        self.annotation_basenames = [filename.split("/")[-1][:-4] for filename in self.annotation_filenames]
        return self.image_basenames, self.annotation_basenames
      
    def read_basenames_sequential(self, max_files):
        self.image_filenames, self.annotation_filenames = self.read_filenames_inference_sequential(max_files)
        self.image_basenames = [filename.split("/")[-1][:-4] for filename in self.image_filenames]
        self.annotation_basenames = [filename.split("/")[-1][:-4] for filename in self.annotation_filenames]
        print("image basenames: %s" % self.image_basenames)
        print("annotation basenames: %s" % self.annotation_basenames)
        return self.image_basenames, self.annotation_basenames
    
    def read_basenames_partitioned(self, part):
        self.image_filenames, self.annotation_filenames = self.read_filenames_inference_partitioned(part)
        print("image filenames: %s" % self.image_filenames)
        print("annotation filenames: %s" % self.annotation_filenames)
        self.image_basenames = [filename.split("/")[-1][:-4] for filename in self.image_filenames]
        self.annotation_basenames = [filename.split("/")[-1][:-4] for filename in self.annotation_filenames]
        return self.image_basenames, self.annotation_basenames

    def read_basenames_partitioned_else(self, part):
        self.image_filenames = self.read_filenames_inference_partitioned_else(part)
        print("image filenames: %s" % self.image_filenames)
        self.image_basenames = [filename.split("/")[-1][:-4] for filename in self.image_filenames]
        return self.image_basenames    
    
    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.annotations = self.annotations[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.annotations[start:end]
    
    def next_batch_inference_sequential(self, max_files):
        start = max_files
        end = max_files + 20
        return self.images[start:end], self.annotations[start:end]
    
    def next_batch_inference_partitioned(self, part):
        if part == 0:
            start = 10 * part   # parts in range [1:infinity]
            end = start + 9 + 1
        else:
            start = 10 * part
            end = start + 10 + 1
        print("start que for images: ", start, "end que for images: ", end)
        return self.images[start:end], self.annotations[start:end]

    def next_batch_inference_partitioned_else(self, part):
        if part == 0:
            start = 10 * part   # parts in range [1:infinity]
            end = start + 9 + 1
        else:
            start = 10 * part
            end = start + 10 + 1
        print("start que for images: ", start, "end que for images: ", end)
        return self.images[start:end]    
    
    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.annotations[indexes]

    def get_all_batches(self, batch_size):
        indexes = np.random.randint(0, len(self.annotations), size=[batch_size]).tolist()
        self.image_basenames, self.annotation_basenames = self.read_basenames()
        print(self.image_basenames, self.annotation_basenames)
        return self.images[indexes], self.annotations[indexes]#, map(int, self.image_basenames)[indexes], map(int, self.annotation_basenames)[indexes]
