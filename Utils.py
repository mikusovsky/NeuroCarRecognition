import numpy as np
import tensorflow as tf

def car_models_to_image_class(carModels, shape):
    i = 0
    ret_images = []
    ret_labels = []
    countClasses = len(carModels)
    for model in carModels:
        images = model.getImages(shape=shape)
        if images is not None:
            label = i#class_to_vector(i, countClasses)
            #label = np.reshape(label, (1, countClasses))

            ret_images.append(images)
            for j in range(0, images.shape[0]):
                ret_labels.append(label)
            #for image in images:
            #    ret_images.append(image)
            #    ret_labels.append(label)
        i = 1 + i
        print(i, len(carModels))
    return np.concatenate(ret_images), np.array(ret_labels)

def class_to_vector(classId, numberOfClasses):
    list = []
    i = 0
    for i in range(numberOfClasses):
        list.append(1 if i == classId else 0)
    return list

def get_batch(arr, batch_size):
    indices = np.arange(len(arr))
    np.random.shuffle(indices)
    return arr[:,0][indices[:batch_size]], arr[:,1][indices[:batch_size]]

def readAllImagesFromPath(path, shape):
    # Make a queue of file names including all the JPEG images files in the relative
    # image directory.
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(path + "*.jpg"))

    # Read an entire image file which is required since they're JPEGs, if the images
    # are too large they could be split in advance to smaller files or use the Fixed
    # reader to split up the file.
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue, the first returned value in the tuple is the
    # filename which we are ignoring.
    _, image_file = image_reader.read(filename_queue)

    # Decode the image as a JPEG file, this will turn it into a Tensor which we can
    # then use in training.
    image = tf.image.decode_jpeg(image_file)
    images = []
    # Start a new session to show example output.
    with tf.Session() as sess:
        # Required to get the filename matching to run.
        tf.initialize_all_variables().run()

        # Coordinate the loading of image files.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # Get an image tensor and print its value.
        image_tensor = sess.run([image])
        resized_image = tf.image.resize_images(image, [shape[0], shape[1]])
        print(image_tensor)
        images.append(image_tensor)

        # Finish off the filename queue coordinator.
        coord.request_stop()
        coord.join(threads)
    return images