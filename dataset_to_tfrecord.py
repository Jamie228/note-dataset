## Convert XML files to TFRecord files
## TFRecord

#Import libraries
import os
import io
import glob
import absl
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random

from PIL import Image
from object_detection.utils import dataset_util

def create_example(xml_file):
        #Process the XML file
        tree = ET.parse(xml_file)
        root = tree.getroot()
        #Get image name, filename and width/height
        image_name = root.find('filename').text
        file_name = image_name.encode('utf8')
        size=root.find('size')
        width = int(size[0].text)
        height = int(size[1].text)
        #Init lists
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        classes = []
        classes_text = []
        truncated = []
        poses = []
        difficult_obj = []
        #Iterate through objects and append info to lists
        for member in root.findall('object'):
           classes_text.append(member[0].text.encode('utf8'))
           xmin.append(float(member[4][0].text) / width)
           ymin.append(float(member[4][1].text) / height)
           xmax.append(float(member[4][2].text) / width)
           ymax.append(float(member[4][3].text) / height)
           difficult_obj.append(0)
           
           ##Convert text label to corresponding integer.
           ##Throw ValueError if class not valid
           def class_text_to_int(row_label):
                if row_label == 'lc':
                    return 1
                if row_label == 'd':
                    return 2
                if row_label == 'e':
                    return 3
                if row_label == 'f':
                    return 4
                if row_label == 'g':
                    return 5
                if row_label == 'a':
                    return 6
                if row_label == 'b':
                    return 7
                if row_label == 'hc':
                    return 8
                else:
                    raise ValueError("Class name invalid")
            
           #Append class int
           classes.append(class_text_to_int(member[0].text))
           #Append misc info to lists
           truncated.append(0)
           poses.append('Unspecified'.encode('utf8'))

        #Read corresponding image
        full_path = os.path.join('../Documents/GitHub/ump-poc/TensorFlow/workspace/training_demo/images/train', '{}'.format(image_name))  #provide the path of images directory
        #Read and encode JPG image
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        #Check image is JPG
        if image.format != 'JPEG':
           raise ValueError('Image format not JPEG')
        #Create SHA256 hash from image data
        key = hashlib.sha256(encoded_jpg).hexdigest()
		
        #create TFRecord Example
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(file_name),
            'image/source_id': dataset_util.bytes_feature(file_name),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
            'image/object/truncated': dataset_util.int64_list_feature(truncated),
            'image/object/view': dataset_util.bytes_list_feature(poses),
        }))	
        return example	
		
def main(_):
    #Define output filepath + file names
    writer_train = tf.python_io.TFRecordWriter('train-nn.record')     
    writer_test = tf.python_io.TFRecordWriter('test-nn.record')
    #Define path to XML annotations
    filename_list=tf.train.match_filenames_once("./note_name_annotations/*.xml")
    #Define TF vars
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess=tf.Session()
    sess.run(init)
    #Create and shuffle list
    list=sess.run(filename_list)
    random.shuffle(list)
    #Instantiate counting vars
    i=1 
    tst=0 
    trn=0
    #Iterate through list of file names
    for xml_file in list:
      #Create example  
      example = create_example(xml_file)
      #Use ~10% of images for testing/evaluation
      if (i%10)==0:
         writer_test.write(example.SerializeToString())
         tst=tst+1
      else:
         writer_train.write(example.SerializeToString())
         trn=trn+1
      i=i+1
      #Output most recent XML file
      print(xml_file)
    #Close writers + output info
    writer_test.close()
    writer_train.close()
    print('Successfully converted dataset to TFRecord.')
    print('training dataset: # ')
    print(trn)
    print('test dataset: # ')
    print(tst)	
	
if __name__ == '__main__':
    tf.app.run()
