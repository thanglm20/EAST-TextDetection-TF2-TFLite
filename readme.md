# EAST: An Efficient and Accurate Scene Text Detector

### Contents
1. [Installation](#installation)
2. [Download](#download)
2. [Test](#train)
3. [Train](#test)
4. [Examples](#examples)
5. [Export](#Export)

### Installation
1. Any version of tensorflow version > 1.0 should be ok.
2. Install depender package in file requirements.txt

    ```
    pip install -r requirements.txt
    export PYTHONPATH= <your_path_EAST>
    ```
### Download
1. Resnet V1 50 provided by tensorflow slim: [slim resnet v1 50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

### Train
If you want to train the model, you should provide the dataset path, in the dataset path, a separate gt text file should be provided for each image
You can see folder 'training_samples'
and run

```
python3 multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=14 --checkpoint_path=model_train/ \
--text_scale=512 --training_data_path=data/ocr --geometry=RBOX --learning_rate=0.0001 --num_readers=24 \
--pretrained_model_path=model_pre/resnet_v1_50.ckpt
```

If you have more than one gpu, you can pass gpu ids to gpu_list(like --gpu_list=0,1,2,3)

**Note: you should change the gt text file of icdar2015's filename to img_\*.txt instead of gt_img_\*.txt(or you can change the code in icdar.py), and some extra characters should be removed from the file.
See the examples in training_samples/**

### Test
run
```
python eval.py --test_data_path=/../images/ --gpu_list=0 --checkpoint_path=/../model_train/ \
--output_dir=/../
```

a text file will be then written to the output path.


### Examples
Here are some test examples on icdar2015, enjoy the beautiful text boxes!
![image_1](demo_images/img_2.jpg)
![image_2](demo_images/img_10.jpg)
![image_3](demo_images/img_14.jpg)
![image_4](demo_images/img_26.jpg)
![image_5](demo_images/img_75.jpg)

###Export 
Modify freeze_checkpoint.py to suit you.
-line 21: tf.app.flags.DEFINE_string('gpu_list', '0', '')
-line 22: tf.app.flags.DEFINE_string('checkpoint_path', './model_train/','')
-line 23: tf.app.flags.DEFINE_string('meta', './east_icdar2015_resnet_v1_50_rboxmodel.ckpt-331.meta','')
-line 24: tf.app.flags.DEFINE_string('output_dir', './output/', '')

python freeze_checkpoint.py

* Convert model pb to TFlite
tflite_convert \
--graph_def_file=/../frozen_east_text_detection.pb \
--output_file=/../model.tflite \
--output_format=TFLITE \
--input_arrays=input_images \
--input_shapes=1,96,128,3 \
--inference_type=FLOAT \
--output_arrays="feature_fusion/Conv_7/Sigmoid,feature_fusion/concat_3" \
--allow_custom_ops


### Troubleshooting
    TypeError: expected str, bytes or os.PathLike object, not PosixPath
