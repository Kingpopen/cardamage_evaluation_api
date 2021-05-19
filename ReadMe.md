# cardamage_evaluation_api
这是关于车损多分支数据集测评包的介绍文档。（基于COCO评测的pycocotools进行微小调整）

- [x] 可以将多分支的数据转化为单分支的24分类，52分类，160分类
- [x] 计算不同iou，area，mdet下的指标AP、AR指标
- [x] 对结果进行了加权平均
- [ ] 对结果进行csv文件形式的保存
- [ ] 画出PR曲线图
- [ ] 对结果进行F1指标函数的计算~（不太好弄）


## 前期工作
由于该评测的工具是基于pycocotools微小调整的，所以在使用之前需要下载安装好pycocotools包:sweat:

### 0. requestment.txt
* pycocotools
* pytorch == 1.2
* python >= 3.6
* 具体报什么错的话就添加什么包~

### 1. 评测数据的准备
1. *instances_val2017.json(或instances_test2017.json)*
2. *predictions.pth (通过[maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)的inference过程得到)*

本包的评测数据需要通过maskrcnn-benchmark进行测试得到.
对maskrcnn-benchmark中的maskrcnn-bechmark/engine/inference.py文件中的return函数返回值设置为空.

``` python
.....
.....
.....
#我们需要的只是predictions.pth文件 具体保存路径，根据自己设定的output_folder文件路径去找
# output_folder在 your_project_path/maskrcnn_benckmark/config/defaults.py中的_C.OUTPUT_DIR 设置
    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    #return evaluate(dataset=dataset,
    #                predictions=predictions,
    #                output_folder=output_folder,
    #               **extra_args)
    return 
```

修改完成之后，使用maskrcnn_benchmark进行inference
```python
python ./tools/test_net.py --config-file  your_config_file_path/config_file.yaml
```

### 2. instances_val2017.json文件介绍
这是COCO官方用来训练或者测试的标注文件格式，只不过我们的模型是多分支的，所以在标注信息中添加了两种类别的信息，
具体见下面介绍(**如果你的格式和下面不一致，需要改成和下面一致的格式:smile:**)：
```python {.line-numbers}
{
        “images”:[
        {
            "height":...
            "width":...
            "id":...
            "filename_name":...(纯图片名 不带路径)
            }
        ...
        ...
        ],

        “categories”:[ (存放损伤标签)
        {
            "supercategory": "scratch",
            "id": 1,
            "name": "scratch"
            }
        ...
        ...
        ],

        "components": [(存放零件标签)
        {
            "supercategory": "bumper",
            "id": 1,
            "name": "bumper"
        }
        ...
        ...
        ],

        "annotations":[
        {
            "iscrowd": 0,
            "image_id": 1,
            "bbox": [...],
            "segmentation": [
                [[...]]
            ],
            "category_id": 1,
            "component_id": 1,
            "id": 1,
            "area": 18610.059492
        },
        ...
        ...
        ],  
    }
```


### 3. predictions.pth文件介绍
predictions.pth是一个包含了inference结果的BoxList对象列表（一张图片对应一个BoxList对象）,
BoxList数据类型见下面官方介绍：
#### BoxList
The `BoxList` class holds a set of bounding boxes (represented as a `Nx4` tensor) for
a specific image, as well as the size of the image as a `(width, height)` tuple.
It also contains a set of methods that allow to perform geometric
transformations to the bounding boxes (such as cropping, scaling and flipping).
The class accepts bounding boxes from two different input formats:
- `xyxy`, where each box is encoded as a `x1`, `y1`, `x2` and `y2` coordinates, and
- `xywh`, where each box is encoded as `x1`, `y1`, `w` and `h`.

Additionally, each `BoxList` instance can also hold arbitrary additional information
for each bounding box, such as labels, visibility, probability scores etc.

简单的来说就是一个BoxList对象中包含有一张图片所有的instances坐标框信息，如果需要添加label，mask等信息，可以通过add_field()方法添加进去。(BoxList还包含有一些对坐标框进行旋转、裁剪等方法)

Here is an example on how to create a `BoxList` from a list of coordinates:
```python {.line-numbers}
from maskrcnn_benchmark.structures.bounding_box import BoxList, FLIP_LEFT_RIGHT
import torch

width = 100
height = 200
boxes = [
  [0, 10, 50, 50],
  [50, 20, 90, 60],
  [10, 10, 50, 50]
]
# create a BoxList with 3 boxes
bbox = BoxList(boxes, image_size=(width, height), mode='xyxy')

# perform some box transformations, has similar API as PIL.Image
bbox_scaled = bbox.resize((width * 2, height * 3))
bbox_flipped = bbox.transpose(FLIP_LEFT_RIGHT)

# add labels for each bbox
labels = torch.tensor([0, 10, 1])
bbox.add_field('labels', labels)

# bbox also support a few operations, like indexing
# here, selects boxes 0 and 2
bbox_subset = bbox[[0, 2]]
```
本cardamage_evaluation_api已经指定了BoxList中固定的fields，
下面主要介绍predictions.pth中的BoxList对象必须包含的fields:
```python
# 可通过fields()方法查看所有的fields
boxlist.fields()
```
**labels**: 保存该图片所有预测的损伤类型  &ensp; eg. *[1, 2, 4, 1 ...]*   
**scores**: 保存该图片所有预测的损伤类型对应得分 &ensp; eg. *[0.8, 0.2, 0.3, 0.6...]*   
**componets**:保存该图片所有预测的零件类型 &ensp; eg. *[3, 2, 5, 1 ...]*   
**component_scores**:保存该图片所有预测的零件类型对应得分 &ensp; eg. *[0.8, 0.4, 0.5, 0.7...]*   
**mask**: 保存每一个instance对应的mask图 shape is (num_instances, 1, mask_size, mask_size)  
**（如果你的predictions.pth中的BoxList对象不包含以上的fields，或者field名称不太一样，那么可能需要进行一定的修改:smile:）**


## 评测
在cardamage_evaluation_api中运行main.py函数：
```python
python main.py
```
```python
if __name__ == '__main__':
    # 存放instances_val2017.json 或者 instances_test2017.json的路径
    anno_file = "./data/instances_val2017.json"
    predict_file_path = "./data/predictions.pth"
    # 读取predictions.pth文件，因为maskrcnn_benchmark中是使用torch进行保存的，所以就用torch进行读取
    predictions = torch.load(predict_file_path)
    # 计算iou的方式使用bbox计算和使用segmentation方式计算
    iou_types = ["bbox", "segm"]

    # 下面这个是运行过一次之后会生成predictions.pth的中间文件
    file_paths = {"bbox": "./data/bbox.json",
                  "segm": "./data/segm.json"}

    # 数据集的类别 多分支的24类别 or 52类别 or 160分类
    # 评测函数内部会自动转化为相应单分支的类型进行指标的计算
    class_type = 24

    eval = Evaluation(anno_file, predictions, iou_types, class_type)
    # 如果是第一次运行使用eval.build()方法
    # 第一次运行之后会得到bbox.json和segm.json的中间文件 
    # 第二次或者后面重复计算 直接使用第一次计算好的中间文件，使用eval.build_simple(file_paths)运行
    result = eval.build()
    #result = eval.build_simple(file_paths)
    print(result)
```
