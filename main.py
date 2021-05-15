from dataset.cardamage import CarDamage
from evaluation.cardamageeval import Cardamageeval
import util.prepare_for_coco as util
import logging
import torch
import tempfile
import os
import json



class Evaluation():
    def __init__(self, ann_file, predict_file, iou_types, class_type, output_folder="./data/"):
        self.ann_file = ann_file
        self.predict_file = predict_file
        self.iou_types = iou_types
        self.class_type = class_type
        self.output_folder = output_folder
        self.logger = logging.getLogger("maskrcnn_benchmark.inference")
        # self._prepare()
    def _prepare(self):
        self.GT = CarDamage(anno_file)
        self.coco_results = {}
        # 通过box来计算iou
        if "bbox" in self.iou_types:
            self.logger.info("Preparing bbox results")
            # 准备box的相关结果
            self.coco_results["bbox"] = util.prepare_for_coco_detection(self.predict_file, self.GT)
        # 通过segmentation计算iou
        if "segm" in self.iou_types:
            self.logger.info("Preparing segm results")
            # 准备分割的相关结果
            self.coco_results["segm"] = util.prepare_for_coco_segmentation(self.predict_file, self.GT)

        self.results = util.COCOResults(*self.iou_types)

    def build(self):
        self._prepare()
        self.logger.info("Evaluating predictions")
        print("iou types are:", self.iou_types)

        for iou_type in self.iou_types:
            coco_result = self.coco_results[iou_type]

            # print("coco_result:", coco_result)

            with tempfile.NamedTemporaryFile() as f:
                file_path = f.name
                if self.output_folder:
                    # 临时文件路径
                    file_path = os.path.join(self.output_folder, iou_type + ".json")
                    with open(file_path, "w") as f:
                        json.dump(coco_result, f)

                DT = self.GT.loadRes(str(file_path))
                if self.class_type == 24:
                    # 进行类别的转换
                    coco_gt = self.GT.transformTo24()
                    coco_dt = DT.transformTo24()
                elif self.class_type == 52:
                    # 进行类别的转换
                    coco_gt = self.GT.transformTo52()
                    coco_dt = DT.transformTo52()
                elif self.class_type == 160:
                    # 进行类别的转换
                    coco_gt = self.GT.transformTo160()
                    coco_dt = DT.transformTo160()
                else:
                    raise NotImplementedError(
                        (
                            "Did not implement this class type:"
                            "%s" % self.class_type
                        )
                    )
                    return


                coco_eval = Cardamageeval(coco_gt, coco_dt, iou_type)
                coco_eval.evaluate()
                coco_eval.accumulate()
                coco_eval.summarize()

                self.results.update(coco_eval)
            self.logger.info(self.results)
        return self.results

    def build_simple(self, file_paths={}):
        self.logger.info("Evaluating predictions")
        print("iou types are:", self.iou_types)
        self.results = util.COCOResults(*self.iou_types)
        self.GT = CarDamage(anno_file)
        for iou_type in self.iou_types:
            DT = self.GT.loadRes(str(file_paths[iou_type]))
            if self.class_type == 24:
                # 进行类别的转换
                coco_gt = self.GT.transformTo24()
                coco_dt = DT.transformTo24()
            elif self.class_type == 52:
                # 进行类别的转换
                coco_gt = self.GT.transformTo52()
                coco_dt = DT.transformTo52()
            elif self.class_type == 160:
                # 进行类别的转换
                coco_gt = self.GT.transformTo160()
                coco_dt = DT.transformTo160()
            else:
                raise NotImplementedError(
                    (
                            "Did not implement this class type:"
                            "%s" % self.class_type
                    )
                )
                return

            coco_eval = Cardamageeval(coco_gt, coco_dt, iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            self.results.update(coco_eval)
        self.logger.info(self.results)
        return self.results


if __name__ == '__main__':
    anno_file = "./data/instances_val2017.json"
    predict_file_path = "./data/predictions.pth"
    predictions = torch.load(predict_file_path)
    iou_types = ["bbox", "segm"]
    file_paths = {"bbox": "./data/bbox.json",
                  "segm": "./data/segm.json"}

    class_type = 24
    eval = Evaluation(anno_file, predictions, iou_types, class_type)
    # result = eval.build()
    result = eval.build_simple(file_paths)
    print(result)

