import logging
from collections import OrderedDict
from tqdm import tqdm
from .maskutil import Masker


# 准备读取detection的相关结果
def prepare_for_coco_detection(predictions, dataset):
    '''
    Args:
        predictions: 为BoxList格式
        dataset: 为cardamagedataset对象

    Returns:

    '''
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    # 将prediction转化为cococardamage的格式
    for image_id, prediction in enumerate(predictions):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        # 获取box socre 以及 label
        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        # 获取零件类别
        component_scores = prediction.get_field("component_scores").tolist()
        components = prediction.get_field("components").tolist()

        # 预测值和类别id之间的映射关系
        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        mapped_components = [dataset.contiguous_component_id_to_json_id[i] for i in components]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "component_id": mapped_components[k],
                    "bbox": box,
                    "score": scores[k] * component_scores[k],
                    "component_scores": component_scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

# 准备预测得到的segmentation格式数据
def prepare_for_coco_segmentation(predictions, dataset):
    import pycocotools.mask as mask_util
    import numpy as np

    masker = Masker(threshold=0.5, padding=1)
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    # 通过图片id进行遍历
    for image_id, prediction in tqdm(enumerate(predictions)):
        original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        # 获得原始的图片信息（宽高）
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        masks = prediction.get_field("mask")
        # t = time.time()
        # Masker is necessary only if masks haven't been already resized.
        if list(masks.shape[-2:]) != [image_height, image_width]:
            masks = masker(masks.expand(1, -1, -1, -1, -1), prediction)
            masks = masks[0]
        # logger.info('Time mask: {}'.format(time.time() - t))
        # prediction = prediction.convert('xywh')

        # boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        # 获取零件类别
        component_scores = prediction.get_field("component_scores").tolist()
        components = prediction.get_field("components").tolist()

        # rles = prediction.get_field('mask')

        rles = [
            mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
            for mask in masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        mapped_labels = [dataset.contiguous_category_id_to_json_id[i] for i in labels]
        # 预测值和类别id之间的映射关系
        mapped_components = [dataset.contiguous_component_id_to_json_id[i] for i in components]

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": mapped_labels[k],
                    "component_id": mapped_components[k],
                    "segmentation": rle,
                    "score": scores[k] * component_scores[k],
                    "component_score": component_scores[k],
                }
                for k, rle in enumerate(rles)
            ]
        )
    return coco_results

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox", "segm")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval
        from evaluation.cardamageeval import Cardamageeval

        assert isinstance(coco_eval, Cardamageeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]

        print("metrics:", metrics)
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        results = '\n'
        for task, metrics in self.results.items():
            results += 'Task: {}\n'.format(task)
            metric_names = metrics.keys()
            metric_vals = ['{:.4f}'.format(v) for v in metrics.values()]
            results += (', '.join(metric_names) + '\n')
            results += (', '.join(metric_vals) + '\n')
        return results


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("maskrcnn_benchmark.inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)


if __name__ == '__main__':
    compid = [6, 5, 4, 3, 2, 1]
    json_component_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(compid)
        }
    contiguous_component_id_to_json_id = {
        v: k for k, v in json_component_id_to_contiguous_id.items()
    }

    print("json_component_id_to_contiguous_id:", json_component_id_to_contiguous_id)
    print("contiguous_component_id_to_json_id:", contiguous_component_id_to_json_id)