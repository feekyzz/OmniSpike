from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch

data = dict(
    train=dict(
        yolo_data=["/home/qh1/Qwen/Yw/ultralytics-Yw/datasets/Objects365.yaml"],
        grounding_data=[
            dict(
                img_path="/home/qh1/Qwen/Yw/ultralytics-Yw/datasets/flickr30k/images",
                json_file="/home/qh1/Qwen/Yw/ultralytics-Yw/datasets/flickr30k/final_flickr_separateGT_train.json",
            ),
            dict(
                img_path="/home/qh1/Qwen/Yw/ultralytics-Yw/datasets/gqa/images",
                json_file="/home/qh1/Qwen/Yw/ultralytics-Yw/datasets/gqa/final_mixed_train_no_coco.json",
            ),
        ],
    ),
    val=dict(yolo_data=["/home/qh1/Qwen/Yw/ultralytics-Yw/datasets/coco.yaml"]),
)


model = YOLOE("ultralytics\cfg\models\26\yoloe-26s.yaml")
model.train(
    data=data,  # or data="yoloe_data.yaml" if using YAML file
    name="yoloe-26s",
    batch=8,
    epochs=30,
    close_mosaic=2,
    optimizer="AdamW",
    lr0=2e-3,
    warmup_bias_lr=0.0,
    weight_decay=0.025,
    momentum=0.9,
    workers=0,
    trainer=YOLOETrainerFromScratch,
    # device="4,5,6,7",
)       

# from ultralytics import YOLOE
# from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch


# model = YOLOE("/home/qh1/Qwen/Yw/ultralytics/runs/detect/Base-YE-11m-0510/weights/last.pt")
# model.train(
#     resume = True,
#     trainer=YOLOETrainerFromScratch,
#     # device="4,5,6,7",
# )       