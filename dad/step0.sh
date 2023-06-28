python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/seed1/1shot.json \
    -t /media/data/coco17/coco/seed1/1shot_novel.json \
    -f coco_novel_class_ids

python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/seed1/3shot.json \
    -t /media/data/coco17/coco/seed1/3shot_novel.json \
    -f coco_novel_class_ids

python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/seed1/5shot.json \
    -t /media/data/coco17/coco/seed1/5shot_novel.json \
    -f coco_novel_class_ids

python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/seed1/10shot.json \
    -t /media/data/coco17/coco/seed1/10shot_novel.json \
    -f coco_novel_class_ids

python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/seed1/30shot.json \
    -t /media/data/coco17/coco/seed1/30shot_novel.json \
    -f coco_novel_class_ids

python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/annotations/instances_train2017.json \
    -t /media/data/coco17/coco/annotations/instances_train2017_novel.json \
    -f coco_novel_class_ids

python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/annotations/instances_val2017.json \
    -t /media/data/coco17/coco/annotations/instances_val2017_novel.json \
    -f coco_novel_class_ids

python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/annotations/instances_train2017.json \
    -t /media/data/coco17/coco/annotations/instances_train2017_base.json \
    -f coco_base_class_ids

python3 0_remove_novel_class_from_coco_annotation.py \
    -s /media/data/coco17/coco/annotations/instances_val2017.json \
    -t /media/data/coco17/coco/annotations/instances_val2017_base.json \
    -f coco_base_class_ids
