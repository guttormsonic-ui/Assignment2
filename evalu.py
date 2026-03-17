import torch
import time
from torchmetrics.detection import MeanAveragePrecision
import config


def _parse_yolov5_output(raw_outputs, images_on_device, device, num_classes=80):
    if isinstance(raw_outputs, tuple):
        pred_tensor = raw_outputs[0]   # (batch, 84, 8400)
    else:
        pred_tensor = raw_outputs

    # (batch, 84, 8400) → (batch, 8400, 84)
    pred_tensor = pred_tensor.permute(0, 2, 1)
    expected_channels = 4 + (num_classes - 1)
    actual_channels = pred_tensor.shape[2]
    if actual_channels != expected_channels:
        raise ValueError(f"[YOLO shape mismatch] got {actual_channels} channels, "
        f"expected {expected_channels} (4 box + {num_classes - 1} classes). "
        f"num_classes={num_classes}"
    )
    

    img_h, img_w = images_on_device.shape[2], images_on_device.shape[3]
    batch_size   = images_on_device.shape[0]
    outputs      = []

    for b in range(batch_size):
        preds = pred_tensor[b]
        boxes_raw   = preds[:, :4] 
        class_probs = preds[:, 4:]     
        scores, class_ids = class_probs.max(dim=1)
        mask = scores > 0.001
        if mask.sum() == 0:
            outputs.append({
                'boxes':  torch.empty((0, 4), device=device),
                'scores': torch.empty((0,),   device=device),
                'labels': torch.empty((0,),   dtype=torch.int64, device=device)
            })
            continue
        raw_boxes = boxes_raw[mask]
        x1 = raw_boxes[:, 0] - raw_boxes[:, 2] / 2
        y1 = raw_boxes[:, 1] - raw_boxes[:, 3] / 2
        x2 = raw_boxes[:, 0] + raw_boxes[:, 2] / 2
        y2 = raw_boxes[:, 1] + raw_boxes[:, 3] / 2

        x1 = x1.clamp(0, img_w)
        y1 = y1.clamp(0, img_h)
        x2 = x2.clamp(0, img_w)
        y2 = y2.clamp(0, img_h)

        pixel_boxes = torch.stack([x1, y1, x2, y2], dim=1)

        labels = (class_ids[mask] + 1).to(torch.int64)

        outputs.append({
            'boxes':  pixel_boxes,
            'scores': scores[mask],
            'labels': labels
        })

    return outputs


def evaluate_model(model, test_dataloader, device, model_type, num_classes):
    model.eval()

    metric = MeanAveragePrecision(
        box_format='xyxy',
        iou_type='bbox',
        class_metrics=True,
        max_detection_thresholds=[1, 10, 100]
    )

    total_inference_time = 0
    num_images           = 0

    total_pred_boxes   = 0
    total_target_boxes = 0
    empty_pred_batches = 0
    pred_label_set     = set()
    target_label_set   = set()
    score_min, score_max = float('inf'), float('-inf')


    with torch.no_grad():
        for i, (images, targets) in enumerate(test_dataloader):

            # ── Prepare images ────────────────────────────────────────────────
            if model_type == config.FASTER_RCNN_MODEL_NAME:
                images_on_device = list(img.to(device) for img in images)
            elif model_type == config.YOLOV5N_MODEL_NAME:
                if isinstance(images, (list, tuple)):
                    images_on_device = torch.stack(images, 0).to(device)
                elif isinstance(images, torch.Tensor):
                    images_on_device = images.to(device)
                else:
                    raise TypeError(f"Unexpected image type for YOLOv5: {type(images)}")
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            if device == 'cuda':
                torch.cuda.synchronize()
            model_time_start = time.perf_counter()

            # ── Run inference ─────────────────────────────────────────────────
            if model_type == config.FASTER_RCNN_MODEL_NAME:
                outputs = model(images_on_device)

            elif model_type == config.YOLOV5N_MODEL_NAME:
                raw_outputs = model(images_on_device)

                if i == 0:
                    if isinstance(raw_outputs, tuple):
                        print(f"\n[DIAG] raw_outputs TUPLE length={len(raw_outputs)}")
                        for idx, item in enumerate(raw_outputs):
                            if isinstance(item, torch.Tensor):
                                print(f"  [{idx}] shape={item.shape}  "
                                      f"min={item.min():.4f}  max={item.max():.4f}")
                            else:
                                print(f"  [{idx}] type={type(item)}")
                    elif isinstance(raw_outputs, torch.Tensor):
                        print(f"\n[DIAG] raw_outputs TENSOR shape={raw_outputs.shape}")

                outputs = _parse_yolov5_output(raw_outputs, images_on_device, device, num_classes=num_classes)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            if device == 'cuda':
                torch.cuda.synchronize()
            model_time_end = time.perf_counter()
            total_inference_time += (model_time_end - model_time_start)

            if model_type == config.FASTER_RCNN_MODEL_NAME:
                num_images += len(images_on_device)
            else:
                num_images += images_on_device.shape[0]

            # ── Process ground truth targets ──────────────────────────────────
            targets_processed = []
            for t in targets:
                if t['boxes'].numel() == 0:
                    targets_processed.append({
                        'boxes':  torch.empty((0, 4), device=device),
                        'labels': torch.empty((0,),   dtype=torch.int64, device=device)
                    })
                else:
                    targets_processed.append({
                        'boxes':  t['boxes'].to(device),
                        'labels': t['labels'].to(device)
                    })

            # ── Diagnostic accumulation ───────────────────────────────────────
            batch_pred_boxes   = sum(o['boxes'].shape[0] for o in outputs)
            batch_target_boxes = sum(t['boxes'].shape[0] for t in targets_processed)
            total_pred_boxes   += batch_pred_boxes
            total_target_boxes += batch_target_boxes

            if batch_pred_boxes == 0:
                empty_pred_batches += 1

            for o in outputs:
                if o['labels'].numel() > 0:
                    pred_label_set.update(o['labels'].tolist())
                if o['scores'].numel() > 0:
                    score_min = min(score_min, o['scores'].min().item())
                    score_max = max(score_max, o['scores'].max().item())

            for t in targets_processed:
                if t['labels'].numel() > 0:
                    target_label_set.update(t['labels'].tolist())

            # Print first 3 batches in detail
            if i < 3:
                print(f"\n[DIAG] Batch {i}:")
                for b in range(len(outputs)):
                    print(f"  Image {b}:")
                    print(f"    pred  boxes : {outputs[b]['boxes'].shape[0]} boxes  "
                          f"labels={outputs[b]['labels'].tolist()[:5]}  "
                          f"scores={[round(s, 3) for s in outputs[b]['scores'].tolist()[:5]]}")
                    print(f"    target boxes: {targets_processed[b]['boxes'].shape[0]} boxes  "
                          f"labels={targets_processed[b]['labels'].tolist()}")
                    if outputs[b]['boxes'].shape[0] > 0 and targets_processed[b]['boxes'].shape[0] > 0:
                        print(f"    pred box sample  : {[round(v, 1) for v in outputs[b]['boxes'][0].tolist()]}")
                        print(f"    target box sample: {[round(v, 1) for v in targets_processed[b]['boxes'][0].tolist()]}")
            # ─────────────────────────────────────────────────────────────────

            metric.update(preds=outputs, target=targets_processed)

    # ── Summary diagnostics ───────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"[DIAG] EVALUATION SUMMARY")
    print(f"  Total images evaluated : {num_images}")
    print(f"  Total predicted boxes  : {total_pred_boxes}")
    print(f"  Total target boxes     : {total_target_boxes}")
    print(f"  Batches with 0 preds   : {empty_pred_batches}")
    print(f"  Unique pred labels     : {sorted(pred_label_set)}")
    print(f"  Unique target labels   : {sorted(target_label_set)}")
    if score_min != float('inf'):
        print(f"  Score range            : {score_min:.4f} -> {score_max:.4f}")
    else:
        print(f"  Score range            : no predictions made")
    print(f"{'='*50}\n")
    # ─────────────────────────────────────────────────────────────────────────

    computed_metrics = metric.compute()

    mAP_0_5  = computed_metrics.get('map_50',  torch.tensor(float('nan'))).item() * 100
    mAP_coco = computed_metrics.get('map',     torch.tensor(float('nan'))).item() * 100
    recall   = computed_metrics.get('mar_100', torch.tensor(float('nan'))).item() * 100

    avg_speed = total_inference_time / num_images if num_images > 0 else 0.0

    print(f"Evaluation Results for {model_type}:")
    print(f"  mAP@0.5:             {mAP_0_5:.4f}%")
    print(f"  COCO mAP:            {mAP_coco:.4f}%")
    print(f"  Recall (MAR@100):    {recall:.4f}%")
    print(f"  Avg Inference Speed: {avg_speed:.4f} seconds/image")

    return {
        'mAP@0.5':                           mAP_0_5,
        'precision':                         mAP_coco,
        'recall':                            recall,
        'total_inference_time_sec':          total_inference_time,
        'num_images_evaluated':              num_images,
        'avg_inference_speed_per_image_sec': avg_speed
    }