import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import os
import collections
import datetime
from types import SimpleNamespace
import config


# --- Utility classes for logging ---

class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    @property
    def median(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).max().item()

    @property
    def value(self):
        return torch.tensor(list(self.deque), dtype=torch.float32)[-1].item()

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="  "):
        self.meters = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time:.2f} s over {len(iterable)} iterations)')


# --- YOLOv5 loss helpers ---

def _get_yolo_loss_fn(model):
    from ultralytics.utils.loss import v8DetectionLoss

    hyp = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)

    # Patch on the wrapper and the inner model to cover all ultralytics versions
    model.args = hyp
    model.hyp  = hyp
    inner = getattr(model, 'model', model)
    inner.args = hyp
    inner.hyp  = hyp

    try:
        loss_fn = v8DetectionLoss(model)
        print("[DEBUG] v8DetectionLoss built successfully")
        return loss_fn
    except Exception as e:
        print(f"[WARNING] v8DetectionLoss failed to build: {e}")
        return None


def _register_detect_hook(model):
    feats = []

    def hook_fn(module, inputs, output):
        feats.clear()
        # inputs[0] is the list of per-stride tensors fed into the Detect head
        raw = inputs[0]
        if isinstance(raw, (list, tuple)):
            feats.extend(list(raw))
        else:
            feats.append(raw)

    detect_head = None
    for m in model.modules():
        if type(m).__name__ == 'Detect':
            detect_head = m
            break

    if detect_head is None:
        raise RuntimeError("Could not find Detect head to register hook")

    handle = detect_head.register_forward_hook(hook_fn)
    return handle, feats


def _compute_yolo_loss(loss_fn, feats, targets_on_device):
    device = targets_on_device[0]['boxes'].device

    batch_idx_list, cls_list, bboxes_list = [], [], []
    for i, t in enumerate(targets_on_device):
        boxes  = t['boxes']
        labels = t['labels']
        if boxes.numel() == 0:
            continue
        n = boxes.shape[0]
        # Convert pixel x1y1x2y2 → normalised cxcywh expected by ultralytics loss
        x_center = (boxes[:, 0] + boxes[:, 2]) / 2 / 640.0
        y_center = (boxes[:, 1] + boxes[:, 3]) / 2 / 640.0
        w        = (boxes[:, 2] - boxes[:, 0])      / 640.0
        h        = (boxes[:, 3] - boxes[:, 1])      / 640.0
        bboxes_list.append(torch.stack([x_center, y_center, w, h], dim=1))
        # labels are 1-indexed (background=0 not used); loss expects 0-indexed → subtract 1
        # unsqueeze to (N, 1) as required by v8DetectionLoss
        cls_list.append((labels - 1).float().unsqueeze(1))
        batch_idx_list.append(torch.full((n,), i, dtype=torch.float32, device=device))

    if bboxes_list:
        batch = {
            'batch_idx': torch.cat(batch_idx_list).to(device),
            'cls':       torch.cat(cls_list).to(device),
            'bboxes':    torch.cat(bboxes_list).to(device),
        }
    else:
        batch = {
            'batch_idx': torch.zeros((0,),   dtype=torch.float32, device=device),
            'cls':       torch.zeros((0, 1), dtype=torch.float32, device=device),
            'bboxes':    torch.zeros((0, 4), dtype=torch.float32, device=device),
        }

    # --- Try real ultralytics loss ---
    if loss_fn is not None:
        try:
            loss, loss_items = loss_fn(feats, batch)
            if loss.dim() > 0:
                loss = loss.sum()
            return loss, {
                'loss_box': loss_items[0].item(),
                'loss_cls': loss_items[1].item(),
                'loss_dfl': loss_items[2].item(),
            }
        except Exception as e:
            print(f"[WARNING] v8DetectionLoss forward failed: {e}")

    # --- Fallback: MSE against zeros (should never be reached after the hook fix) ---
    print("[WARNING] Using MSE fallback loss — gradients will be meaningless")
    if isinstance(feats, (list, tuple)):
        loss = sum(torch.nn.functional.mse_loss(o, torch.zeros_like(o)) for o in feats)
    else:
        loss = torch.nn.functional.mse_loss(feats, torch.zeros_like(feats))
    if loss.dim() > 0:
        loss = loss.mean()
    return loss, {'loss_mse_fallback': loss.item()}


# --- Training functions ---

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, model_type, yolo_loss_fn=None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'

    # Register Detect hook once for the entire epoch
    hook_handle, feats = None, []
    if model_type == config.YOLOV5N_MODEL_NAME:
        hook_handle, feats = _register_detect_hook(model)

    if epoch == 0 and hasattr(optimizer, 'param_groups') and len(data_loader) > 1:
        warmup_factor = 1.0 / 1000
        warmup_iters  = min(1000, len(data_loader) - 1)
        lr_scheduler  = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )
    else:
        lr_scheduler = None

    try:
        for images, targets in metric_logger.log_every(data_loader, print_freq, header):
            if model_type == config.FASTER_RCNN_MODEL_NAME:
                images_on_device = list(image.to(device) for image in images)
            elif model_type == config.YOLOV5N_MODEL_NAME:
                images_on_device = (
                    torch.stack(images, 0) if isinstance(images, (list, tuple)) else images
                ).to(device)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if model_type == config.FASTER_RCNN_MODEL_NAME:
                loss_dict = model(images_on_device, targets_on_device)
                losses = sum(loss for loss in loss_dict.values())

            elif model_type == config.YOLOV5N_MODEL_NAME:
                # Forward pass — the hook captures the raw feature maps into `feats`
                model(images_on_device)
                losses, loss_dict = _compute_yolo_loss(yolo_loss_fn, feats, targets_on_device)

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            metric_logger.update(loss=losses.item(), **loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    finally:
        # Always remove the hook even if an exception occurs mid-epoch
        if hook_handle is not None:
            hook_handle.remove()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_model(model, model_type, train_dataloader, val_dataloader, optimizer, lr_scheduler, num_epochs, device, checkpoint_dir):
    start_total_time = time.time()
    best_val_loss = float('inf')

    # Build the loss function once and reuse across all epochs
    yolo_loss_fn = None
    if model_type == config.YOLOV5N_MODEL_NAME:
        yolo_loss_fn = _get_yolo_loss_fn(model)

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        epoch_start_time = time.time()

        # --- Training ---
        train_metrics = train_one_epoch(
            model, optimizer, train_dataloader, device,
            epoch + 1, 50, model_type, yolo_loss_fn=yolo_loss_fn
        )
        print(f"Epoch {epoch+1} training finished. Training Metrics: {train_metrics}")

        # --- Validation ---
        val_metric_logger = MetricLogger(delimiter="  ")
        header = f'Val: [{epoch}]'
        model.train()  # train mode so the Detect head produces feature maps

        # Register hook for the validation loop
        val_hook_handle, val_feats = None, []
        if model_type == config.YOLOV5N_MODEL_NAME:
            val_hook_handle, val_feats = _register_detect_hook(model)

        try:
            with torch.no_grad():
                for images, targets in val_metric_logger.log_every(val_dataloader, 50, header):
                    if model_type == config.FASTER_RCNN_MODEL_NAME:
                        images_on_device = list(image.to(device) for image in images)
                    elif model_type == config.YOLOV5N_MODEL_NAME:
                        images_on_device = (
                            torch.stack(images, 0) if isinstance(images, (list, tuple)) else images
                        ).to(device)
                    else:
                        raise ValueError(f"Unknown model_type: {model_type}")

                    targets_on_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    if model_type == config.FASTER_RCNN_MODEL_NAME:
                        loss_dict = model(images_on_device, targets_on_device)
                        losses = sum(loss for loss in loss_dict.values())

                    elif model_type == config.YOLOV5N_MODEL_NAME:
                        model(images_on_device)
                        losses, loss_dict = _compute_yolo_loss(yolo_loss_fn, val_feats, targets_on_device)

                    else:
                        raise ValueError(f"Unknown model_type: {model_type}")

                    val_metric_logger.update(loss=losses.item(), **loss_dict)

        finally:
            if val_hook_handle is not None:
                val_hook_handle.remove()

        model.eval()

        avg_val_loss = val_metric_logger.meters['loss'].global_avg
        print(f"Epoch {epoch+1} validation finished. Avg Val Loss: {avg_val_loss:.4f}")

        if lr_scheduler:
            lr_scheduler.step()
            print(f"LR after scheduler: {optimizer.param_groups[0]['lr']:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path} (Val Loss: {best_val_loss:.4f})")
        else:
            print(f"Val Loss ({avg_val_loss:.4f}) did not improve over best ({best_val_loss:.4f}).")

        print(f"Epoch {epoch+1} took {time.time() - epoch_start_time:.2f}s.")

    total_training_time = time.time() - start_total_time
    print(f"\nTotal training time: {total_training_time:.2f} seconds.")
    return total_training_time