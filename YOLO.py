import torch

def get_yolov5_model(num_classes):
    """
    Load YOLOv5n and replace the Detect head so it outputs
    (4 + num_classes) channels instead of the default 84 (4+80).
    num_classes should be the number of REAL classes (no background).
    """
    try:
        from ultralytics import YOLO
        import torch.nn as nn

        # num_classes coming from config includes __background__, strip it
        real_classes = num_classes - 1

        yolo = YOLO('yolov5nu.pt')
        model = yolo.model

        # Find the Detect head
        detect_head = None
        for m in model.modules():
            if type(m).__name__ == 'Detect':
                detect_head = m
                break

        if detect_head is None:
            raise RuntimeError("Could not find Detect head in YOLOv5 model")

        old_nc = detect_head.nc
        detect_head.nc = real_classes

    
        new_cv3 = nn.ModuleList()
        for cv3_conv in detect_head.cv3:
            # Each is a Sequential or Conv; get in_channels from existing layer
            if isinstance(cv3_conv, nn.Sequential):
                in_ch = cv3_conv[-1].in_channels
                out_ch = real_classes
                # Preserve all but last layer, replace last
                layers = list(cv3_conv.children())
                last = layers[-1]
                new_last = nn.Conv2d(
                    last.in_channels, out_ch,
                    kernel_size=last.kernel_size,
                    stride=last.stride,
                    padding=last.padding,
                    bias=True
                )
                nn.init.normal_(new_last.weight, 0, 0.01)
                nn.init.zeros_(new_last.bias)
                new_cv3.append(nn.Sequential(*layers[:-1], new_last))
            else:
                # Direct Conv2d
                new_conv = nn.Conv2d(
                    cv3_conv.in_channels, real_classes,
                    kernel_size=cv3_conv.kernel_size,
                    stride=cv3_conv.stride,
                    padding=cv3_conv.padding,
                    bias=True
                )
                nn.init.normal_(new_conv.weight, 0, 0.01)
                nn.init.zeros_(new_conv.bias)
                new_cv3.append(new_conv)

        detect_head.cv3 = new_cv3

        print(f"Rebuilt Detect head: {old_nc} → {real_classes} classes")
        print(f"New output channels per anchor: {4 + real_classes}")

        for p in model.parameters():
            p.requires_grad = True

        return model

    except Exception as e:
        raise RuntimeError(f"Could not load YOLOv5n model: {e}")