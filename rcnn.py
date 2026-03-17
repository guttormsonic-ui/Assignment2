import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#RCNN from torchvision
def get_faster_rcnn_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor