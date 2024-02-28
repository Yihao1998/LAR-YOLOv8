def bbox_RIOU(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, RIoU=False, eps=1e-7):
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if GIoU or DIoU or CIoU or RIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  

        if CIoU or DIoU or RIoU:  
            c2 = cw ** 2 + ch ** 2 + eps  
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4 
            if DIoU:
                return iou - rho2 / c2  
            elif CIoU:  
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  
            elif RIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + torch.log(1+rho_w2) / torch.log(1+cw2) + torch.log(1+rho_h2) / torch.log(1+ch2))
        else:  
            c_area = cw * ch + eps  
            return iou - (c_area - union) / c_area  
    else:
        return iou  