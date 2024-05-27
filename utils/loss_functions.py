import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module) :
    def __init__(self, gamma=0, alpha=None, size_average=True) :
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)) :
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list) :
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target) :
        if input.dim() > 2 :
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target[:, 1 :].contiguous()
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target.to(torch.int64))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None :
            if self.alpha.type() != input.data.type() :
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1).to(torch.int64))
            logpt = logpt * Variable(at)

        loss = -(1 - pt) ** self.gamma * logpt
        if self.size_average :
            return loss.mean()
        else :
            return loss.sum()


def dice_loss(prediction, target) :
    smooth = 1.0

    prediction = torch.softmax(prediction, dim=1)[:, 1:].contiguous()
    target = target[:, 1:].contiguous()

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))


def calc_loss(prediction, target, ce_weight=0.5) :

    focal_loss = FocalLoss(gamma=2, alpha=torch.FloatTensor([1., 1.]))
    ce = focal_loss(prediction, target)

    dice = dice_loss(prediction, target)

    loss = ce * ce_weight + dice * (1 - ce_weight)

    return loss


def dice_score(prediction, target) :
    prediction = torch.sigmoid(prediction)
    smooth = 1.0
    i_flat = prediction.view(-1)
    t_flat = target.view(-1)
    intersection = (i_flat * t_flat).sum()
    return (2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth)


def prediction_map_distillation(y, teacher_scores, T=4) :
    
    channel_decomp = nn.Conv2d(y.shape[1], teacher_scores.shape[1], kernel_size=1).cuda()
    y = channel_decomp(y)
    if y.shape[2] != teacher_scores.shape[2]:
            y = F.interpolate(y, teacher_scores.size()[-2:], mode='bilinear')
    # channel_decomp = nn.Conv2d(teacher_scores.shape[1], y.shape[1], kernel_size=1).cuda()
    # teacher_scores = channel_decomp(teacher_scores)
    # if teacher_scores.shape[2] != y.shape[2]:
    #         teacher_scores = F.interpolate(teacher_scores, y.size()[-2:], mode='bilinear')
    # print(y.shape, teacher_scores.shape)

    # if y.shape[2] != teacher_scores.shape[2]:
    #     y = F.interpolate(y, teacher_scores.size()[-2:], mode='bilinear')

    p = F.log_softmax(y / T, dim=1)
    q = F.softmax(teacher_scores / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)
    return l_kl


def cat_prediction_map_distillation(y, teacher_scores, T=4):
    
    # Find the spatial dimensions of the smallest student feature map
    min_height, min_width = min([s.shape[2:] for s in y])

    # Interpolate the larger student feature maps to match the smallest spatial dimensions
    for i in range(len(y)):
        if y[i].shape[2:] != (min_height, min_width):
            y[i] = F.interpolate(y[i], size=(min_height, min_width), mode='bilinear', align_corners=False)

    # Concatenate all tensors in y along the channel dimension (dim=1)
    concatenated_y = torch.cat(y, dim=1)

    # Ensure that both input and weight are on the same device (CUDA in this case)
    concatenated_y = concatenated_y.cuda()

    channel_decomp = nn.Conv2d(concatenated_y.shape[1], teacher_scores.shape[1], kernel_size=1).cuda()
    concatenated_y = channel_decomp(concatenated_y)

    if concatenated_y.shape[2:] != teacher_scores.shape[2:]:
        concatenated_y = F.interpolate(concatenated_y, teacher_scores.shape[2:], mode='bilinear', align_corners=False)

    # Calculate KL divergence loss
    p = F.log_softmax(concatenated_y / T, dim=1)
    q = F.softmax(teacher_scores.cuda() / T, dim=1)

    p = p.view(-1, 2)
    q = q.view(-1, 2)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)

    return l_kl

def teacher_mid_2_student_min(s_min, t_mid, T):
    min_height, min_width = min([s.shape[2:] for s in t_mid])

    # Interpolate the larger student feature maps to match the smallest spatial dimensions
    for i in range(len(t_mid)):
        if t_mid[i].shape[2:] != (min_height, min_width):
            t_mid[i] = F.interpolate(t_mid[i], size=(min_height, min_width), mode='bilinear', align_corners=False)

    # Concatenate all tensors in y along the channel dimension (dim=1)
    concatenated_t_mid = torch.cat(t_mid, dim=1)

    # Ensure that both input and weight are on the same device (CUDA in this case)
    concatenated_t_mid = concatenated_t_mid.cuda()

    channel_decomp = nn.Conv2d(concatenated_t_mid.shape[1], s_min.shape[1], kernel_size=1).cuda()
    concatenated_t_mid = channel_decomp(concatenated_t_mid)

    if concatenated_t_mid.shape[2:] != s_min.shape[2:]:
        concatenated_t_mid = F.interpolate(concatenated_t_mid, s_min.shape[2:], mode='bilinear', align_corners=False)

    # Calculate KL divergence loss
    p = F.log_softmax(s_min / T, dim=1)
    q = F.softmax(concatenated_t_mid.cuda() / T, dim=1)

    return p, q

def rec_prediction_map_distillation(s_min, t_mid, T=4):
    
    # Find the spatial dimensions of the smallest student feature map
    p, q = teacher_mid_2_student_min(s_min, t_mid, T)
    
    p = p.view(-1, 2)
    q = q.view(-1, 2)

    l_kl = F.kl_div(p, q, reduction='batchmean') * (T ** 2)

    return l_kl

def at(x, exp):
    """
    attention value of a feature map
    :param x: feature
    :return: attention value
    """
    return F.normalize(x.pow(exp).mean(1).view(x.size(0), -1))


def importance_maps_distillation(s, t, exp=4):
   
    if s.shape[2] != t.shape[2]:
        s = F.interpolate(s, t.size()[-2:], mode='bilinear')
    return torch.sum((at(s, exp) - at(t, exp)).pow(2), dim=1).mean()


def cat_importance_maps_distillation(s_list, t, exp=4):
   
    num_feature_maps = len(s_list)

    # Find the spatial dimensions of the smallest student feature map
    min_height, min_width = min([s.shape[2:] for s in s_list])

    # Interpolate the larger student feature maps to match the smallest spatial dimensions
    for i in range(len(s_list)):
        if s_list[i].shape[2:] != (min_height, min_width):
            s_list[i] = F.interpolate(s_list[i], size=(min_height, min_width), mode='bilinear', align_corners=False)

    # Concatenate the tensors in s_list into a single tensor
    concatenated_s = torch.cat(s_list, dim=1)

    # Check and interpolate if necessary to match the spatial dimensions of the teacher feature map
    if concatenated_s.shape[2:] != t.shape[2:]:
        concatenated_s = F.interpolate(concatenated_s, t.shape[2:], mode='bilinear', align_corners=False)

    # Calculate the IMD loss
    loss = torch.sum((at(concatenated_s, exp) - at(t, exp)).pow(2), dim=1)

    return loss.mean()

def recur_cat_importance_maps_distillation(s, t_list, exp=4):
    
    num_feature_maps = len(t_list)

    # Find the spatial dimensions of the smallest student feature map
    min_height, min_width = min([s.shape[2:] for s in t_list])

    # Interpolate the larger student feature maps to match the smallest spatial dimensions
    for i in range(len(t_list)):
        if t_list[i].shape[2:] != (min_height, min_width):
            t_list[i] = F.interpolate(t_list[i], size=(min_height, min_width), mode='bilinear', align_corners=False)

    # Concatenate the tensors in s_list into a single tensor
    concatenated_t = torch.cat(t_list, dim=1)

    # Check and interpolate if necessary to match the spatial dimensions of the teacher feature map
    if concatenated_t.shape[2:] != s.shape[2:]:
        concatenated_t = F.interpolate(concatenated_t, s.shape[2:], mode='bilinear', align_corners=False)

    # Calculate the IMD loss
    loss = torch.sum((at(s, exp) - at(concatenated_t, exp)).pow(2), dim=1)

    return loss.mean()

