"""
get model func    Script  ver： Oct 18th 19:20
"""
import os, sys
sys.path.append(os.path.realpath('.'))

import torch
import torch.nn as nn
from torchvision import models


# get model
def get_model(num_classes=1000, edge_size=224, model_idx=None, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
              pretrained_backbone=True, use_cls_token=True, use_pos_embedding=True, use_att_module='SimAM'):
    """
    :param num_classes: classification required number of your dataset
    :param edge_size: the input edge size of the dataloder
    :param model_idx: the model we are going to use. by the format of Model_size_other_info

    :param drop_rate: The dropout layer's probility of proposed models
    :param attn_drop_rate: The dropout layer(right after the MHSA block or MHGA block)'s probility of proposed models
    :param drop_path_rate: The probility of stochastic depth

    :param pretrained_backbone: The backbone CNN is initiate randomly or by its official Pretrained models

    :param use_cls_token: To use the class token
    :param use_pos_embedding: To use the positional enbedding
    :param use_att_module: To use which attention module in the FGD Focus block

    :return: prepared model
    """
    if model_idx[0:5] == 'ViT_h':
        # Transfer learning for ViT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_huge_patch14_224_in21k', pretrained=pretrained_backbone, num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:5] == 'ViT_l':
        # Transfer learning for ViT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_large_patch16_224', pretrained=pretrained_backbone, num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_large_patch16_384', pretrained=pretrained_backbone, num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:5] == 'ViT_s':
        # Transfer learning for ViT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_small_patch16_224', pretrained=pretrained_backbone, num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_small_patch16_384', pretrained=pretrained_backbone, num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:5] == 'ViT_t':
        # Transfer learning for ViT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained_backbone, num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_tiny_patch16_384', pretrained=pretrained_backbone, num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:5] == 'ViT_b' or model_idx[0:3] == 'ViT':  # vit_base
        # Transfer learning for ViT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit*')
        pprint(model_names)
        if edge_size == 224:
            model = timm.create_model('vit_base_patch16_224', pretrained=pretrained_backbone, num_classes=num_classes)
        elif edge_size == 384:
            model = timm.create_model('vit_base_patch16_384', pretrained=pretrained_backbone, num_classes=num_classes)
        else:
            print('not a avaliable image size with', model_idx)

    elif model_idx[0:3] == 'vgg':
        # Transfer learning for vgg16_bn
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vgg*')
        pprint(model_names)
        if model_idx[0:8] == 'vgg16_bn':
            model = timm.create_model('vgg16_bn', pretrained=pretrained_backbone, num_classes=num_classes)
        elif model_idx[0:5] == 'vgg16':
            model = timm.create_model('vgg16', pretrained=pretrained_backbone, num_classes=num_classes)
        elif model_idx[0:8] == 'vgg19_bn':
            model = timm.create_model('vgg19_bn', pretrained=pretrained_backbone, num_classes=num_classes)
        elif model_idx[0:5] == 'vgg19':
            model = timm.create_model('vgg19', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:4] == 'deit':  # Transfer learning for DeiT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*deit*')
        pprint(model_names)
        if edge_size == 384:
            model = timm.create_model('deit_base_patch16_384', pretrained=pretrained_backbone, num_classes=2)
        elif edge_size == 224:
            model = timm.create_model('deit_base_patch16_224', pretrained=pretrained_backbone, num_classes=2)
        else:
            pass

    elif model_idx[0:5] == 'twins':  # Transfer learning for twins
        import timm
        from pprint import pprint

        model_names = timm.list_models('*twins*')
        pprint(model_names)
        model = timm.create_model('twins_pcpvt_base', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:5] == 'pit_b' and edge_size == 224:  # Transfer learning for coat_mini
        import timm
        from pprint import pprint

        model_names = timm.list_models('*pit*')
        pprint(model_names)
        model = timm.create_model('pit_b_224', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:6] == 'convit' and edge_size == 224:  # Transfer learning for ConViT
        import timm
        from pprint import pprint

        model_names = timm.list_models('*convit*')
        pprint(model_names)
        model = timm.create_model('convit_base', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:6] == 'ResNet':  # Transfer learning for the ResNets
        if model_idx[0:8] == 'ResNet34':
            model = models.resnet34(pretrained=pretrained_backbone)
        elif model_idx[0:8] == 'ResNet50':
            model = models.resnet50(pretrained=pretrained_backbone)
        elif model_idx[0:9] == 'ResNet101':
            model = models.resnet101(pretrained=pretrained_backbone)
        else:
            print('this model is not defined in get model')
            return -1
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_idx[0:7] == 'bot_256' and edge_size == 256:  # Model: BoT
        import timm
        from pprint import pprint
        model_names = timm.list_models('*bot*')
        pprint(model_names)
        # NOTICE: we find no weight for BoT in timm
        # ['botnet26t_256', 'botnet50ts_256', 'eca_botnext26ts_256']
        model = timm.create_model('botnet26t_256', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:8] == 'densenet':  # Transfer learning for densenet
        import timm
        from pprint import pprint

        model_names = timm.list_models('*densenet*')
        pprint(model_names)
        model = timm.create_model('densenet121', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:8] == 'xception':  # Transfer learning for Xception
        import timm
        from pprint import pprint
        model_names = timm.list_models('*xception*')
        pprint(model_names)
        model = timm.create_model('xception', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:9] == 'visformer' and edge_size == 224:  # Transfer learning for Visformer
        import timm
        from pprint import pprint
        model_names = timm.list_models('*visformer*')
        pprint(model_names)
        model = timm.create_model('visformer_small', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:9] == 'conformer':  # Transfer learning for Conformer base
        from Models import conformer

        embed_dim = 576
        channel_ratio = 6

        if pretrained_backbone:
            model = conformer.Conformer(num_classes=1000, patch_size=16, channel_ratio=channel_ratio,
                                        embed_dim=embed_dim, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)
            # this is the related path to <code>, not <Models>
            save_model_path = '../saved_models/Conformer_base_patch16.pth'  # model is downloaded at this path
            # downloaded from official model state at https://github.com/pengzhiliang/Conformer
            model.load_state_dict(torch.load(save_model_path), False)

            model.trans_cls_head = nn.Linear(embed_dim, num_classes)
            model.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)
            model.cls_head = nn.Linear(int(2 * num_classes), num_classes)

        else:
            model = conformer.Conformer(num_classes=num_classes, patch_size=16, channel_ratio=channel_ratio,
                                        embed_dim=embed_dim, depth=12, num_heads=9, mlp_ratio=4, qkv_bias=True)

    elif model_idx[0:9] == 'coat_mini' and edge_size == 224:  # Transfer learning for coat_mini
        import timm
        from pprint import pprint

        model_names = timm.list_models('*coat*')
        pprint(model_names)
        model = timm.create_model('coat_mini', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:10] == 'swin_b_384' and edge_size == 384:  # Transfer learning for Swin Transformer (swin_b_384)
        import timm
        from pprint import pprint
        model_names = timm.list_models('*swin*')
        pprint(model_names)  # swin_base_patch4_window12_384  swin_base_patch4_window12_384_in22k
        model = timm.create_model('swin_base_patch4_window12_384', pretrained=pretrained_backbone,
                                  num_classes=num_classes)

    elif model_idx[0:10] == 'swin_b_224' and edge_size == 224:  # Transfer learning for Swin Transformer (swin_b_384)
        import timm
        from pprint import pprint
        model_names = timm.list_models('*swin*')
        pprint(model_names)  # swin_base_patch4_window7_224  swin_base_patch4_window7_224_in22k
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained_backbone,
                                  num_classes=num_classes)

    elif model_idx[0:11] == 'mobilenetv3':  # Transfer learning for mobilenetv3
        import timm
        from pprint import pprint
        model_names = timm.list_models('*mobilenet*')
        pprint(model_names)
        model = timm.create_model('mobilenetv3_large_100', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:11] == 'inceptionv3':  # Transfer learning for Inception v3
        import timm
        from pprint import pprint
        model_names = timm.list_models('*inception*')
        pprint(model_names)
        model = timm.create_model('inception_v3', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:12] == 'cross_former' and edge_size == 224:  # Transfer learning for crossformer base
        from Models import crossformer
        backbone = crossformer.CrossFormer(img_size=edge_size,
                                           patch_size=[4, 8, 16, 32],
                                           in_chans=3,
                                           num_classes=0,  # get backbone only
                                           embed_dim=96,
                                           depths=[2, 2, 18, 2],
                                           num_heads=[3, 6, 12, 24],
                                           group_size=[7, 7, 7, 7],
                                           mlp_ratio=4.,
                                           qkv_bias=True,
                                           qk_scale=None,
                                           drop_rate=0.0,
                                           drop_path_rate=0.3,
                                           ape=False,
                                           patch_norm=True,
                                           use_checkpoint=False,
                                           merge_size=[[2, 4], [2, 4], [2, 4]], )
        if pretrained_backbone:
            save_model_path = '../saved_models/crossformer-b.pth'  # model is downloaded at this path
            # downloaded from official model state at https://github.com/cheerss/CrossFormer
            backbone.load_state_dict(torch.load(save_model_path)['model'], False)
        model = crossformer.cross_former_cls_head_warp(backbone, num_classes)

    elif model_idx[0:14] == 'efficientnet_b':  # Transfer learning for efficientnet_b3,4
        import timm
        from pprint import pprint
        model_names = timm.list_models('*efficientnet*')
        pprint(model_names)
        model = timm.create_model(model_idx[0:15], pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:14] == 'ResN50_ViT_384':  # ResNet+ViT融合模型384
        import timm
        from pprint import pprint
        model_names = timm.list_models('*vit_base_resnet*')
        pprint(model_names)
        model = timm.create_model('vit_base_resnet50_384', pretrained=pretrained_backbone, num_classes=num_classes)

    elif model_idx[0:15] == 'coat_lite_small' and edge_size == 224:  # Transfer learning for coat_lite_small
        import timm
        from pprint import pprint

        model_names = timm.list_models('*coat*')
        pprint(model_names)
        model = timm.create_model('coat_lite_small', pretrained=pretrained_backbone, num_classes=num_classes)

    else:
        print('\nThe model', model_idx, 'with the edge size of', edge_size)
        print("is not defined in the script！！", '\n')
        return -1

    try:
        img = torch.randn(1, 3, edge_size, edge_size)
        preds = model(img)  # (1, class_number)
        print('test model output：', preds)
    except:
        print("Problem exist in the model defining process！！")
        return -1
    else:
        print('model is ready now!')
        return model
