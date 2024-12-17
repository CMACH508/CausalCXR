import os
import torch
import shutil

def load_checkpoint(config, model, opt, lrs, scaler, map_location, logger):
    logger.info(f"==============> Resuming from {config['resume_path']}....................")
    epoch = config['resume_epoch']
    checkpoint = torch.load(os.path.join(config['resume_path'], f'ckpt_epoch_{epoch}.pth'), map_location=map_location)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    opt[0].load_state_dict(checkpoint['opt_cls'])
    lrs[0].load_state_dict(checkpoint['lrs_cls'])
    scaler[0].load_state_dict(checkpoint['scaler_cls'])
    opt[1].load_state_dict(checkpoint['opt_spl'])
    lrs[1].load_state_dict(checkpoint['lrs_spl'])
    scaler[1].load_state_dict(checkpoint['scaler_spl'])
    logger.info(f"=> loaded successfully (epoch {config['resume_epoch']})")
            
    del checkpoint
    torch.cuda.empty_cache()
    
def load_weights(path, model, map_location, logger):
    logger.info(f"==============> Resuming from {path}....................")
    checkpoint = torch.load(path, map_location=map_location)
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)

    del checkpoint
    torch.cuda.empty_cache()

def load_pretrained(config, model, map_location, logger):
    pretrain_path = os.path.join(config['pretrained_path'], config['pretrained_model'])
    logger.info(f"==============> Loading weight {pretrain_path} for fine-tuning......")
    checkpoint = torch.load(pretrain_path, map_location=map_location)
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)
    
    logger.info(f"=> loaded successfully '{pretrain_path}'")

    del checkpoint
    torch.cuda.empty_cache()

def save_checkpoint(epoch, config, model, opt, lrs, scaler, is_best, logger):
    save_state = {'model': model.state_dict(),
                  'opt_cls': opt[0].state_dict(),
                  'lrs_cls': lrs[0].state_dict(),
                  'scaler_cls': scaler[0].state_dict(),
                  'opt_spl': opt[1].state_dict(),
                  'lrs_spl': lrs[1].state_dict(),
                  'scaler_spl': scaler[1].state_dict()}
    save_path = os.path.join(config['resume_path'], f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    if is_best:
        best_path = os.path.join(config['resume_path'], 'ckpt_best.pth')
        shutil.copyfile(save_path, best_path)
    logger.info(f"{save_path} saved !!!")