from torch.cuda import amp

def build_all_scaler():
    scaler_cls = amp.GradScaler()
    scaler_spl = amp.GradScaler()
    return scaler_cls, scaler_spl