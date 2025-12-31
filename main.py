import json
import os
import sys
import time
from os import path as osp
from pathlib import Path
from shutil import copyfile
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from model_temporal_kf import TCNSeqNetwork, TransformerSeqNetwork, HybridTCNTransformer
from differentiable_kf import DifferentiableKalmanFilter, BatchDifferentiableKF, integrate_velocities
from utils import load_config, MSEAverageMeter
from data_glob_speed import GlobSpeedSequence, SequenceToSequenceDataset
from transformations import ComposeTransform, RandomHoriRotateSeq
from metric import compute_absolute_trajectory_error, compute_relative_trajectory_error
from pretrained_wt_load import load_tcn_pretrained_weights, load_tfm_pretrained_weights
from kf_util import kf_smooth_velocity #using default KF params
'''
Temporal models with learnable Kalman Filter parameters
Configurations
    - Model types 
        TCN - type=tcn
        Transformer - type=transformer
        Hybrid - type=hybrid
'''
torch.multiprocessing.set_sharing_strategy('file_system')
_nano_to_sec = 1e09
_input_channel, _output_channel = 6, 2
device = 'cuda'

class GlobalPosLoss(torch.nn.Module):
    def __init__(self, mode='full', history=None):
        """
        Calculate position loss in global coordinate frame
        Target :- Global Velocity
        Prediction :- Global Velocity
        """
        super(GlobalPosLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

        assert mode in ['full', 'part']
        self.mode = mode
        if self.mode == 'part':
            assert history is not None
            self.history = history
        elif self.mode == 'full':
            self.history = 1

    def forward(self, pred, targ):
        gt_pos = torch.cumsum(targ[:, 1:, ], 1)
        pred_pos = torch.cumsum(pred[:, 1:, ], 1)
        if self.mode == 'part':
            gt_pos = gt_pos[:, self.history:, :] - gt_pos[:, :-self.history, :]
            pred_pos = pred_pos[:, self.history:, :] - pred_pos[:, :-self.history, :]
        loss = self.mse_loss(pred_pos, gt_pos)
        return torch.mean(loss)


class KFFilteredPosLoss(torch.nn.Module):
    """
    Loss that applies differentiable KF before computing position error
    Allows end-to-end training of model + KF parameters
    """
    def __init__(self, dt=0.005, mode='part', history=None, use_batched_kf=True):
        super(KFFilteredPosLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')
        self.dt = dt
        self.mode = mode
        self.history = history if history is not None else 1
        
        # Use batched version for better performance
        if use_batched_kf:
            self.kf = BatchDifferentiableKF()
        else:
            self.kf = DifferentiableKalmanFilter()
    
    def forward(self, pred_vel, targ_vel, q_pos, q_vel, r_vel, p0=None):
        """
        Args:
            pred_vel: (batch, T, 2) predicted velocities
            targ_vel: (batch, T, 2) ground truth velocities
            q_pos, q_vel, r_vel: KF noise parameters
            p0: (batch, 2) initial positions (if None, use zeros)
        """
        batch_size = pred_vel.shape[0]
        device = pred_vel.device
        
        # Initialize p0 if not provided
        if p0 is None:
            p0 = torch.zeros(batch_size, 2, device=device, dtype=pred_vel.dtype)
        
        # Apply KF to predictions
        pred_pos_filtered, _ = self.kf(pred_vel, self.dt, p0, q_pos, q_vel, r_vel)
        
        # Integrate ground truth velocities
        gt_pos = integrate_velocities(targ_vel, self.dt, p0)
        
        # Compute loss based on mode
        if self.mode == 'full':
            loss = self.mse_loss(pred_pos_filtered, gt_pos)
        elif self.mode == 'part':
            # Windowed loss
            pred_windowed = pred_pos_filtered[:, self.history:, :] - pred_pos_filtered[:, :-self.history, :]
            gt_windowed = gt_pos[:, self.history:, :] - gt_pos[:, :-self.history, :]
            loss = self.mse_loss(pred_windowed, gt_windowed)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return torch.mean(loss)


def write_config(args, **kwargs):
    if args.out_dir:
        with open(osp.join(args.out_dir, 'config.json'), 'w') as f:
            values = vars(args)
            values['file'] = "main"
            if kwargs:
                values['kwargs'] = kwargs
            json.dump(values, f, sort_keys=True)


def get_dataset(root_dir, data_list, args, **kwargs):
    input_format, output_format = [0, 3, 6], [0, _output_channel]
    mode = kwargs.get('mode', 'train')

    random_shift, shuffle, transforms, grv_only = 0, False, [], False

    if mode == 'train':
        random_shift = args.step_size // 2
        shuffle = True
        transforms.append(RandomHoriRotateSeq(input_format, output_format))
    elif mode == 'val':
        shuffle = True
    elif mode == 'test':
        shuffle = False
        grv_only = True
    transforms = ComposeTransform(transforms)

    seq_type = GlobSpeedSequence
    
    dataset = SequenceToSequenceDataset(seq_type, root_dir, data_list, args.cache_path, args.step_size, args.window_size,
                                        random_shift=random_shift, transform=transforms, shuffle=shuffle,
                                        grv_only=grv_only, **kwargs)

    return dataset


def get_dataset_from_list(root_dir, list_path, args, **kwargs):
    with open(list_path) as f:
        data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    return get_dataset(root_dir, data_list, args, **kwargs)


def get_model(args, **kwargs):
    _input_channel = kwargs.get('input_channel', 6)
    _output_channel = kwargs.get('output_channel', 2)
    use_learnable_kf = kwargs.get('use_learnable_kf', False)
    
    config = {}
    if kwargs.get('dropout'):
        config['dropout'] = kwargs.get('dropout')
    
    config['use_learnable_kf'] = use_learnable_kf

    if args.type == 'tcn':
        network = TCNSeqNetwork(_input_channel, _output_channel, args.kernel_size,
                                layer_channels=args.channels, **config)
        print("TCN Network. Receptive field: {} ".format(network.get_receptive_field()))

        # Load pretrained weights if specified
        if hasattr(args, 'pretrained_tcn') and args.pretrained_tcn : #and not use_learnable_kf
            network = load_tcn_pretrained_weights(network, args.pretrained_tcn, 
                                                 network_type='tcn')
    
    elif args.type == 'transformer':
        network = TransformerSeqNetwork(
            _input_channel, _output_channel,
            d_model=args.d_model if hasattr(args, 'd_model') else 128,
            nhead=args.nhead if hasattr(args, 'nhead') else 4,
            **config
        )
        # Load pretrained weights if specified
        if hasattr(args, 'pretrained_tfm') and args.pretrained_tfm : #and not use_learnable_kf
            network = load_tfm_pretrained_weights(network, args.pretrained_tfm, 
                                                    network_type='transformer')

        print("Transformer Network. Receptive field: {} ".format(network.get_receptive_field()))
    
    elif args.type == 'hybrid':
        # Parse TCN channels if provided as string
        tcn_channels = args.channels if isinstance(args.channels, list) else \
                       [int(c) for c in args.channels.split(',')] if args.channels else [64, 64]
        
        network = HybridTCNTransformer(
            _input_channel, _output_channel,
            tcn_kernel_size=args.kernel_size,
            tcn_layer_channels=tcn_channels,
            transformer_d_model=args.d_model if hasattr(args, 'd_model') else 128,
            transformer_nhead=args.nhead if hasattr(args, 'nhead') else 4,
            dropout=config.get('dropout', 0.2),
            combination=args.combination if hasattr(args, 'combination') else 'concat',
            **config
        )
        print("Hybrid TCN-Transformer Network. Combination: {}".format(
            args.combination if hasattr(args, 'combination') else 'concat'))
        
        # Load pretrained TCN weights if specified
        if hasattr(args, 'pretrained_tcn') and args.pretrained_tcn :            #and not use_learnable_kf
            network = load_tcn_pretrained_weights(network, args.pretrained_tcn, 
                                                 network_type='hybrid')
        # Load pretrained weights if specified
        if hasattr(args, 'pretrained_tfm') and args.pretrained_tfm : #and not use_learnable_kf
            network = load_tfm_pretrained_weights(network, args.pretrained_tfm, 
                                                    network_type='hybrid')
        
            
    else:
        raise ValueError(f"Unknown network type: {args.type}")
    
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    
    if use_learnable_kf:
        print("Using learnable Kalman Filter parameters")
        kf_params = network.get_kf_params()
        if kf_params is not None:
            q_pos, q_vel, r_vel = kf_params
            print(f"  Initial KF params: q_pos={q_pos.item():.6f}, q_vel={q_vel.item():.6f}, r_vel={r_vel.item():.6f}")
    
    return network


def get_loss_function(history, args, **kwargs):
    """
    Get appropriate loss function based on model type and training mode
    """
    use_learnable_kf = kwargs.get('use_learnable_kf', False)
    
    if use_learnable_kf:
        # Use KF-filtered loss for end-to-end training
        dt = kwargs.get('dt', 0.005)  #0.01 #Time step, should be set from data
        print(f"✓ Using RoNIN preprocessing default: {dt:.6f} seconds (200 Hz)")
        
        if args.type == 'tcn':
            config = {'mode': 'part', 'history': history, 'dt': dt}
            print(f"Using KFFilteredPosLoss with mode='part', history={history}")
        elif args.type in ['transformer', 'hybrid']:
            config = {'mode': 'part', 'history': history, 'dt': dt}
            print(f"Using KFFilteredPosLoss with mode='part',history={history}")
        
        criterion = KFFilteredPosLoss(**config)
    
    else:
        # Use standard velocity-based loss
        if args.type == 'tcn':
            config = {'mode': 'part', 'history': history}
            print(f"Using GlobalPosLoss with mode='part', history={history}")
        elif args.type in ['transformer', 'hybrid']:
            config = {'mode': 'part','history': history,}
            print(f"Using GlobalPosLoss with mode='part',history={history}")
        
        criterion = GlobalPosLoss(**config)
    
    return criterion


def format_string(*argv, sep=' '):
    result = ''
    for val in argv:
        if isinstance(val, (tuple, list, np.ndarray)):
            for v in val:
                result += format_string(v, sep=sep) + sep
        else:
            result += str(val) + sep
    return result[:-1]


def train(args, **kwargs):
    # Loading data
    start_t = time.time()
    #use_learnable_kf = kwargs.get('use_learnable_kf', False)
    use_learnable_kf = args.use_learnable_kf
    
    train_dataset = get_dataset_from_list(args.data_dir, args.train_list, args, mode='train', **kwargs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True, prefetch_factor=2) 
    end_t = time.time()

    print('Training set loaded. Time usage: {:.3f}s'.format(end_t - start_t))
    val_dataset, val_loader = None, None
    if args.val_list is not None:
        val_dataset = get_dataset_from_list(args.data_dir, args.val_list, args, mode='val', **kwargs)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, persistent_workers=True, drop_last=True, prefetch_factor=2)
        print('Validation set loaded')

    global device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if args.out_dir:
        if not osp.isdir(args.out_dir):
            os.makedirs(args.out_dir)
        if not osp.isdir(osp.join(args.out_dir, 'checkpoints')):
            os.makedirs(osp.join(args.out_dir, 'checkpoints'))
        if not osp.isdir(osp.join(args.out_dir, 'logs')):
            os.makedirs(osp.join(args.out_dir, 'logs'))
        copyfile(args.train_list, osp.join(args.out_dir, "train_list"))
        if args.val_list is not None:
            copyfile(args.val_list, osp.join(args.out_dir, "validation_list"))
        write_config(args, **kwargs)

    print('\nNumber of train samples: {}'.format(len(train_dataset)))
    train_mini_batches = len(train_loader)
    if val_dataset:
        print('Number of val samples: {}'.format(len(val_dataset)))
        val_mini_batches = len(val_loader)

    kwargs['use_learnable_kf']=args.use_learnable_kf
    network = get_model(args, **kwargs).to(device)  
    history = network.get_receptive_field() if args.type == 'tcn' else args.window_size // 2
    
    criterion = get_loss_function(history, args, **kwargs)
    optimizer = torch.optim.Adam(network.parameters(), args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.75)
    use_amp = torch.cuda.is_available()
    if use_amp:
        scaler = GradScaler("cuda") 
        print("Using Automatic Mixed Precision (AMP) training")

    quiet_mode = kwargs.get('quiet', False)
    use_scheduler = kwargs.get('use_scheduler', False)

    log_file = None
    if args.out_dir:
        log_file = osp.join(args.out_dir, 'logs', 'log.txt')
        if osp.exists(log_file):
            if args.continue_from is None:
                os.remove(log_file)
            else:
                copyfile(log_file, osp.join(args.out_dir, 'logs', 'log_old.txt'))

    start_epoch = 0
    if args.continue_from is not None and osp.exists(args.continue_from):
        with open(osp.join(str(Path(args.continue_from).parents[1]), 'config.json'), 'r') as f:
            model_data = json.load(f)

        if device.type == 'cpu':
            checkpoints = torch.load(args.continue_from, map_location=lambda storage, location: storage, weights_only=False)
        else:
            checkpoints = torch.load(args.continue_from, map_location={model_data['device']: args.device}, weights_only=False)

        start_epoch = checkpoints.get('epoch', 0)
        network.load_state_dict(checkpoints.get('model_state_dict'), strict=False)
        optimizer.load_state_dict(checkpoints.get('optimizer_state_dict'))
    
    if kwargs.get('force_lr', False):
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

    step = 0
    best_val_loss = np.inf
    train_errs = np.zeros(args.epochs)

    print("Starting from epoch {}".format(start_epoch))
    try:
        for epoch in range(start_epoch, args.epochs):
            log_line = ''
            network.train()
            train_vel = MSEAverageMeter(3, [2], _output_channel)
            train_loss = 0
            start_t = time.time()

            for bid, batch in enumerate(train_loader):
                feat, targ, _, _ = batch
                feat, targ = feat.to(device), targ.to(device)
                optimizer.zero_grad()
                
                # Get predictions
                predicted = network(feat)
                train_vel.add(predicted.cpu().detach().numpy(), targ.cpu().detach().numpy())
                
                # Compute loss
                if use_learnable_kf:
                    # Get learned KF parameters
                    kf_params = network.get_kf_params()
                    if kf_params is not None:
                        q_pos, q_vel, r_vel = kf_params
                        p0 = torch.zeros(feat.shape[0], 2, device=device)
                        loss = criterion(predicted, targ, q_pos, q_vel, r_vel, p0)
                    else:
                        raise ValueError("KF parameters not available but use_learnable_kf is True")
                else:
                    loss = criterion(predicted, targ)
                
                train_loss += loss.cpu().detach().numpy()
                loss.backward()
                optimizer.step()
                step += 1

            train_errs[epoch] = train_loss / train_mini_batches
            end_t = time.time()
            
            # Log KF parameters if using learnable KF
            kf_log = ""
            if use_learnable_kf:
                kf_params = network.get_kf_params()
                if kf_params is not None:
                    q_pos, q_vel, r_vel = kf_params
                    kf_log = f" | KF: q_pos={q_pos.item():.6f} q_vel={q_vel.item():.6f} r_vel={r_vel.item():.6f}"
            
            if not quiet_mode:
                print('-' * 25)
                print('Epoch {}, time usage: {:.3f}s, loss: {}, vel_loss {}/{:.6f}{}'.format(
                    epoch, end_t - start_t, train_errs[epoch], train_vel.get_channel_avg(), 
                    train_vel.get_total_avg(), kf_log))
            
            log_line = format_string(log_line, epoch, optimizer.param_groups[0]['lr'], train_errs[epoch],
                                     *train_vel.get_channel_avg())

            saved_model = False
            if val_loader and epoch % 5 == 0:
                network.eval()
                val_vel = MSEAverageMeter(3, [2], _output_channel)
                val_loss = 0
                for bid, batch in enumerate(val_loader):
                    feat, targ, _, _ = batch
                    feat, targ = feat.to(device), targ.to(device)
                    
                    with torch.no_grad():
                        pred = network(feat)
                        val_vel.add(pred.cpu().detach().numpy(), targ.cpu().detach().numpy())
                        
                        if use_learnable_kf:
                            kf_params = network.get_kf_params()
                            if kf_params is not None:
                                q_pos, q_vel, r_vel = kf_params
                                p0 = torch.zeros(feat.shape[0], 2, device=device)
                                val_loss += criterion(pred, targ, q_pos, q_vel, r_vel, p0).cpu().detach().numpy()
                        else:
                            val_loss += criterion(pred, targ).cpu().detach().numpy()
                
                val_loss = val_loss / val_mini_batches
                log_line = format_string(log_line, val_loss, *val_vel.get_channel_avg())
                if not quiet_mode:
                    print('Validation loss: {} vel_loss: {}/{:.6f}{}'.format(
                        val_loss, val_vel.get_channel_avg(), val_vel.get_total_avg(), kf_log))
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    saved_model = True
                    if args.out_dir:
                        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_%d.pt' % epoch)
                        torch.save({'model_state_dict': network.state_dict(),
                                    'epoch': epoch,
                                    'loss': train_errs[epoch],
                                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
                        print('Best Validation Model saved to ', model_path)
                if use_scheduler:
                    scheduler.step(val_loss)

            if args.out_dir and not saved_model and (epoch + 1) % args.save_interval == 0:
                model_path = osp.join(args.out_dir, 'checkpoints', 'icheckpoint_%d.pt' % epoch)
                torch.save({'model_state_dict': network.state_dict(),
                            'epoch': epoch,
                            'loss': train_errs[epoch],
                            'optimizer_state_dict': optimizer.state_dict()}, model_path)
                print('Model saved to ', model_path)

            if log_file:
                log_line += '\n'
                with open(log_file, 'a') as f:
                    f.write(log_line)
            if np.isnan(train_loss):
                print("Invalid value. Stopping training.")
                break
    except KeyboardInterrupt:
        print('-' * 60)
        print('Early terminate')

    print('Training completed')
    if args.out_dir:
        model_path = osp.join(args.out_dir, 'checkpoints', 'checkpoint_latest.pt')
        torch.save({'model_state_dict': network.state_dict(),
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)


def recon_traj_with_preds_global(dataset, preds, ind=None, seq_id=0, type='preds', **kwargs):
    ind = ind if ind is not None else np.array([i[1] for i in dataset.index_map if i[0] == seq_id], dtype=np.int)

    if type == 'gt':
        pos = dataset.gt_pos[seq_id][:, :2]
    else:
        ts = dataset.ts[seq_id]
        dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
        pos = preds * dts
        pos[0, :] = dataset.gt_pos[seq_id][0, :2]
        pos = np.cumsum(pos, axis=0)
    veloc = preds
    ori = dataset.orientations[seq_id]

    return pos, veloc, ori


def test(args, **kwargs):
    global device, _output_channel
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    #use_learnable_kf = kwargs.get('use_learnable_kf', False)
    use_learnable_kf = args.use_learnable_kf

    if args.test_path is not None:
        if args.test_path[-1] == '/':
            args.test_path = args.test_path[:-1]
        #root_dir = osp.split(args.test_path)[0]
        root_dir = kwargs.get('data_dir', args.data_dir) if kwargs.get('data_dir') else osp.split(args.test_list)[0]
        test_data_list = [osp.split(args.test_path)[1]]
    elif args.test_list is not None:
        #root_dir = args.data_dir if args.data_dir else osp.split(args.test_list)[0]
        root_dir = kwargs.get('data_dir', args.data_dir) if kwargs.get('data_dir') else osp.split(args.test_list)[0]
        with open(args.test_list) as f:
            test_data_list = [s.strip().split(',')[0] for s in f.readlines() if len(s) > 0 and s[0] != '#']
    else:
        raise ValueError('Either test_path or test_list must be specified.')
    _ = get_dataset(root_dir, [test_data_list[0]], args, mode='test')

    if args.out_dir and not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    with open(osp.join(str(Path(args.model_path).parents[1]), 'config.json'), 'r') as f:
        model_data = json.load(f)

    if device.type == 'cpu':
        checkpoint = torch.load(args.model_path, map_location=lambda storage, location: storage, weights_only=False)
    else:
        checkpoint = torch.load(args.model_path, map_location={model_data['device']: args.device}, weights_only=False)

    kwargs['use_learnable_kf']=args.use_learnable_kf
    network = get_model(args, **kwargs)
    network.load_state_dict(checkpoint.get('model_state_dict'), strict=False)
    network.eval().to(device)
    print('Model {} loaded to device {}.'.format(args.model_path, device))
    
    # Get learned KF parameters if available
    learned_kf_params = None
    if use_learnable_kf:
        kf_params = network.get_kf_params()
        if kf_params is not None:
            q_pos, q_vel, r_vel = kf_params
            learned_kf_params = (q_pos.item(), q_vel.item(), r_vel.item())
            print(f"Using learned KF params: q_pos={q_pos.item():.6f}, q_vel={q_vel.item():.6f}, r_vel={r_vel.item():.6f}")

    log_file = None
    if args.test_list and args.out_dir:
        log_file = osp.join(args.out_dir, osp.split(args.test_list)[-1].split('.')[0] + '_log.txt')
        with open(log_file, 'w') as f:
            f.write(args.model_path + '\n')
            if learned_kf_params:
                f.write(f'Learned KF params: q_pos={learned_kf_params[0]:.6f}, q_vel={learned_kf_params[1]:.6f}, r_vel={learned_kf_params[2]:.6f}\n')
            f.write('Seq traj_len velocity ate rte\n')

    losses_vel = MSEAverageMeter(2, [1], _output_channel)
    ate_all, rte_all = [], []
    pred_per_min = 200 * 60

    seq_dataset = get_dataset(root_dir, test_data_list, args, **kwargs)

    window = 512          
    stride = 512        # or use smaller than window for overlap
    pred_list = []

    for idx, data in enumerate(test_data_list):
        assert data == osp.split(seq_dataset.data_path[idx])[1]
        
        feat, vel = seq_dataset.get_test_seq(idx)
        feat = torch.Tensor(feat).to(device)
        #preds = np.squeeze(network(feat).cpu().detach().numpy())[-vel.shape[0]:, :_output_channel]

        with torch.no_grad():
            T = feat.shape[1]
            for start in range(0, T, stride):
                end = min(start + window, T)
                chunk = feat[:, start:end, :]     # [1, chunk_len, C]

                out = network(chunk)              # [1, chunk_len, output_channels]
                pred_list.append(out.cpu())

        preds_full = torch.cat(pred_list, dim=1)  # concat along time axis
        preds = np.squeeze(preds_full.numpy())[-vel.shape[0]:, :_output_channel]

        ind = np.arange(vel.shape[0])
        vel_losses = np.mean((vel - preds) ** 2, axis=0)

        print('Reconstructing trajectory')

        # Ground-truth trajectory
        pos_gt, gv_gt, _ = recon_traj_with_preds_global(seq_dataset, vel, ind=ind, type='gt', seq_id=idx)

        # Apply filtering based on configuration
        if args.filter_type == 'kf':
            ts = seq_dataset.ts[idx]
            dts = np.mean(ts[ind[1:]] - ts[ind[:-1]])
            p0 = seq_dataset.gt_pos[idx][0, :2]
            
            # Use learned KF params if available, otherwise use defaults
            if learned_kf_params:
                pos_pred, gv_pred = kf_smooth_velocity(
                    preds, dts, p0, 
                    q_pos=learned_kf_params[0],
                    q_vel=learned_kf_params[1],
                    r_vel=learned_kf_params[2]
                )
                print(f"Applied learned KF with params: {learned_kf_params}")
            else:
                print("No learned q_pos, q_vel, r_vel params found")
                pos_pred, gv_pred = kf_smooth_velocity(preds, dts, p0)
                print("Applied default KF")
              
            
            losses_vel.add(vel, gv_pred)
        else:
            # No filtering
            pos_pred, gv_pred, _ = recon_traj_with_preds_global(
                seq_dataset, preds, ind=ind, type='pred', seq_id=idx)
            losses_vel.add(vel, preds)
        
        if args.out_dir is not None and osp.isdir(args.out_dir):
            np.save(osp.join(args.out_dir, '{}_{}.npy'.format(data, args.type)),
                    np.concatenate([pos_pred, pos_gt], axis=1))

        ate = compute_absolute_trajectory_error(pos_pred, pos_gt)
        if pos_pred.shape[0] < pred_per_min:
            ratio = pred_per_min / pos_pred.shape[0]
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pos_pred.shape[0] - 1) * ratio
        else:
            rte = compute_relative_trajectory_error(pos_pred, pos_gt, delta=pred_per_min)
        pos_cum_error = np.linalg.norm(pos_pred - pos_gt, axis=1)
        ate_all.append(ate)
        rte_all.append(rte)

        print('Sequence {}, Velocity loss {} / {}, ATE: {}, RTE:{}'.format(data, vel_losses, np.mean(vel_losses), ate, rte))
        log_line = format_string(data, np.mean(vel_losses), ate, rte)

        if not args.fast_test:
            kp = preds.shape[1]
            if kp == 2:
                targ_names = ['vx', 'vy']
            elif kp == 3:
                targ_names = ['vx', 'vy', 'vz']

            plt.figure('{}'.format(data), figsize=(16, 9))
            plt.subplot2grid((kp, 2), (0, 0), rowspan=kp - 1)
            plt.plot(pos_pred[:, 0], pos_pred[:, 1])
            plt.plot(pos_gt[:, 0], pos_gt[:, 1])
            plt.title(data)
            plt.axis('equal')
            plt.legend(['Predicted', 'Ground truth'])
            plt.subplot2grid((kp, 2), (kp - 1, 0))
            plt.plot(pos_cum_error)
            plt.legend(['ATE:{:.3f}, RTE:{:.3f}'.format(ate_all[-1], rte_all[-1])])
            for i in range(kp):
                plt.subplot2grid((kp, 2), (i, 1))
                plt.plot(ind, preds[:, i])
                plt.plot(ind, vel[:, i])
                plt.legend(['Predicted', 'Ground truth'])
                plt.title('{}, error: {:.6f}'.format(targ_names[i], vel_losses[i]))
            plt.tight_layout()

            if args.show_plot:
                plt.show()

            if args.out_dir is not None and osp.isdir(args.out_dir):
                plt.savefig(osp.join(args.out_dir, '{}_{}.png'.format(data, args.type)))

        if log_file is not None:
            with open(log_file, 'a') as f:
                log_line += '\n'
                f.write(log_line)

        plt.close('all')

    ate_all = np.array(ate_all)
    rte_all = np.array(rte_all)

    measure = format_string('ATE', 'RTE', sep='\t')
    values = format_string(np.mean(ate_all), np.mean(rte_all), sep='\t')
    print(measure, '\n', values)

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(measure + '\n')
            f.write(values)


if __name__ == '__main__':
    default_config_file = osp.abspath(osp.join(osp.dirname(__file__), 'config/temporal_model_defaults.json'))
                          
    import argparse

    parser = argparse.ArgumentParser(description="Run seq2seq model with learnable KF parameters")
    parser.add_argument('--config', type=str, help='Configuration file', default=default_config_file)
    
    # common
    parser.add_argument('--type', type=str, choices=['tcn', 'transformer', 'hybrid'], help='Model type')
    parser.add_argument('--data_dir', type=str, default='dataset', help='Directory for data files')
    parser.add_argument('--cache_path', type=str, default=None)
    parser.add_argument('--feature_sigma', type=float, help='Gaussian for smoothing features')
    parser.add_argument('--target_sigma', type=float, help='Gaussian for smoothing target')
    parser.add_argument('--window_size', type=int)
    parser.add_argument('--step_size', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, help='Cuda device or cpu')
    parser.add_argument('--dataset', type=str, choices=['ronin'])
    
    # Learnable KF
    parser.add_argument('--use_learnable_kf', action='store_true',
                       help='Enable learnable Kalman Filter parameters (end-to-end training)')
    
    # tcn
    tcn_cmd = parser.add_argument_group('tcn', 'configuration for TCN')
    tcn_cmd.add_argument('--kernel_size', type=int)
    tcn_cmd.add_argument('--channels', type=str, help='Channel sizes for TCN layers')
    
    # transformer
    transformer_cmd = parser.add_argument_group('transformer', 'configuration for Transformer')
    transformer_cmd.add_argument('--d_model', type=int, default=128)
    transformer_cmd.add_argument('--nhead', type=int, default=4)
    transformer_cmd.add_argument('--num_layers', type=int, default=6)
    transformer_cmd.add_argument('--dim_feedforward', type=int, default=256)

    # hybrid
    hybrid_cmd = parser.add_argument_group('hybrid', 'configuration for Hybrid')
    hybrid_cmd.add_argument('--combination', type=str, default='concat',
                        choices=['concat', 'add', 'weighted'])
    
    mode = parser.add_subparsers(title='mode', dest='mode', help='train or test')
    mode.required = True
    
    # Pretrained weights
    parser.add_argument('--pretrained_tcn', type=str, default='ronin_tcn_checkpoint.pt',
                    help='Path to pretrained TCN checkpoint')
    
    parser.add_argument('--pretrained_tfm', type=str, default= 'runs/tfm_run1/kf/checkpoints/checkpoint_latest.pt',
                    help='Path to pretrained Transformer checkpoint')
      
    # train
    train_cmd = mode.add_parser('train')
    train_cmd.add_argument('--train_list', type=str)
    train_cmd.add_argument('--val_list', type=str)
    train_cmd.add_argument('--continue_from', type=str, default=None)
    train_cmd.add_argument('--epochs', type=int)
    train_cmd.add_argument('--save_interval', type=int)
    train_cmd.add_argument('--lr', '--learning_rate', type=float)
    
    # test
    test_cmd = mode.add_parser('test')
    test_cmd.add_argument('--test_path', type=str, default=None)
    test_cmd.add_argument('--test_list', type=str, default=None)
    test_cmd.add_argument('--model_path', type=str, default=None)
    test_cmd.add_argument('--fast_test', action='store_true')
    test_cmd.add_argument('--show_plot', action='store_true')
    test_cmd.add_argument('--filter_type', type=str, choices=['none', 'kf'], default='none', help='Post-processing filter')

    args, unknown_args = parser.parse_known_args()
    np.set_printoptions(formatter={'all': lambda x: '{:.6f}'.format(x)})

    args, kwargs = load_config(default_config_file, args, unknown_args)

    print(args, kwargs)
    if args.mode == 'train':
        train(args, **kwargs)
    elif args.mode == 'test':
        if not args.model_path:
            raise ValueError("Model path required")
        args.batch_size = 1
        test(args, **kwargs)

