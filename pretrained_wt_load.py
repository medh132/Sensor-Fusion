import torch
import os
from model_temporal_kf import TCNSeqNetwork, TransformerSeqNetwork, HybridTCNTransformer


def load_tcn_pretrained_weights(network, checkpoint_path, network_type='tcn', strict=False):
    """
    Load pretrained TCN weights into network
    
    Args:
        network: The model to load weights into
        checkpoint_path: Path to pretrained checkpoint
        network_type: 'tcn' or 'hybrid'
        strict: If True, raise error on missing/unexpected keys
    
    Returns:
        network with loaded weights
    """
    import os
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return network
    
    print(f"Loading pretrained weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
    else:
        pretrained_dict = checkpoint
    
    if network_type == 'tcn':
        # Direct loading for TCN
        model_dict = network.state_dict()
        
        # Filter out keys that don't match in size
        pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(pretrained_dict)
        network.load_state_dict(model_dict, strict=strict)
        print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained TCN")
        
    elif network_type == 'hybrid':
        # Load weights into TCN branch of hybrid model
        model_dict = network.state_dict()
        
        # Map pretrained keys to hybrid model's TCN branch
        tcn_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            # Map to TCN branch in hybrid model
            if k.startswith('tcn.') or k.startswith('output_layer.'):
                # tcn.network.X.Y -> tcn.X.Y
                new_key = k.replace('net.network.', 'tcn.')
                
                # Also try mapping output_layer
                if k.startswith('output_layer.'):
                    new_key = k.replace('output_layer.', 'tcn_output.')
                else:
                    new_key = k
                
                # Check if key exists and shapes match
                if new_key in model_dict and v.shape == model_dict[new_key].shape:
                    tcn_pretrained_dict[new_key] = v
        
        if len(tcn_pretrained_dict) > 0:
            model_dict.update(tcn_pretrained_dict)
            network.load_state_dict(model_dict, strict=False)
            print(f"Loaded {len(tcn_pretrained_dict)} layers into TCN branch of hybrid model")
        else:
            print("Warning: No matching keys found for TCN branch. Trying alternative mapping...")
            # Alternative: try to load directly to see key structure
            print("Pretrained keys:", list(pretrained_dict.keys())[:5])
            print("Model keys (TCN):", [k for k in model_dict.keys() if 'tcn' in k][:5])
    
    return network


# Example usage in get_model function
def get_model(args, **kwargs):
    _input_channel = kwargs.get('input_channel', 6)
    _output_channel = kwargs.get('output_channel', 2)
    config = {}
    if kwargs.get('dropout'):
        config['dropout'] = kwargs.get('dropout')
    
    if args.type == 'tcn':
        network = TCNSeqNetwork(_input_channel, _output_channel, args.kernel_size,
                                layer_channels=args.channels, **config)
        print("TCN Network. Receptive field: {} ".format(network.get_receptive_field()))
        
        # Load pretrained weights if specified
        if hasattr(args, 'pretrained_tcn') and args.pretrained_tcn:
            network = load_tcn_pretrained_weights(network, args.pretrained_tcn, 
                                                 network_type='tcn')
    
    elif args.type == 'transformer':
        network = TransformerSeqNetwork(
            _input_channel, _output_channel,
            d_model=args.d_model if hasattr(args, 'd_model') else 128,
            nhead=args.nhead if hasattr(args, 'nhead') else 4,
            **config
        )
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
            combination=args.combination if hasattr(args, 'combination') else 'concat'
        )
        print("Hybrid TCN-Transformer Network. Combination: {}".format(
            args.combination if hasattr(args, 'combination') else 'concat'))
        
        # Load pretrained TCN weights if specified
        if hasattr(args, 'pretrained_tcn') and args.pretrained_tcn:
            network = load_tcn_pretrained_weights(network, args.pretrained_tcn, 
                                                 network_type='hybrid')
    
    else:
        raise ValueError(f"Unknown network type: {args.type}")
    
    pytorch_total_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Network constructed. trainable parameters: {}'.format(pytorch_total_params))
    return network


def load_tfm_pretrained_weights(network, checkpoint_path, network_type='hybrid', strict=False):
    """
    Load pretrained Transformer weights into network.

    Args:
        network: The model to load weights into
                - TransformerSeqNetwork when network_type in {'tfm', 'transformer'}
                - HybridTCNTransformer   when network_type == 'hybrid'
        checkpoint_path: Path to pretrained checkpoint
        network_type: 'tfm' / 'transformer' or 'hybrid'
        strict: If True, raise error on missing/unexpected keys

    Returns:
        network with loaded weights
    """       
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return network

    print(f"Loading pretrained transformer weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            pretrained_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            pretrained_dict = checkpoint['state_dict']
        else:
            pretrained_dict = checkpoint
    else:
        pretrained_dict = checkpoint

    # ------------------------------------------------------------------
    # Case 1: network IS the transformer (TransformerSeqNetwork)
    # ------------------------------------------------------------------
    if network_type == 'transformer':
        model_dict = network.state_dict()

        # Filter out keys that don't match in size
        filtered_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(filtered_dict)
        network.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(filtered_dict)}/{len(model_dict)} transformer layers into TransformerSeqNetwork")
        return network

    # ------------------------------------------------------------------
    # Case 2: network is HybridTCNTransformer, load into .transformer
    # ------------------------------------------------------------------
    if network_type == 'hybrid':
        model_dict = network.state_dict()
        tfm_pretrained_dict = {}

        for k, v in pretrained_dict.items():
            candidate_keys = []

            # If checkpoint is from a *hybrid* model, keys are already 'transformer.*'
            if k.startswith('transformer.'):
                candidate_keys.append(k)
            else:
                # If checkpoint is from a standalone TransformerSeqNetwork,
                # its keys are like 'input_projection.weight', so we prepend 'transformer.'
                candidate_keys.append(f"transformer.{k}")

            # Deduplicate while preserving order
            seen = set()
            uniq_candidates = []
            for ck in candidate_keys:
                if ck not in seen:
                    seen.add(ck)
                    uniq_candidates.append(ck)

            # Pick the first candidate that exists and has matching shape
            for ck in uniq_candidates:
                if ck in model_dict and model_dict[ck].shape == v.shape:
                    tfm_pretrained_dict[ck] = v
                    break

        if len(tfm_pretrained_dict) == 0:
            print("Warning: No matching transformer keys found for hybrid model.")
            print("Pretrained keys (first 5):", list(pretrained_dict.keys())[:5])
            print(
                "Model keys (transformer-related, first 10):",
                [k for k in model_dict.keys() if k.startswith('transformer.')][:10]
            )
            return network

        model_dict.update(tfm_pretrained_dict)
        # Use strict=False here so extra TCN / KF keys in the checkpoint don't cause issues
        network.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(tfm_pretrained_dict)} transformer layers into HybridTCNTransformer")
        return network

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    print(f"Warning: Unknown network_type '{network_type}'. No weights loaded.")
    return network
