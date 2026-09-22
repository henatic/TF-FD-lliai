def calculate_vgg_cost(channel_config, num_classes=100):
    """
    Calculates exact MACs (FLOPs proxy) and Parameter count for a dynamically 
    sliced VGG19 architecture on a 32x32 input.
    """
    # Spatial dimensions of feature maps after each pooling stage
    spatial_sizes = [
        32, 32,         # Block 1 (Stem 1)
        16, 16,         # Block 2 (Stem 2)
        8, 8, 8, 8,     # Block 3 (RB1)
        4, 4, 4, 4,     # Block 4 (RB2)
        2, 2, 2, 2      # Block 5 (RB3)
    ]
    
    # The input to layer i is the output of layer i-1 (Input image has 3 channels)
    in_channels = [3] + channel_config[:-1]
    
    total_flops = 0
    total_params = 0
    
    # 1. Tally Convolutional Layers
    for i in range(16):
        c_in = in_channels[i]
        c_out = channel_config[i]
        h_w = spatial_sizes[i]
        
        # Parameters = (C_in * Kernel_H * Kernel_W + Bias) * C_out
        layer_params = (c_in * 3 * 3 + 1) * c_out
        
        # FLOPs (MACs) = Parameters * Spatial_H * Spatial_W
        layer_flops = layer_params * h_w * h_w
        
        total_params += layer_params
        total_flops += layer_flops

    # 2. Tally Classifier (Linear Layers)
    # The final conv layer (conv16) feeds into the first linear layer
    fc1_params = (channel_config[-1] + 1) * 512
    fc2_params = (512 + 1) * num_classes
    
    total_params += (fc1_params + fc2_params)
    total_flops += (fc1_params + fc2_params) # Linear layer FLOPs == Params
    
    return total_flops, total_params

def compute_reward(val_loss, sampled_flops, target_flops, lambda_penalty=2.0):
    # Base reward is the negative loss (we want to maximize reward / minimize loss)
    reward = -val_loss
    
    # Apply penalty only if the sampled architecture exceeds the FLOP budget
    if sampled_flops > target_flops:
        penalty = lambda_penalty * ((sampled_flops / target_flops) - 1)
        reward -= penalty
        
    return reward