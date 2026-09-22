import torch
from network import define_tsnet

def test_dynamic_vgg19():
    print("Initializing VGG19_Supernet...")
    # Initialize the network without DataParallel for cleaner local testing
    net = define_tsnet('vgg19', num_class=100, cuda=False)
    
    # Create a dummy batch of 2 CIFAR-100 images (32x32)
    dummy_input = torch.randn(2, 3, 32, 32)
    
    print("\n--- Testing Full Configuration (Baseline) ---")
    # This should default to the max_config defined in your class
    stem, rb1, rb2, rb3, feat, out = net(dummy_input)
    print(f"Stem output channels: {stem[1].shape[1]} (Expected 128)")
    print(f"RB1  output channels: {rb1[1].shape[1]}  (Expected 256)")
    print(f"RB2  output channels: {rb2[1].shape[1]}  (Expected 512)")
    print(f"RB3  output channels: {rb3[1].shape[1]}  (Expected 512)")
    print(f"Logits shape: {out.shape} (Expected [2, 100])")

    print("\n--- Testing Dynamic Sub-Network Configuration ---")
    # Define a custom, heavily pruned channel configuration simulating an RL controller output
    # Format matches your 16 conv layers
    pruned_config = [32, 32,  # Stem 1
                     64, 64,  # Stem 2
                     128, 128, 128, 128, # RB1
                     256, 256, 256, 256, # RB2
                     256, 256, 256, 256] # RB3
    
    stem_p, rb1_p, rb2_p, rb3_p, feat_p, out_p = net(dummy_input, channel_config=pruned_config)
    print(f"Stem output channels: {stem_p[1].shape[1]} (Expected 64)")
    print(f"RB1  output channels: {rb1_p[1].shape[1]}  (Expected 128)")
    print(f"RB2  output channels: {rb2_p[1].shape[1]}  (Expected 256)")
    print(f"RB3  output channels: {rb3_p[1].shape[1]}  (Expected 256)")
    print(f"Logits shape: {out_p.shape} (Expected [2, 100])")

if __name__ == '__main__':
    test_dynamic_vgg19()