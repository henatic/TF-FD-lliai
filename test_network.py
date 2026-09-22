import torch
from network import define_tsnet, DynamicConv2d

print("--- 1. Testing VGG19_KD ---")
net = define_tsnet("vgg19", num_class=100, cuda=torch.cuda.is_available())
dummy_input = torch.randn(2, 3, 32, 32)
if torch.cuda.is_available():
    dummy_input = dummy_input.cuda()

stem, rb1, rb2, rb3, feat, out = net(dummy_input)

print(f"Input shape:  {dummy_input.shape}")
print(f"Stem output:  {stem[1].shape}")
print(f"RB1 output:   {rb1[1].shape}")
print(f"RB2 output:   {rb2[1].shape}")
print(f"RB3 output:   {rb3[1].shape}")
print(f"Feat shape:   {feat.shape}")
print(f"Logits shape: {out.shape} (Expected: [2, 100])")

print("\n--- 2. Testing DynamicConv2d ---")
dyn_conv = DynamicConv2d(max_in_channels=64, max_out_channels=64, kernel_size=3, padding=1)
dyn_input = torch.randn(2, 32, 16, 16)  # 32 active input channels
dyn_out = dyn_conv(dyn_input, active_out_channels=48)  # request 48 active output channels
print(f"DynamicConv2d Output: {dyn_out.shape} (Expected: [2, 48, 16, 16])")