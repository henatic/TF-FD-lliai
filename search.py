import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dst
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# Import your custom modules
from network import define_tsnet
from controller import NASController
from reward import calculate_vgg_cost, compute_reward

def get_nas_dataloaders(img_root='./datasets', batch_size=128):
    mean = (0.5071, 0.4865, 0.4409)
    std  = (0.2673, 0.2564, 0.2762)

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    full_train_dataset = dst.CIFAR100(root=img_root, train=True, download=True, transform=train_transform)
    
    # Partition 40k for supernet weights, 10k for controller policy
    generator = torch.Generator().manual_seed(42)
    supernet_data, controller_data = random_split(full_train_dataset, [40000, 10000], generator=generator)

    supernet_loader = DataLoader(supernet_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    controller_loader = DataLoader(controller_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return supernet_loader, controller_loader


def train_supernet(supernet, loader, optimizer, criterion, channel_choices, device):
    supernet.train()
    total_loss = 0.0
    
    for img, target in loader:
        img, target = img.to(device), target.to(device)
        
        # Fair sampling: pick a random architecture for this batch to train all weights evenly
        random_config = [random.choice(channel_choices) for _ in range(16)]
        
        optimizer.zero_grad()
        # VGG_Supernet returns (stem, rb1, rb2, rb3, feat, out)
        _, _, _, _, _, logits = supernet(img, channel_config=random_config)
        
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)


def train_controller(controller, supernet, loader, optimizer, criterion, channel_choices, target_flops, device):
    controller.train()
    supernet.eval() # Freeze supernet weights during controller update
    total_reward = 0.0
    
    for img, target in loader:
        img, target = img.to(device), target.to(device)
        optimizer.zero_grad()
        
        # 1. Controller samples an architecture
        action_indices, log_prob = controller()
        channel_config = [channel_choices[idx] for idx in action_indices]
        
        # 2. Evaluate hardware cost
        sampled_flops, _ = calculate_vgg_cost(channel_config, num_classes=100)
        
        # 3. Evaluate accuracy on validation batch (zero-shot)
        with torch.no_grad():
            _, _, _, _, _, logits = supernet(img, channel_config=channel_config)
            val_loss = criterion(logits, target).item()
            
        # 4. Compute reward and apply REINFORCE
        reward = compute_reward(val_loss, sampled_flops, target_flops, lambda_penalty=2.0)
        policy_loss = -log_prob * reward
        
        policy_loss.backward()
        optimizer.step()
        
        total_reward += reward
        
    return total_reward / len(loader), channel_config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channel_choices = [32, 64, 128, 256, 512]
    
    # Calculate baseline FLOPs dynamically
    max_config = [64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512, 512, 512, 512, 512]
    vgg_baseline_flops, _ = calculate_vgg_cost(max_config, num_classes=100)
    
    # Set the target to 10% of the full VGG19 compute cost
    target_flops = vgg_baseline_flops * 0.10 
    
    print(f"VGG19 Baseline FLOPs: {vgg_baseline_flops / 1e6:.1f}M")
    print(f"Search Target FLOPs: {target_flops / 1e6:.1f}M")
    
    print("Initializing NAS Pipeline...")
    supernet_loader, controller_loader = get_nas_dataloaders()
    
    supernet = define_tsnet('vgg19', num_class=100, cuda=torch.cuda.is_available())
    supernet_optimizer = optim.SGD(supernet.parameters(), lr=0.025, momentum=0.9, weight_decay=3e-4)
    
    controller = NASController(num_layers=16, num_choices=len(channel_choices)).to(device)
    controller_optimizer = optim.Adam(controller.parameters(), lr=0.003)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Setup CSV Logging
    log_file = "nas_search_log.csv"
    with open(log_file, "w") as f:
        f.write("Epoch,Supernet_Loss,Controller_Reward,Sampled_Config\n")
    
    epochs = 100
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Phase 1: Train Supernet Weights
        sup_loss = train_supernet(supernet, supernet_loader, supernet_optimizer, criterion, channel_choices, device)
        print(f"Supernet Loss: {sup_loss:.4f}")
        
        # Phase 2: Train Controller Policy
        avg_reward, last_config = train_controller(
            controller, supernet, controller_loader, controller_optimizer, 
            criterion, channel_choices, target_flops, device
        )
        print(f"Controller Reward: {avg_reward:.4f} | Latest Sample: {last_config}")
        
        # Append metrics to log file
        with open(log_file, "a") as f:
            # Wrap the config list in quotes so it occupies a single CSV column
            config_str = f'"{str(last_config)}"'
            f.write(f"{epoch+1},{sup_loss:.4f},{avg_reward:.4f},{config_str}\n")

if __name__ == '__main__':
    main()