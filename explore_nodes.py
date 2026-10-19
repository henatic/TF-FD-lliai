import torch
from torchvision.models.feature_extraction import get_graph_node_names

def main():
    # Define the hub repo and model name you want to inspect
    hub_repo = 'pytorch/vision:v0.10.0'
    model_name = 'resnet18'
    
    print(f"Loading {model_name} from {hub_repo}...")
    net = torch.hub.load(hub_repo, model_name, pretrained=True)
    
    # Extract and print the nodes
    train_nodes, eval_nodes = get_graph_node_names(net)
    
    print("\n--- Available Nodes ---")
    for node in train_nodes:
        print(node)

if __name__ == '__main__':
    main()