import torch
from nas_supernet import EdgeSupernet, NASChoiceBlock

# 1. Initialize the Supernet
supernet = EdgeSupernet(num_classes=100)
supernet.eval()

# 2. Load the trained weights
checkpoint = torch.load("./results/nas-tfd/nas-supernet-test/checkpoint.pth.tar", map_location='cpu')
clean_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['snet'].items()}
supernet.load_state_dict(clean_state_dict)

# 3. Lock the routing to your winning genome
genome = [3, 1, 0, 1, 0, 3]
blocks = [m for m in supernet.modules() if isinstance(m, NASChoiceBlock)]
for block, choice in zip(blocks, genome):
    block.active_choice = choice

# 4. Trace and export the active graph
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    supernet, 
    dummy_input, 
    "optimal_edge_net.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['intermediate_features', 'feat', 'classification_output']
)

print("Successfully compiled optimal_edge_net.onnx")