import torch
import random
import argparse
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as dst
from utils import AverageMeter, accuracy, count_parameters_in_MB
from nas_supernet import EdgeSupernet, NASChoiceBlock

class EvolutionarySearch:
    def __init__(self, supernet, val_loader, pop_size=50, max_epochs=20, max_param_mb=1.5):
        self.supernet = supernet
        self.val_loader = val_loader
        self.pop_size = pop_size
        self.max_epochs = max_epochs
        self.max_param_mb = max_param_mb # Strict edge-device hardware constraint
        
        # Discover the search space dimensions dynamically
        self.blocks = [m for m in supernet.modules() if isinstance(m, NASChoiceBlock)]
        self.num_blocks = len(self.blocks)
        self.num_choices = len(self.blocks[0].choices)
        
    def decode_genome(self, genome):
        """Applies a genome (e.g., [0, 3, 2, 1, 3, 2]) to the Supernet blocks."""
        for block, choice in zip(self.blocks, genome):
            block.active_choice = choice

    def evaluate_fitness(self, genome):
        """Calculates Top-1 Accuracy, returning 0.0 if the hardware budget is exceeded."""
        self.decode_genome(genome)
        
        # USE THE NEW ACTIVE PARAMETER COUNTER
        param_mb = self.count_active_params()
        if param_mb > self.max_param_mb:
            return 0.0, param_mb
            
        self.supernet.eval()
        top1 = AverageMeter()
        
        with torch.no_grad():
            for img, target in self.val_loader:
                img, target = img.cuda(), target.cuda()
                
                _, _, out = self.supernet(img)
                prec1, _ = accuracy(out, target, topk=(1, 5))
                top1.update(prec1.item(), img.size(0))
                
        return top1.avg, param_mb

    def crossover(self, parent1, parent2):
        """Randomly mixes genes from two high-performing parents."""
        child = []
        for g1, g2 in zip(parent1, parent2):
            child.append(g1 if random.random() > 0.5 else g2)
        return child

    def mutate(self, genome, mutation_rate=0.1):
        """Randomly alters individual genes to prevent local minima."""
        mutated = []
        for gene in genome:
            if random.random() < mutation_rate:
                mutated.append(random.randint(0, self.num_choices - 1))
            else:
                mutated.append(gene)
        return mutated

    def search(self):
        # 1. Initialize random population
        population = [[random.randint(0, self.num_choices - 1) for _ in range(self.num_blocks)] 
                      for _ in range(self.pop_size)]
        
        for epoch in range(self.max_epochs):
            print(f"\n--- Evolutionary Search Epoch {epoch+1}/{self.max_epochs} ---")
            
            # 2. Evaluate Fitness
            fitness_scores = []
            for genome in tqdm(population, desc="Evaluating Population"):
                acc, params = self.evaluate_fitness(genome)
                fitness_scores.append({'genome': genome, 'acc': acc, 'params': params})
                
            # Sort by accuracy (descending)
            fitness_scores.sort(key=lambda x: x['acc'], reverse=True)
            
            best_genome = fitness_scores[0]
            print(f"Best this epoch: Acc {best_genome['acc']:.2f}% | Params {best_genome['params']:.2f}MB | Genome {best_genome['genome']}")
            
            # 3. Elitism (Keep top 20% untouched)
            retain_length = int(self.pop_size * 0.2)
            parents = [x['genome'] for x in fitness_scores[:retain_length]]
            
            # 4. Breed next generation
            next_generation = parents.copy()
            while len(next_generation) < self.pop_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                next_generation.append(child)
                
            population = next_generation
            
        print("\n--- Search Complete ---")
        print(f"Optimal Sub-Network Genome: {fitness_scores[0]['genome']}")
        return fitness_scores[0]['genome']

    def count_active_params(self):
        """Counts parameters only for the active sub-network routing."""
        total_params = 0
        
        # 1. Count fixed components (stem and classifier)
        for param in self.supernet.stem.parameters():
            total_params += param.numel()
        for param in self.supernet.classifier.parameters():
            total_params += param.numel()
            
        # 2. Count ONLY the currently active choice in each NAS block
        for block in self.blocks:
            if block.active_choice != -1:
                active_module = block.choices[block.active_choice]
                for param in active_module.parameters():
                    total_params += param.numel()
                    
        return total_params / 1e6

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evolutionary Search for EdgeSupernet')
    parser.add_argument('--supernet_path', type=str, required=True, help='Path to trained Supernet weights')
    parser.add_argument('--img_root', type=str, default='./datasets', help='Path to CIFAR-100')
    parser.add_argument('--max_params', type=float, default=1.5, help='Hardware budget in MB')
    args = parser.parse_args()

    # Load Dataset
    mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    val_transform = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_loader = torch.utils.data.DataLoader(
        dst.CIFAR100(root=args.img_root, transform=val_transform, train=False, download=True),
        batch_size=256, shuffle=False, num_workers=4, pin_memory=True
    )

   # Initialize and Load Supernet
    supernet = EdgeSupernet(num_classes=100).cuda()
    checkpoint = torch.load(args.supernet_path)
    
    # Strip the 'module.' prefix added by DataParallel during training
    state_dict = checkpoint['snet']
    clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Load the cleaned dictionary
    supernet.load_state_dict(clean_state_dict)
    
    # Run Search
    searcher = EvolutionarySearch(supernet, val_loader, pop_size=40, max_epochs=15, max_param_mb=args.max_params)
    optimal_genome = searcher.search()