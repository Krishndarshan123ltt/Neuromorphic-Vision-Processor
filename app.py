import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results', exist_ok=True)

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Using device: {device}")

BATCH_SIZE  = 128
NUM_STEPS   = 25
BETA        = 0.95
LR          = 1e-3
NUM_CLASSES = 10
NUM_INPUTS  = 784
NUM_HIDDEN  = 1000


def load_mnist(batch_size=BATCH_SIZE):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])
    train_set = datasets.MNIST(
        root='./data', train=True,
        download=True, transform=transform)
    test_set = datasets.MNIST(
        root='./data', train=False,
        download=True, transform=transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=0)
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=0)
    print(f"📦 MNIST loaded: "
          f"Train={len(train_set)} "
          f"Test={len(test_set)}")
    return train_loader, test_loader


class NeuromorphicSNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.fc1   = nn.Linear(NUM_INPUTS, NUM_HIDDEN)
        self.lif1  = snn.Leaky(
            beta=BETA,
            spike_grad=spike_grad,
            init_hidden=False)
        self.fc2   = nn.Linear(NUM_HIDDEN, NUM_CLASSES)
        self.lif2  = snn.Leaky(
            beta=BETA,
            spike_grad=spike_grad,
            init_hidden=False)

    def forward(self, x):
        mem1    = self.lif1.init_leaky()
        mem2    = self.lif2.init_leaky()
        spk_rec = []
        mem_rec = []
        x = x.view(x.size(0), -1)
        for _ in range(NUM_STEPS):
            cur1       = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2       = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec.append(spk2)
            mem_rec.append(mem2)
        return (torch.stack(spk_rec),
                torch.stack(mem_rec))


def train_model(model, train_loader,
                test_loader, epochs=10):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR)
    loss_fn   = SF.ce_rate_loss()
    best_acc  = 0.0
    train_history = []
    test_history  = []

    print(f"\n{'='*50}")
    print(f"  🧠 SNN TRAINING — {epochs} epochs")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        correct = 0
        total   = 0

        for batch_idx, (data, targets) in enumerate(
                train_loader):
            data    = data.to(device)
            targets = targets.to(device)
            spk_out, _ = model(data)
            loss       = loss_fn(spk_out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predicted = spk_out.sum(0).argmax(1)
            correct  += (predicted == targets
                         ).sum().item()
            total    += targets.size(0)
            if (batch_idx + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Step {batch_idx+1} | "
                      f"Acc: {correct/total*100:.1f}%")

        train_acc = correct / total * 100
        test_acc  = evaluate_model(model, test_loader)
        train_history.append(train_acc)
        test_history.append(test_acc)

        print(f"\n  ✅ Epoch {epoch+1} | "
              f"Train: {train_acc:.1f}% | "
              f"Test: {test_acc:.1f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(),
                       'results/best_weights.pth')
            print(f"  💾 Best: {best_acc:.1f}% saved!")

    print(f"\n🏆 Training done! Best: {best_acc:.1f}%")

    with open('results/last_accuracy.txt', 'w') as f:
        f.write(str(round(best_acc, 1)))

    fig, ax = plt.subplots(
        figsize=(8, 4), facecolor='#1a1a2e')
    ax.set_facecolor('#0d0d1f')
    ax.plot(range(1, epochs+1), train_history,
            'o-', color='#2196F3', label='Train')
    ax.plot(range(1, epochs+1), test_history,
            's-', color='#4CAF50', label='Test')
    ax.axhline(92, color='#FF9800',
               linestyle='--', label='92% target')
    ax.set_xlabel('Epoch', color='white')
    ax.set_ylabel('Accuracy (%)', color='white')
    ax.set_title('Training Curve', color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e',
              labelcolor='white')
    plt.tight_layout()
    plt.savefig('results/training_curve.png',
                dpi=120, facecolor='#1a1a2e')
    plt.close()
    print("📊 Training curve saved!")
    return best_acc


def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for data, targets in loader:
            data    = data.to(device)
            targets = targets.to(device)
            spk_out, _ = model(data)
            predicted  = spk_out.sum(0).argmax(1)
            correct   += (predicted == targets
                          ).sum().item()
            total     += targets.size(0)
    return correct / total * 100


def predict_image(model, image):
    model.eval()
    if image.ndim == 2:
        img_t = torch.tensor(
            image,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)
    else:
        img_t = torch.tensor(
            image,
            dtype=torch.float32).unsqueeze(0)
    if img_t.shape[-1] != 28:
        img_t = torch.nn.functional.interpolate(
            img_t, size=(28, 28))
    img_t = img_t.to(device)
    with torch.no_grad():
        spk_out, mem_out = model(img_t)
    spike_counts = spk_out.sum(0).squeeze(
        ).cpu().numpy()
    total      = spike_counts.sum()
    confidence = (spike_counts / total
                  if total > 0 else np.zeros(10))
    prediction = int(np.argmax(spike_counts))
    return {
        'prediction':      prediction,
        'spike_counts':    spike_counts,
        'confidence':      confidence,
        'spike_recording': spk_out.squeeze(
            1).cpu().numpy(),
        'mem_trace':       mem_out.squeeze(
            1).cpu().numpy(),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train',
                   action='store_true')
    p.add_argument('--epochs',
                   type=int, default=10)
    p.add_argument('--weights', type=str,
                   default='results/best_weights.pth')
    return p.parse_args()


def main():
    args = parse_args()
    print("\n" + "="*50)
    print("  🧠 Neuromorphic Vision — snnTorch")
    print("="*50)
    model = NeuromorphicSNN().to(device)
    print(f"✅ Model: 784→1000 LIF→10 LIF")
    train_loader, test_loader = load_mnist()
    if args.train:
        train_model(model, train_loader,
                    test_loader,
                    epochs=args.epochs)
    if os.path.exists(args.weights):
        model.load_state_dict(
            torch.load(args.weights,
                       map_location=device))
        print(f"✅ Weights loaded!")
    acc = evaluate_model(model, test_loader)
    print(f"\n🎯 Final Accuracy: {acc:.1f}%")
    with open('results/last_accuracy.txt', 'w') as f:
        f.write(str(round(acc, 1)))
    print("✅ Done!")


if __name__ == '__main__':
    main()
