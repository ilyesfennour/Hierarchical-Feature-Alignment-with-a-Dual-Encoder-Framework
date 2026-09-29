import os
import json
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

def get_parameters(exp_dir):
    """Resume training while preserving all previous history"""

    # Load config
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"🔄 Resuming training from: {exp_dir}")
    print(f"📊 Original config:")

    autoencoder             = config.get('model', 'normal')
    losses_config           = config.get('losses_config', None)
    dataset                 = config.get('dataset', 'MedMNIST')
    root_dir                = config.get('root_dir', None)
    degradation             = config.get('degradation', None)
    max_samples_per_class   = config.get('max_samples_per_class', None)
    epochs                  = config.get('num_epochs', 50)
    batch_size              = config.get('batch_size', 32)
    learning_rate           = config.get('learning_rate', 2e-3)
    workers                 = config.get('workers', 0)
    num_layers              = config.get('num_layers', 4)
    pretrained_path         = None

    return autoencoder, losses_config, dataset, root_dir, degradation, max_samples_per_class, epochs, batch_size, learning_rate, workers, num_layers, pretrained_path

def resume_training(exp_dir, model, device, num_epochs, additional_epochs):
    """Resume training exactly from where it stopped"""

    checkpoint_path = os.path.join(exp_dir, "latest_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    start_epoch = checkpoint['epoch']
    learning_rate = checkpoint['history']['learning_rates'][-1]
    num_epochs = num_epochs + additional_epochs

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, T_max=num_epochs)

    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if torch.cuda.is_available() and checkpoint['scaler_state_dict'] is not None:
        scaler = torch.cuda.amp.GradScaler()
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
        class DummyScaler:
            def scale(self, loss): return loss
            def step(self, optimizer): optimizer.step()
            def update(self): pass
            def unscale_(self, optimizer): pass
        scaler = DummyScaler()
    history = checkpoint['history']
    if 'val_psnr' in history:
        best_val = max(history['val_psnr'])
    elif 'val_acc' in history:
        best_val = max(history['val_acc'])
    
    print(f"✅ Resumed from epoch {start_epoch}")
    print(f"   Last val_loss: {history['val_loss'][-1]:.4f}")
    if 'val_psnr' in history:
        print(f"   Last val_psnr: {history['val_psnr'][-1]:.2f}dB")
    elif 'val_acc' in history:
        print(f"   Last val_acc:  {history['val_acc'][-1]:.3f}")
    print(f"   Current LR: {learning_rate:.6f}")
    print(f"🎯 Training for {num_epochs - start_epoch} more epochs")
    
    return model, opt, scheduler, scaler, history, start_epoch, best_val, num_epochs

def load_pretrained(model, pretrained_path, device):
    """Load pre-trained model and optionally optimizer/scheduler/history"""
    
    pretrained = torch.load(pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(pretrained['model_state_dict'])
    
    # Print info
    print(f"  - Pre-trained for {pretrained['epoch']} epochs")
    if 'val_psnr' in pretrained['history']:
        print(f"  - Original best PSNR: {max(pretrained['history']['val_psnr']):.2f}dB")
    elif 'val_acc' in pretrained['history']:
        print(f"  - Original best Acc:  {max(pretrained['history']['val_acc']):.3f}")
    print(f"  - Original best Loss: {max(pretrained['history']['val_loss']):.4f}")
    return model