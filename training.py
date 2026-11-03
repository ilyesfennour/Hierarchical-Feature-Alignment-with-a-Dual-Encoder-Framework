import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from training_utilities import calculate_psnr, compute_total_loss

def reconstruction_train(model, train_loader, val_loader, losses_config, device, exp_dir, num_epochs=100,
                         tap_layers=3, learning_rate=2e-3, checkpoint_dir=None, additional_epochs=0):
    """Train the model"""

    if checkpoint_dir:
        from checkpoint_utils import resume_training
        model, opt, scheduler, scaler, history, start_epoch, best_val_psnr, num_epochs = resume_training(checkpoint_dir, model, device, num_epochs, additional_epochs)
    else:
        # Initialize optimizer and losses
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, T_max=num_epochs)
    
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_psnr': [], 'val_psnr': [],
            'learning_rates': []
        }
        
        best_val_psnr = 0
        start_epoch = 0
        
        # Enable gradient scaler for mixed precision training
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        else:
            # Use a dummy scaler for CPU
            class DummyScaler:
                def scale(self, loss): return loss
                def step(self, optimizer): optimizer.step()
                def update(self): pass
                def unscale_(self, optimizer): pass
            scaler = DummyScaler()
    
    print(f"Starting training for {num_epochs} epochs...")
    
    for ep in range(start_epoch+1, num_epochs+1):
        # Training
        model.train()
        tot_loss, tot_psnr = 0, 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {ep}/{num_epochs} [Train]',
                         leave=False, dynamic_ncols=True)
        
        for b in train_pbar:
            deg = b['degraded'].to(device, non_blocking=True)
            clean = b['clean'].to(device, non_blocking=True)
            
            # Use mixed precision training
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    out, z_deg, enc_clean_feats, dec_feats = model(deg, clean)
                    
                    # Calculate enhancement losses
                    loss = compute_total_loss(
                        model=model,
                        losses_config=losses_config,
                        prediction=out,
                        target=clean,
                        z=z_deg,
                        enc_clean_feats=enc_clean_feats,
                        dec_feats=dec_feats,
                        tap_layers=tap_layers
                    )
            else:
                # CPU training
                out, z_deg, enc_clean_feats, dec_feats = model(deg, clean)
                
                # Calculate enhancement losses
                loss = compute_total_loss(
                    model=model,
                    losses_config=losses_config,
                    prediction=out,
                    target=clean,
                    z=z_deg,
                    enc_clean_feats=enc_clean_feats,
                    dec_feats=dec_feats,
                    tap_layers=tap_layers
                )
            
            # Backpropagation
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            
            tot_loss += loss.item()
            
            # Calculate PSNR
            with torch.no_grad():
                batch_psnr = 0
                for i in range(min(4, out.size(0))):
                    batch_psnr += calculate_psnr(out[i].cpu(), clean[i].cpu())
                tot_psnr += batch_psnr
                
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'PSNR': f'{batch_psnr/min(4, out.size(0)):.2f}dB'
                })
        
        tr_loss = tot_loss / len(train_loader)
        tr_psnr = tot_psnr / (len(train_loader) * 4)
        
        # Validation
        model.eval()
        v_loss, v_psnr = 0, 0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {ep}/{num_epochs} [Val]',
                      leave=False, dynamic_ncols=True)
        
        with torch.no_grad():
            for b in val_pbar:
                deg = b['degraded'].to(device, non_blocking=True)
                clean = b['clean'].to(device, non_blocking=True)
                
                # Forward pass
                out, z_deg, enc_clean_feats, dec_feats = model(deg, clean)
                
                # Calculate losses
                batch_loss = compute_total_loss(
                    model=model,
                    losses_config=losses_config,
                    prediction=out,
                    target=clean,
                    z=z_deg,
                    enc_clean_feats=enc_clean_feats,
                    dec_feats=dec_feats,
                    tap_layers=tap_layers
                )
                
                v_loss += batch_loss.item()
                
                # Calculate PSNR
                batch_psnr = 0
                for i in range(min(4, out.size(0))):
                    batch_psnr += calculate_psnr(out[i].cpu(), clean[i].cpu())
                v_psnr += batch_psnr
                
                val_pbar.set_postfix({
                    'loss': f'{batch_loss.item():.4f}',
                    'PSNR': f'{batch_psnr/min(4, out.size(0)):.2f}dB'
                })
        
        v_loss /= len(val_loader)
        v_psnr = v_psnr / (len(val_loader) * 4)
        
        # Update learning rate and track it
        current_lr = scheduler.get_last_lr()[0]
        history['learning_rates'].append(current_lr)
        scheduler.step()
        
        # Update history
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_psnr'].append(tr_psnr)
        history['val_psnr'].append(v_psnr)
        
        # Save checkpoint
        checkpoint = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if torch.cuda.is_available() else None,
            'history': history
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(exp_dir, "latest_checkpoint.pth"))
        
        # Save best model based on PSNR
        if v_psnr >= best_val_psnr:
            best_val_psnr = v_psnr
            torch.save(checkpoint, os.path.join(exp_dir, "best_model.pth"))
            print(f"💾 Saved new best model with PSNR: {v_psnr:.2f}dB")
        
        print(f"Epoch {ep}/{num_epochs} | LR={current_lr:.6f} | "
              f"Train L={tr_loss:.4f}, PSNR={tr_psnr:.2f} | "
              f"Val L={v_loss:.4f}, PSNR={v_psnr:.2f}")
    
    # Save final model and create plots
    torch.save(checkpoint, os.path.join(exp_dir, "final_model.pth"))
    
    return history

def classification_train(model, train_loader, val_loader, losses_config, device, exp_dir, num_epochs=100,
                         tap_layers=3, learning_rate=2e-3, checkpoint_dir=None, additional_epochs=0,
                         class_weights_dict=None, num_classes=None):

    if checkpoint_dir:
        from checkpoint_utils import resume_training
        model, opt, scheduler, scaler, history, start_epoch, best_val_acc, num_epochs = resume_training(checkpoint_dir, model, device, num_epochs, additional_epochs)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, T_max=num_epochs)

        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': [],
            'train_class_acc': {i: [] for i in range(num_classes)},
            'val_class_acc': {i: [] for i in range(num_classes)}
        }

        best_val_acc = 0
        start_epoch = 0
        
        # Enable gradient scaler for mixed precision training
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        else:
            # Use a dummy scaler for CPU
            class DummyScaler:
                def scale(self, loss): return loss
                def step(self, optimizer): optimizer.step()
                def update(self): pass
                def unscale_(self, optimizer): pass
            scaler = DummyScaler()

    if class_weights_dict is not None:
        weights_list = [class_weights_dict[i] for i in range(len(class_weights_dict))]
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
        print(f"Using class weights: {weights_tensor}")
        ce_loss = nn.CrossEntropyLoss(weight=weights_tensor)
    else:
        ce_loss = nn.CrossEntropyLoss()

    if 'center' in losses_config:
        from training_utilities import CenterLoss
        center_loss = CenterLoss(num_classes, 128).to(device)

    for ep in range(start_epoch+1, num_epochs+1):
        # Training
        model.train()
        tot_loss = 0
        correct = 0
        total = 0

        epoch_per_class_correct = {i: 0 for i in range(num_classes)}
        epoch_per_class_total = {i: 0 for i in range(num_classes)}

        train_pbar = tqdm(train_loader, desc=f'Epoch {ep}/{num_epochs} [Train]',
                          leave=False, dynamic_ncols=True)

        for b in train_pbar:

            deg = b['degraded'].to(device, non_blocking=True)
            clean = b['clean'].to(device, non_blocking=True)
            labels = b['label'].to(device, non_blocking=True)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    out, z, logits, enc_clean_feats, dec_feats = model(deg, clean)
                    loss = compute_total_loss(
                        model=model,
                        losses_config=losses_config,
                        prediction=out,
                        target=clean,
                        z=z,
                        labels=labels,
                        enc_clean_feats=enc_clean_feats,
                        dec_feats=dec_feats,
                        tap_layers=tap_layers
                    )
                    loss += losses_config['ce']['lambda'] * ce_loss(logits, labels)
                    if 'center' in losses_config:
                        loss += losses_config['center']['lambda'] * center_loss(z, labels)
            else:
                out, z, logits, enc_clean_feats, dec_feats = model(deg, clean)
                loss = compute_total_loss(
                    model=model,
                    losses_config=losses_config,
                    prediction=out,
                    target=clean,
                    z=z,
                    labels=labels,
                    enc_clean_feats=enc_clean_feats,
                    dec_feats=dec_feats,
                    tap_layers=tap_layers
                )
                loss += losses_config['ce']['lambda'] * ce_loss(logits, labels)
                if 'center' in losses_config:
                    loss += losses_config['center']['lambda'] * center_loss(z, labels)

            # Backpropagation
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            tot_loss += loss.item()

            with torch.no_grad():
                # Accuracy
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Per-class tracking
                for class_idx in range(num_classes):
                    mask = labels == class_idx
                    if mask.any():
                        class_correct = (predicted[mask] == labels[mask]).sum().item()
                        class_total = mask.sum().item()
                        epoch_per_class_correct[class_idx] += class_correct
                        epoch_per_class_total[class_idx] += class_total

                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'Acc': f'{(predicted == labels).float().mean().item():.3f}'
                })

        tr_loss = tot_loss / len(train_loader)
        tr_acc = correct / total if total > 0 else 0

        # Store per-class train accuracy
        for class_idx in range(num_classes):
            if epoch_per_class_total[class_idx] > 0:
                acc = epoch_per_class_correct[class_idx] / epoch_per_class_total[class_idx]
            else:
                acc = 0.0
            history['train_class_acc'][class_idx].append(acc)

        # Validation
        model.eval()
        v_loss = 0
        val_correct = 0
        val_total = 0

        val_per_class_correct = {i: 0 for i in range(num_classes)}
        val_per_class_total = {i: 0 for i in range(num_classes)}

        val_pbar = tqdm(val_loader, desc=f'Epoch {ep}/{num_epochs} [Val]',
                        leave=False, dynamic_ncols=True)

        with torch.no_grad():
            for b in val_pbar:

                deg = b['degraded'].to(device, non_blocking=True)
                clean = b['clean'].to(device, non_blocking=True)
                labels = b['label'].to(device, non_blocking=True)

                out, z, logits, enc_clean_feats, dec_feats = model(deg, clean)
                loss = compute_total_loss(
                    model=model,
                    losses_config=losses_config,
                    prediction=out,
                    target=clean,
                    z=z,
                    labels=labels,
                    enc_clean_feats=enc_clean_feats,
                    dec_feats=dec_feats,
                    tap_layers=tap_layers
                )
                loss += losses_config['ce']['lambda'] * ce_loss(logits, labels)
                if 'center' in losses_config:
                    loss += losses_config['center']['lambda'] * center_loss(z, labels)

                v_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

                for class_idx in range(num_classes):
                    mask = labels == class_idx
                    if mask.any():
                        class_correct = (predicted[mask] == labels[mask]).sum().item()
                        class_total = mask.sum().item()
                        val_per_class_correct[class_idx] += class_correct
                        val_per_class_total[class_idx] += class_total

                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'Acc': f'{(predicted == labels).float().mean().item():.3f}'
                })

        v_loss /= len(val_loader)
        v_acc = val_correct / val_total if val_total > 0 else 0

        for class_idx in range(num_classes):
            if val_per_class_total[class_idx] > 0:
                acc = val_per_class_correct[class_idx] / val_per_class_total[class_idx]
            else:
                acc = 0.0
            history['val_class_acc'][class_idx].append(acc)

        # Scheduler
        current_lr = scheduler.get_last_lr()[0]
        history['learning_rates'].append(current_lr)
        scheduler.step()

        # Store history
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(v_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(v_acc)

        checkpoint = {
            'epoch': ep,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if torch.cuda.is_available() else None,
            'history': history
        }

        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(exp_dir, "latest_checkpoint.pth"))
        
        # Save best model
        if v_acc >= best_val_acc:
            best_val_acc = v_acc
            torch.save(checkpoint, os.path.join(exp_dir, "best_model.pth"))
            print(f"💾 Saved new best model with ACC: {v_acc:.3f}")

        train_pbar.close()
        val_pbar.close()
        print(f"Epoch {ep}/{num_epochs} | LR={current_lr:.6f} | "
              f"Train Loss={tr_loss:.4f}, Acc={tr_acc:.3f} | "
              f"Val Loss={v_loss:.4f}, Acc={v_acc:.3f}")

    torch.save(model.state_dict(), os.path.join(exp_dir, "final_model.pth"))

    return history
