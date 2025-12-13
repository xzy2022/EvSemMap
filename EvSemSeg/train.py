import os
import torch
import torch.nn as nn # 引入 nn 模块用于 DataParallel
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler, Adam

def train(model_dir, dataset, model, epoch_start, writer, args):
    # [优化建议] num_workers=16 设置较高，在显存紧张或CPU较弱时可能导致共享内存溢出或卡顿，一般建议 4 或 8
    trainloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16, shuffle=True, drop_last=True)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    if args.model == 'evidential':
        model.set_max_iter(len(trainloader))
    
    model.cuda()
    
    # Setup optimizer, scheduler objects
    # [重要提示] 优化器初始化必须保留在 DataParallel 包装之前！
    # 原因：这里使用了 model.encoder.parameters()。如果先包装模型，model 会变成 DataParallel 对象，
    # 导致直接访问 model.encoder 报错（需要改成 model.module.encoder）。
    # 保持现在的顺序（先 model.cuda() -> 再 init optimizer -> 最后 DataParallel）是最安全的。
    optimizer = Adam([{'params' : model.encoder.parameters(), 'lr' : args.l_rate}], lr=args.l_rate, betas=(0.5, 0.999))
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # 2. 【新增】断点续训逻辑 (Resume Logic)
    # 如果 args.load 被指定，且我们处于训练模式，且文件存在，尝试加载优化器状态
    if args.load != '$NONE$' and os.path.exists(args.load):
        print(f"--> [Resume] Attempting to resume training from: {args.load}")
        try:
            checkpoint = torch.load(args.load)
            
            # (A) 恢复 Epoch
            if 'epoch' in checkpoint:
                epoch_start = checkpoint['epoch'] + 1
                print(f"    Resuming from Epoch: {epoch_start}")
            
            # (B) 恢复 优化器状态
            if 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("    Optimizer state loaded.")
            else:
                print("    [Warning] No optimizer state found in checkpoint. Starting fresh optimizer.")

            # (C) 恢复 调度器状态
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
                print("    Scheduler state loaded.")
            else:
                print("    [Warning] No scheduler state found. Starting fresh scheduler.")
                
        except Exception as e:
            print(f"    [Error] Failed to resume optimizer/scheduler: {e}")
            print("    Continuing with loaded weights but fresh optimizer.")

    # 3. 多卡并行包装 (必须在 load_state_dict 之后，防止 key 不匹配)
    if torch.cuda.device_count() > 1:
        print(f"--> [System] Detected {torch.cuda.device_count()} GPUs. Enabling DataParallel!")
        model = nn.DataParallel(model)

    print(f"--> Training Start from Epoch {epoch_start} to {args.n_epoch}")

    total_iters = 0
    for epoch in range(epoch_start, args.n_epoch+1):
        model.train()
        
        # 记录一下当前的学习率，方便在 TensorBoard 查看
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar("HyperParams/LearningRate", current_lr, epoch)

        for i, data in enumerate(trainloader, start=1):
            optimizer.zero_grad()

            img, lbl = data
            img, lbl = img.cuda(), lbl.cuda()

            if args.model == 'vanilla':
                loss = model(img, lbl, with_acc_ece = False)
            else:
                loss = model(img, lbl, i, epoch, with_acc_ece = False)
                
            loss = loss.mean() # Handle DataParallel
            loss.backward()
            optimizer.step()
        
            total_iters += args.batch_size

            if i % 10 == 0:
                print(f"[{args.remark}][EPOCH {epoch}][Iter {i}/{len(trainloader)}] Loss: {loss.item() :.4f} | LR: {current_lr:.6f}")

        writer.add_scalar("Epoch_loss/loss", loss.item(), epoch)
        scheduler.step()
    
        if epoch % args.save_freq == 0:
            print(f"Saving checkpoint to {model_dir}/{epoch}.pth")
            
            # 保存模型时的兼容性处理
            # 如果模型被 DataParallel 包装了，state_dict 里的 key 会带有 'module.' 前缀
            # 为了保证以后加载模型时（无论是单卡还是多卡）更加通用，建议剥离 .module 再保存
            if isinstance(model, nn.DataParallel):
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            # 4. 【修改】保存完整的训练状态
            save_dict = {
                'epoch': epoch,
                'network': model_state,                # 模型权重
                'optimizer': optimizer.state_dict(),   # 优化器状态 (动量等)
                'scheduler': scheduler.state_dict()    # 学习率调度器状态
            }

            torch.save(save_dict, os.path.join(model_dir, f"{epoch}.pth"))
        
        writer.flush()