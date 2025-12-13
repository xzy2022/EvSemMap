import os
import torch
import torch.nn as nn  # [修改1] 引入 nn 模块用于 DataParallel
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

    # [修改2] 多卡并行包装逻辑 (核心修改)
    if torch.cuda.device_count() > 1:
        print(f"--> [System] Detected {torch.cuda.device_count()} GPUs. Enabling DataParallel training!")
        model = nn.DataParallel(model)

    total_iters = 0
    for epoch in range(epoch_start, args.n_epoch+1):
        model.train()
        epoch_iter = 0 

        # epoch_acc, epoch_ece = 0.0, 0.0
        for i, data in enumerate(trainloader, start=1):
            optimizer.zero_grad()

            img, lbl = data
            img, lbl = img.cuda(), lbl.cuda()

            if args.model == 'vanilla':
                # loss, acc, ece = model(img, lbl, with_acc_ece = True)
                loss = model(img, lbl, with_acc_ece = False)
            else:
                # loss, acc, ece = model(img, lbl, i, epoch, with_acc_ece = True)
                loss = model(img, lbl, i, epoch, with_acc_ece = False)
            # epoch_acc += acc
            # epoch_ece += ece
            
            # [修改说明] 原代码已经有的 loss = loss.mean() 在 DP 模式下非常关键且必须。
            # 单卡时 loss 是标量，mean() 不改变值；
            # 多卡时 loss 是向量 [loss_gpu0, loss_gpu1, ...]，必须 mean() 聚合后才能 backward。
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        
            total_iters += args.batch_size
            epoch_iter += args.batch_size

            if i % 10 == 0:
                print(f"[{args.remark}][EPOCH {epoch}, ({i} / {len(trainloader)}) loss {loss.item() :.4f}")

        # writer.add_scalar("Epoch_loss/acc", epoch_acc.item() / len(trainloader), epoch)
        # writer.add_scalar("Epoch_loss/ece", epoch_ece.item() / len(trainloader), epoch)
        writer.add_scalar("Epoch_loss/loss", loss.item(), epoch)
        scheduler.step()
    
        if epoch % args.save_freq == 0:
            print("Saving the model")
            # [修改3] 保存模型时的兼容性处理
            # 如果模型被 DataParallel 包装了，state_dict 里的 key 会带有 'module.' 前缀
            # 为了保证以后加载模型时（无论是单卡还是多卡）更加通用，建议剥离 .module 再保存
            if isinstance(model, nn.DataParallel):
                state_dict_to_save = model.module.state_dict()
            else:
                state_dict_to_save = model.state_dict()

            torch.save({ 'epoch': epoch, 'network' : state_dict_to_save }, os.path.join(model_dir, f"{epoch}.pth"))
        writer.flush()