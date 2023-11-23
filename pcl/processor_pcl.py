import logging
import os
import torch
import torch.nn.functional as F
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
from .utils import *
from .loss import ClusterMemoryAMP, CrossEntropyLabelSmooth


def train_pcl(cfg,
              model,
              train_loader,
              val_loader,
              cluster_loader,
              optimizer,
              scheduler,
              num_query,
              num_classes):
    
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("PCL")
    logger.info('start training')
    
    model.to(device)
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    xent = CrossEntropyLabelSmooth(num_classes)
    logger.info(f'smoothed cross entropy loss on {num_classes} classes.')

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    
    # training epochs
    for epoch in range(1, epochs+1):
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        # create memory bank
        image_features, gt_labels = extract_image_features(model, cluster_loader, use_amp=True)
        image_features = image_features.float()
        image_features = F.normalize(image_features, dim=1)
            
        num_classes = len(gt_labels.unique()) - 1 if -1 in gt_labels else len(gt_labels.unique())
        logger.info(f'Memory has {num_classes} classes.')
        
        train_loader.new_epoch()
        
        # CAP memory
        memory = ClusterMemoryAMP(momentum=cfg.MODEL.MEMORY_MOMENTUM, use_hard=True).to(device)
        memory.features = compute_cluster_centroids(image_features, gt_labels).to(device)
        logger.info('Create memory bank with shape = {}'.format(memory.features.shape))
        
        # train one iteration
        model.train()
        num_iters = len(train_loader)
        for n_iter in range(num_iters):
            img, target, target_cam, _ = train_loader.next()
            
            optimizer.zero_grad()
            
            img = img.to(device)
            target = target.to(device)
            target_cam = target_cam.to(device)
            
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
                
            with amp.autocast(enabled=True):
                feat, logit = model(img, cam_label=target_cam, view_label=target_view)
                loss = memory(feat, target) * cfg.MODEL.PCL_LOSS_WEIGHT
                if cfg.MODEL.ID_LOSS_WEIGHT > 0:
                    loss_id = xent(logit, target) * cfg.MODEL.ID_LOSS_WEIGHT
                else:
                    loss_id = 0
                loss = loss + loss_id

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, scheduler.get_lr()[0]))
        
        scheduler.step()
        logger.info("Epoch {} done.".format(epoch))
        
        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            
        if epoch % eval_period == 0:
            model.eval()
            for n_iter, (img, vid, camid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.MODEL.SIE_CAMERA:
                        camids = camids.to(device)
                    else: 
                        camids = None
                    if cfg.MODEL.SIE_VIEW:
                        target_view = target_view.to(device)
                    else: 
                        target_view = None
                    feat = model(img, cam_label=camids, view_label=target_view)
                    evaluator.update((feat, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()
    logger.info('Training done.')
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("PCL")
    logger.info("Enter inferencing")
    model.to(device)

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()


    model.eval()
    for n_iter, (img, pid, camid, _) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]