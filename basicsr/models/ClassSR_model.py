from basicsr.metrics.psnr_ssim import calculate_psnr
import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
import math
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import cv2
import numpy as np
from PIL import Image

@MODEL_REGISTRY.register()
class ClassSR_Model(BaseModel):
    """ClassSR model for single image super-resolution."""

    def __init__(self, opt):
        super(ClassSR_Model, self).__init__(opt)

         # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.patch_size = int(opt["patch_size"])
        self.step = int(opt["step"])
        self.scale = int(opt["scale"])
        self.which_model = opt['network_g']['type']

        # load pretrained models
        load_path_g = self.opt['path'].get('pretrain_network_g', None)
        load_path_classifier = self.opt['path']['pretrain_network_classifier']
        load_path_G_branch3 = self.opt['path']['pretrain_network_G_branch3']
        load_path_G_branch2= self.opt['path']['pretrain_network_G_branch2']
        load_path_G_branch1 = self.opt['path']['pretrain_network_G_branch1']
        load_path_Gs=[load_path_G_branch1,load_path_G_branch2,load_path_G_branch3]
        if load_path_g is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path_g, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_classifier is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network_classifier(self.net_g, load_path_classifier, self.opt['path'].get('strict_load_g', True), param_key)
        if load_path_G_branch3 is not None and load_path_G_branch1 is not None and load_path_G_branch2 is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network_classSR_3class(self.net_g, load_path_Gs, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.pf = self.opt['logger']['print_freq']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path_g = self.opt['path'].get('pretrain_network_g', None)
            load_path_classifier = self.opt['path']['pretrain_network_classifier']
            load_path_G_branch3 = self.opt['path']['pretrain_network_G_branch3']
            load_path_G_branch2 = self.opt['path']['pretrain_network_G_branch2']
            load_path_G_branch1 = self.opt['path']['pretrain_network_G_branch1']
            load_path_Gs=[load_path_G_branch1,load_path_G_branch2,load_path_G_branch3]
            if load_path_g is not None:
                self.load_network(self.net_g_ema, load_path_g, self.opt['path'].get('strict_load_g', True), 'params_ema')
            if load_path_classifier is not None:
                self.load_network_classifier(self.net_g_ema, load_path_classifier, self.opt['path'].get('strict_load_g', True), 'params_ema')
            if load_path_G_branch3 is not None and load_path_G_branch1 is not None and load_path_G_branch2 is not None:
                self.load_network_classSR_3class(self.net_g_ema, load_path_Gs, self.opt['path'].get('strict_load_g', True), 'params_ema')
            if load_path_g == None:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

         # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('class_opt'):
            self.cri_class = build_loss(train_opt['class_opt']).to(self.device)

        if train_opt.get('average_opt'):
            self.cri_average = build_loss(train_opt['average_opt']).to(self.device)

        if self.cri_pix is None and self.cri_perceptual is None and self.cri_class is None and self.cri_average is None:
            raise ValueError('Both pixel and perceptual and class and average losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        if self.opt['fix_SR_module']:
            for k, v in self.net_g.named_parameters():  # can optimize for a part of the model
                if v.grad is None:
                    print(v)
                if v.requires_grad and "class" not in k:
                    v.requires_grad=False
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.lq_path = data['lq_path'][0]
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
            self.gt_path = data['gt_path'][0]


    def optimize_parameters(self, step):
        #这个根据ClassSR重新写的
        self.optimizer_g.zero_grad()
        self.output, self.type = self.net_g(self.lq, self.is_train)
        #print(self.type)
        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        
        # class loss
        if self.cri_class:
            class_loss = self.cri_class(self.type)
            l_total += class_loss
            loss_dict['class_loss'] = class_loss

        # average loss
        if self.cri_average:
            average_loss=self.cri_average(self.type)
            l_total += average_loss
            loss_dict['average_loss'] = average_loss

        if step % self.pf == 0:
           self.print_res(self.type)

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        self.net_g.eval()
        self.lq = cv2.imread(self.lq_path, cv2.IMREAD_UNCHANGED)
        self.gt = cv2.imread(self.gt_path, cv2.IMREAD_UNCHANGED)
        lr_list, num_h, num_w, h, w = self.crop_cpu(self.lq, self.patch_size, self.step)
        gt_list=self.crop_cpu(self.gt,self.patch_size*4,self.step*4)[0]
        sr_list = []
        index = 0

        psnr_type1 = 0
        psnr_type2 = 0
        psnr_type3 = 0

        for LR_img,GT_img in zip(lr_list,gt_list):
            if self.which_model=='classSR_3class_rcan':
                img = LR_img.astype(np.float32)
            else:
                img = LR_img.astype(np.float32) / 255.
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
            # some images have 4 channels
            if img.shape[2] > 3:
                img = img[:, :, :3]
            img = img[:, :, [2, 1, 0]]
            img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()[None, ...].to(
            self.device)
            with torch.no_grad():
                srt, type = self.net_g(img, False)

            if self.which_model == 'classSR_3class_rcan':
                sr_img = tensor2img(srt.detach()[0].float().cpu(), out_type=np.uint8, min_max=(0, 255))
            else:
                sr_img = tensor2img(srt.detach()[0].float().cpu())
            sr_list.append(sr_img)

            if index == 0:
                type_res = type
            else:
                type_res = torch.cat((type_res, type), 0)

            psnr= self.calculate_psnr(sr_img, GT_img)
            flag=torch.max(type, 1)[1].data.squeeze()
            if flag == 0:
                psnr_type1 += psnr
            if flag == 1:
                psnr_type2 += psnr\


            if flag == 2:
                psnr_type3 += psnr

            index += 1

        self.fake_H = self.combine(sr_list, num_h, num_w, h, w, self.patch_size, self.step)
        if self.opt['add_mask']:
            self.fake_H_mask = self.combine_addmask(sr_list, num_h, num_w, h, w, self.patch_size, self.step,type_res)
        self.gt = self.gt[0:h * self.scale, 0:w * self.scale, :]
        self.num_res = self.print_res(type_res)
        self.psnr_res=[psnr_type1,psnr_type2,psnr_type3]

        self.net_g.train()

    def crop_cpu(self,img,crop_sz,step):
        n_channels = len(img.shape)
        if n_channels == 2:
            h, w = img.shape
        elif n_channels == 3:
            h, w, c = img.shape
        else:
            raise ValueError('Wrong image shape - {}'.format(n_channels))
        h_space = np.arange(0, h - crop_sz + 1, step)
        w_space = np.arange(0, w - crop_sz + 1, step)
        index = 0
        num_h = 0
        lr_list=[]
        for x in h_space:
            num_h += 1
            num_w = 0
            for y in w_space:
                num_w += 1
                index += 1
                if n_channels == 2:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz]
                else:
                    crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
                lr_list.append(crop_img)
        h=x + crop_sz
        w=y + crop_sz
        return lr_list,num_h, num_w,h,w

    def combine(self,sr_list,num_h, num_w,h,w,patch_size,step):
        index=0
        sr_img = np.zeros((h*self.scale, w*self.scale, 3), 'float32')
        for i in range(num_h):
            for j in range(num_w):
                sr_img[i*step*self.scale:i*step*self.scale+patch_size*self.scale,j*step*self.scale:j*step*self.scale+patch_size*self.scale,:]+=sr_list[index]
                index+=1
        sr_img=sr_img.astype('float32')

        for j in range(1,num_w):
            sr_img[:,j*step*self.scale:j*step*self.scale+(patch_size-step)*self.scale,:]/=2

        for i in range(1,num_h):
            sr_img[i*step*self.scale:i*step*self.scale+(patch_size-step)*self.scale,:,:]/=2
        return sr_img

    def combine_addmask(self, sr_list, num_h, num_w, h, w, patch_size, step, type):
        index = 0
        sr_img = np.zeros((h * self.scale, w * self.scale, 3), 'float32')

        for i in range(num_h):
            for j in range(num_w):
                sr_img[i * step * self.scale:i * step * self.scale + patch_size * self.scale,
                j * step * self.scale:j * step * self.scale + patch_size * self.scale, :] += sr_list[index]
                index += 1
        sr_img = sr_img.astype('float32')

        for j in range(1, num_w):
            sr_img[:, j * step * self.scale:j * step * self.scale + (patch_size - step) * self.scale, :] /= 2

        for i in range(1, num_h):
            sr_img[i * step * self.scale:i * step * self.scale + (patch_size - step) * self.scale, :, :] /= 2

        index2 = 0
        for i in range(num_h):
            for j in range(num_w):
                # add_mask
                alpha = 1
                beta = 0.2
                gamma = 0
                bbox1 = [j * step * self.scale + 8, i * step * self.scale + 8,
                         j * step * self.scale + patch_size * self.scale - 9,
                         i * step * self.scale + patch_size * self.scale - 9]  # xl,yl,xr,yr
                zeros1 = np.zeros((sr_img.shape), 'float32')

                if torch.max(type, 1)[1].data.squeeze()[index2] == 0:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                      color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1),
                                         color=(0, 255, 0), thickness=-1)# simple green
                elif torch.max(type, 1)[1].data.squeeze()[index2] == 1:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                       color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1),
                                          color=(0, 255, 255), thickness=-1)# medium yellow
                elif torch.max(type, 1)[1].data.squeeze()[index2] == 2:
                    # mask1 = cv2.rectangle(zeros1, (bbox1[0], bbox1[1]), (bbox1[2], bbox1[3]),
                    #                       color=(0, 0, 0), thickness=1)
                    mask2 = cv2.rectangle(zeros1, (bbox1[0]+1, bbox1[1]+1), (bbox1[2]-1, bbox1[3]-1),
                                          color=(0, 0, 255), thickness=-1)# hard red

                sr_img = cv2.addWeighted(sr_img, alpha, mask2, beta, gamma)
                # sr_img = cv2.addWeighted(sr_img, alpha, mask1, 1, gamma)
                index2+=1
        return sr_img

    def print_res(self, type_res):
        #计算test的时候经过每个分支的数目，这个地方是为了统计flops
        num0 = 0
        num1 = 0
        num2 = 0

        for i in torch.max(type_res, 1)[1].data.squeeze():
            if i == 0:
                num0 += 1
            if i == 1:
                num1 += 1
            if i == 2:
                num2 += 1

        return [num0, num1,num2]

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations
        # modified from https://github.com/thstkdgus35/EDSR-PyTorch

        def _transform(v, op):
            # if self.precision != 'single': v = v.float()
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        num_ress = [0, 0, 0]
        psnr_ress=[0, 0, 0]

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            # print('type of self.lq after feed_data:',type(self.lq)) --->class 'torch.Tensor'
            self.test()

            visuals = self.get_current_visuals()
            # print('type of self.fake_H', type(visuals['result'])) --->class 'numpy.ndarray'
            # print('type of self.lq', type(visuals['lq']))

            # sr_img = Image.fromarray(np.uint8(visuals['result']))
            # print('sr_img_value:', np.uint8(visuals['result'])[:,2])
            # 这里print出来的值都是两百多
            sr_img = np.uint8(visuals['result'])
            # metric_data['img'] = sr_img
            if 'gt' in visuals:
                # gt_img = Image.fromarray(np.uint8(visuals['gt']))
                gt_img = np.uint8(visuals['gt'])
                # metric_data['img2'] = gt_img
                del self.gt

            num_res = visuals['num_res']
            psnr_res = visuals['psnr_res']
            num_ress[0] += num_res[0]
            num_ress[1] += num_res[1]
            num_ress[2] += num_res[2]

            psnr_ress[0] += psnr_res[0]
            psnr_ress[1] += psnr_res[1]
            psnr_ress[2] += psnr_res[2]
            # tentative for out of GPU memory
            del self.lq
            del self.fake_H
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                sr_img, gt_img = self.crop_border([sr_img, gt_img], self.opt['scale'])
                metric_data['img'] = sr_img
                metric_data['img2'] = gt_img
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()
        if num_ress[0]==0:
            num_ress[0]=1
        if num_ress[1]==0:
            num_ress[1]=1
        if num_ress[2]==0:
            num_ress[2]=1

        # add tensorboard logger
        if tb_logger:
            tb_logger.add_scalar('class1_num', num_ress[0], current_iter)
            tb_logger.add_scalar('class2_num', num_ress[1], current_iter)
            tb_logger.add_scalar('class3_num', num_ress[2], current_iter)

            tb_logger.add_scalar('Class1_PSNR', psnr_ress[0]/num_ress[0], current_iter)
            tb_logger.add_scalar('Class2_PSNR', psnr_ress[1]/num_ress[1], current_iter)
            tb_logger.add_scalar('class3_PSNR', psnr_ress[2]/num_ress[2], current_iter)
            
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)
            

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq
        # out_dict['lq'] = self.lq.detach().cpu()
        out_dict['num_res'] = self.num_res
        out_dict['psnr_res']=self.psnr_res
        out_dict['result'] = self.fake_H
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt
        if self.opt['add_mask']:
            out_dict['rlt_mask']=self.fake_H_mask

        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def calculate_psnr(self, img1, img2):
        # img1 and img2 have range [0, 255]
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float(80)
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def crop_border(self, img_list, crop_border):
        """Crop borders of images
        Args:
            img_list (list [Numpy]): HWC
            crop_border (int): crop border for each end of height and weight

        Returns:
            (list [Numpy]): cropped image list
        """

        if crop_border == 0:
            return img_list
        else:
            return [v[crop_border:-crop_border, crop_border:-crop_border] for v in img_list]
