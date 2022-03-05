from ablations.layer_importance.trainer import *

class FeatPooler(nn.Module):
    def __init__(self, is_transformer, pool_features=True):
        super().__init__()
        pool_fn = nn.AdaptiveAvgPool1d(1) if is_transformer else nn.AdaptiveAvgPool2d((1, 1))
        self.pooling_fn = pool_fn if pool_features else None
        self.is_transformer = is_transformer
        
    def forward(self, x, layer_name):
        if self.pooling_fn is not None and layer_name != 'fc':
            if self.is_transformer:
                x = x.transpose(1,-1).flatten(2)
            x = self.pooling_fn(x).flatten(1)
        return x


class KnnHookedModel(HookedModel):
    def __init__(self, model, layer_names, pool_features=True, cat_feature_types=True):
        super().__init__(model, layer_names)
        is_transformer = hasattr(transformers, self.model.backbone_type)        
        self.pooling_fn = FeatPooler(is_transformer, pool_features)
        self.cat_features = cat_feature_types

    def hook_fn(self, layer_name):
        def hook_app(_, __, output):
            
            self.outputs[layer_name] = {}
            if isinstance(self.model.backbone, VisionTransformer) and isinstance(output, tuple):
                self.outputs[layer_name]['all'] = output[0].clone().detach().cpu()
            elif isinstance(self.model.backbone, (SwinTransformer, FocalTransformer)) and "attn" in layer_name:
                _B, N, C = output.shape
                batched_out = output.clone().detach().view(self.batch_size, -1, N, C)
                self.outputs[layer_name]['all'] = batched_out.cpu()
            else:
                self.outputs[layer_name]['all'] = output.clone().detach().cpu()     
                
            if isinstance(self.model.backbone, VisionTransformer):
                if layer_name != "backbone.patch_embed" and "backbone" in layer_name:
                    self.outputs[layer_name]['cls'] = self.outputs[layer_name]['all'][:,0,:].clone()
                    self.outputs[layer_name]['patches'] = self.outputs[layer_name]['all'][:,1:,:].clone()                
             
            if 'patches' in self.outputs[layer_name]:
                self.outputs[layer_name]['patches'] = self.pooling_fn(self.outputs[layer_name]['patches'], layer_name)           
            if 'all' in self.outputs[layer_name]:
                if self.cat_features and 'patches' in self.outputs[layer_name] and 'cls' in self.outputs[layer_name]:
                    if self.pooling_fn.pooling_fn is not None:
                        assert self.outputs[layer_name]['patches'].shape == self.outputs[layer_name]['cls'].shape
                        self.outputs[layer_name]['all'] = torch.cat([self.outputs[layer_name]['cls'], 
                                                                 self.outputs[layer_name]['patches']], dim=-1)
                else:
                    self.outputs[layer_name]['all'] = self.pooling_fn(self.outputs[layer_name]['all'], layer_name)

        return hook_app
    
        
class LayerwiseKnnTester(LayerImportanceTester):
    def __init__(self, wraped_defs, pool_features=True, is_memory_efficient=False):
        super().__init__(wraped_defs) 
        self.wraped_defs = wraped_defs
        self.pool_features = pool_features
        self.is_memory_efficient = is_memory_efficient
        
    def test(self, dataloader=None, **kwargs):
        print(f"\n\033[32mRunning layerwise knn experiment with pool_features: {self.pool_features} and memory_efficient: {self.is_memory_efficient} \033[0m")
        if not self.is_rank0: return
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.set_models_precision(False)
        
        if dataloader == None:
            dataloader=self.testloader  
        
        # get named modules
        layer_names = self.get_target_modules()
        
        results_dir = os.path.join(self.save_dir, 'results', self.model_name)
        check_dir(results_dir)
        results = {_l:edict() for _l in layer_names}
        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes    
        multi_label = not dataloader.dataset.is_multiclass
        target_metric = dataloader.dataset.target_metric
                    
        knn_metric = {}
        if self.is_memory_efficient:
            for layer_name in layer_names:     
                # Defining intermediate layers and wrap model
                self.model = self.wraped_defs.init_model()
                self.model = KnnHookedModel(self.model, [layer_name], self.pool_features)    
                # load weights from checkpoint
                self.load_session()                           
                assert [layer_name] == self.model.layer_names
                self.build_feature_bank(is_memory_efficient=self.is_memory_efficient)
                iter_bar = tqdm(dataloader, desc=f"\033[92mTesting {layer_name}\033[0m", leave=True, total=len(dataloader))
                self.forward_layerwise(iter_bar, knn_metric, dataloader)                          
                        
        else:
            # Defining intermediate layers and wrap model
            self.model = KnnHookedModel(self.model, layer_names, self.pool_features)
            # load weights from checkpoint
            self.load_session()            
            # Get feature bank and define iterator
            self.build_feature_bank(is_memory_efficient=self.is_memory_efficient)
            iter_bar = tqdm(dataloader, desc='Testing', leave=True, total=len(dataloader))

            self.forward_layerwise(iter_bar, knn_metric, dataloader)

        test_metrics = {}
        for layer in knn_metric.keys():
            test_metrics[layer] = {}
            for fmode, metric_ in knn_metric[layer].items():
                test_metrics[layer][fmode] = metric_.get_value(use_dist=isinstance(dataloader,DS))
        
        self.model.train()
        self.set_models_precision(self.use_mixed_precision)

        splitted = self.model_name.split("-run_")
        if len(splitted) == 1:
            knn_mname = f"{splitted[0]}-layer_wise_knn"
        elif len(splitted) == 2:
            knn_mname = f"{splitted[0]}-layer_wise_knn-run_{splitted[1]}"
        else:
            raise ValueError(f"Model name {self.model_name} does not follow the naming convention")
        knn_results_dir = os.path.join(self.save_dir, 'results', knn_mname)
        knn_metrics_path = os.path.join(knn_results_dir, "metrics_results.json")

        check_dir(knn_results_dir)
        save_json(test_metrics, knn_metrics_path)
        print(f'Saved results to: {knn_metrics_path}')  
        
    def forward_layerwise(self, iter_bar, knn_metric, dataloader):
        self.model.eval()
        int_to_labels = dataloader.dataset.int_to_labels
        knn_nhood = dataloader.dataset.knn_nhood
        n_classes = dataloader.dataset.n_classes    
        multi_label = not dataloader.dataset.is_multiclass        
        with torch.no_grad():
            for images, labels in iter_bar: 
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   

                if is_ddp(self.model):
                    raise NotImplementedError("LayerWise inference does not support DDP")
                else:
                    outputs = self.model(images)

                # looping through the layer's outputs
                for layer, features in outputs.items():
                    if layer not in knn_metric:
                        knn_metric[layer] = {}
                    for fmode, feat in features.items():
                        if fmode not in knn_metric[layer]:
                            knn_metric[layer][fmode] = self.metric_fn(n_classes, int_to_labels, mode="knn_val")
                        feat = F.normalize(feat.to(self.device_id).flatten(1), dim=1)
                        pred_labels = self.knn_predict(feature = feat, 
                                                       feature_bank=self.feature_bank[layer][fmode].to(self.device_id), 
                                                       feature_labels= self.targets_bank.to(self.device_id), 
                                                       knn_k=knn_nhood, knn_t=0.1, classes=n_classes,
                                                       multi_label = multi_label)
                        knn_metric[layer][fmode].add_preds(pred_labels, labels, using_knn=True)         
        
        
    def build_feature_bank(self, dataloader=None, **kwargs):
        """Build feature bank function.
        
        This function is meant to store the feature representation of the training images along with their respective labels 

        This is pretty much the same thing with global_step() but with torch.no_grad()
        Also note that DDP is not used here. There is not much point to DDP, since 
        we are not doing backprop anyway.
        """
        
        self.model.eval()
        if dataloader is None:
            dataloader = self.fbank_loader         
        
        n_classes = dataloader.dataset.n_classes
        if self.is_rank0:
            iter_bar = tqdm(dataloader, desc='Building Feature Bank', leave=False, total=len(dataloader))
        else:
            iter_bar = dataloader
            
        layer_names = self.model.layer_names
        
        self.feature_bank = {}
        self.targets_bank = []
        with torch.no_grad():  
            for images, labels in iter_bar:
                if len(labels) == 2 and isinstance(labels, list):
                    ids    = labels[1]
                    labels = labels[0]
                labels = labels.to(self.device_id, non_blocking=True)
                images = images.to(self.device_id, non_blocking=True)                   
                
                if is_ddp(self.model):
                    raise NotImplementedError("LayerWise inference does not support DDP")
                else:
                    outputs = self.model(images)
                self.targets_bank.append(labels.cpu())

                for layer, features in outputs.items():
                    if layer not in self.feature_bank:
                        self.feature_bank[layer] = {}
                    for fmode, feat in features.items():
                        if fmode not in self.feature_bank[layer]:
                            self.feature_bank[layer][fmode] = []
                        feat = F.normalize(feat.flatten(1), dim=1)
                        self.feature_bank[layer][fmode].append(feat)

            self.targets_bank = torch.cat(self.targets_bank, dim=0).t().contiguous()       
            for layer in self.feature_bank.keys():
                for fmode, featlist in self.feature_bank[layer].items():
                    self.feature_bank[layer][fmode] = torch.cat(featlist, dim=0).t().contiguous()

            synchronize()
            if is_ddp(self.model):
                raise NotImplementedError("LayerWise inference does not support DDP")            
                self.feature_bank = dist_gather(self.feature_bank, cat_dim=-1)
                self.targets_bank = dist_gather(self.targets_bank, cat_dim=-1)                

        self.model.train()
        
    def load_session(self, model_path=None):
        self.get_saved_model_path(model_path=model_path)
        if os.path.isfile(self.model_path):        
            print("Loading model from {}".format(self.model_path))
            checkpoint = torch.load(self.model_path)
            if is_parallel(self.model):
                self.model.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(self.device_id)
            self.org_model_state = model_to_CPU_state(self.model)
            self.best_model = deepcopy(self.org_model_state)
            if self.scaler is not None and "scaler" in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])            

        else:
            raise ValueError("=> no checkpoint found at '{}'".format(self.model_path))        
        