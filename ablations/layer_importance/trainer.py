from defaults import *

def rand_weights_init(m, sampling_fn=sample_iid_gaussian):  
    if sampling_fn is None:
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) 
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(m.bias, -bound, bound)        
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5)) 
            if m.bias is not None:
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                torch.nn.init.uniform_(m.bias, -bound, bound)        
    else:
        tmp_state = m.state_dict()
        for name_, module_ in m.named_modules():
            if isinstance(module_, (nn.Linear,nn.Conv2d)):
                weight_name = f"{name_}{'.' if name_ else ''}weight"
                bias_name = f"{name_}{'.' if name_ else ''}bias"
                tmp_state[weight_name] = sampling_fn(tmp_state[weight_name])
                if bias_name in tmp_state:
                    tmp_state[bias_name] = sampling_fn(tmp_state[bias_name])        
        m.load_state_dict(tmp_state)
        
        
class LayerImportanceTester(Trainer):
    def __init__(self, wraped_defs, use_momentum=True, layer_init_mode='reinit'):
        super().__init__(wraped_defs)
        self.mode = layer_init_mode

    def load_session(self, model_path=None):
        self.get_saved_model_path(model_path=model_path)
        if os.path.isfile(self.model_path):        
            print("Loading model from {}".format(self.model_path))
            checkpoint = torch.load(self.model_path)

            self.model.load_state_dict(checkpoint['state_dict'])
            model_params = checkpoint["parameters"]["model_params"]
                
            if "original_state" in checkpoint:
                original_state = checkpoint['original_state']
            elif model_params["pretrained"] or model_params["wtst_init"]:
                temp_model =  Classifier(model_params)
                original_state = model_to_CPU_state(temp_model)
                print("\033[93m Original state not found!!! Using params to re-init:  \033[0m")
                print("\033[93m \tAssuming fixed ImageNet pretraining or fixed WT.  \033[0m")
                print("\033[93m \tNote that some laeyrs (e.g. the last one) might not be valid.  \033[0m")                
            else:
                raise RuntimeError("Layer importance using new random init is not valid")
                
            self.model.to(self.device_id)
            self.original_state = original_state
            self.model_state = checkpoint['state_dict']
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(self.model_path))  
            
            
    def get_target_modules(self):
        module_names = []

        # Loop through model's children
        for name, module in self.model.named_children():
            if name == "backbone":
                # Loop through backbone's children
                for bname, bmodule in self.model.backbone.named_children():
                    
                    # Named children's tree for ResNet
                    if isinstance(self.model.backbone, cnn_models.ResNet):
                        if bname == "conv1":
                            module_names.append(f"backbone.{bname}")
                        elif "layer" in bname:
                            # Loop through resnet block's children
                            for block, blockmod in bmodule.named_children():
                                module_names.append(f"backbone.{bname}.{block}")
                        else:
                            pass
                        
                    # Named children's tree for Inception
                    if isinstance(self.model.backbone, cnn_models.Inception3):
                        if "Conv2d_" in bname:
                            module_names.append(f"backbone.{bname}")
                        elif "Mixed_" in bname:
                            # add each inception block
                            module_names.append(f"backbone.{bname}")
                        else:
                            pass                        
                            
                    # Named children's tree for VisionTransformer 
                    elif isinstance(self.model.backbone, VisionTransformer):
                        if bname == "patch_embed":
                            module_names.append(f"backbone.{bname}")
                        elif bname == "blocks":
                            # Loop through transformer block's children
                            for block, blockmod in self.model.backbone.blocks.named_children():
                                # Loop through transformer internal block's children
                                for wname, wm in self.model.backbone.blocks[int(block)].named_children():
                                    if "attn" in wname or "mlp" in wname:
                                        module_names.append(f"backbone.{bname}.{block}.{wname}")
                                        
                    # Named children's tree for SwinTransformer and FocalTransformer
                    elif isinstance(self.model.backbone, (SwinTransformer, FocalTransformer)):
                        if bname == "patch_embed":
                            module_names.append(f"backbone.{bname}")
                        elif bname == 'layers':
                            # Loop through transformer block's children
                            for block, blockmod in self.model.backbone.layers.named_children():
                                # Loop through transformer internal block's children
                                for bid, bwm in self.model.backbone.layers[int(block)].blocks.named_children():
                                    # loop through each block
                                    for wname, wm in self.model.backbone.layers[int(block)].blocks[int(bid)].named_children():
                                        if "attn" in wname or "mlp" in wname:
                                            module_names.append(f"backbone.{bname}.{block}.blocks.{bid}.{wname}")
                                            
                    else:
                        raise NotImplementedError(f"Tree-based naming for {type(self.model.backbone)} is not implemented")

            elif name == "fc":
                module_names.append(name)
            else:
                raise RuntimeError(f"Unknown module {name} found in the core model")
                
        return module_names
    
    def replace_layer(self, target_layer, original_state=None):
        if self.mode == 'reinit':
            if original_state is None:
                raise RuntimeError("Re-initialization requires the weights of the init checkpoint")
                
            target_state = OrderedDict([(k, v) 
                                        for k, v in original_state.items() 
                                        if target_layer in k])
            dif_keys = self.model.load_state_dict(target_state, strict=False)
            check2 = set(dif_keys[0]) & set(target_state.keys())
            check1 = dif_keys[1]
            if (set(dif_keys[0]) & set(target_state.keys())) or dif_keys[1]:
                raise RuntimeError("State_dict keys error: Please make sure that all key-names match")
        elif self.mode == 'rerand':
            for name, layer in self.model.named_modules():
                if name == target_layer:
                    layer.apply(rand_weights_init)
        else:
            raise NotImplementedError(f"Expected modes [reinit, rerand] -- {self.mode} found")
        
    def test(self, dataloader=None, **kwargs):
        print(f"\n\033[32mStarting layer importance experiment with init_mode: {self.mode} \033[0m")
        if not self.is_rank0: return
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.set_models_precision(False)
                
        self.load_session()

        if dataloader == None:
            dataloader=self.testloader  
            
        results_dir = os.path.join(self.save_dir, 'results', self.model_name)
        check_dir(results_dir)
        test_metrics = {}
        
        for layer in self.get_target_modules():  
            
            # load default pretrained model
            model_state = deepcopy(self.model_state)
            self.model.load_state_dict(model_state)
            # load layer to be changed
            original_state = deepcopy(self.original_state)
            self.replace_layer(layer, original_state)
            print(f"\033[92m Evaluating importance of layer: {layer}  \033[0m")
            
            test_loss = []
            feature_bank = []
            results = edict()
            if self.log_embeddings:
                self.build_feature_bank()            
                
            knn_nhood = dataloader.dataset.knn_nhood
            n_classes = dataloader.dataset.n_classes    
            target_metric = dataloader.dataset.target_metric
            if self.is_supervised:
                metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="test")
            if self.knn_eval or not self.is_supervised:
                knn_metric = self.metric_fn(n_classes, dataloader.dataset.int_to_labels, mode="knn_val")
            iter_bar = tqdm(dataloader, desc='Testing', leave=True, total=len(dataloader))

            self.model.eval()                    
            with torch.no_grad():
                for images, labels in iter_bar: 
                    if len(labels) == 2 and isinstance(labels, list):
                        ids    = labels[1]
                        labels = labels[0]
                    labels = labels.to(self.device_id, non_blocking=True)
                    images = images.to(self.device_id, non_blocking=True)                   

                    if is_ddp(self.model):
                        outputs, features = self.model.module(images, return_embedding=True)
                    else:
                        outputs, features = self.model(images, return_embedding=True)

                    if self.log_embeddings:
                        feature_bank.append(features.clone().detach().cpu())                      
                    if self.knn_eval:
                        features = F.normalize(features, dim=1)
                        pred_labels = self.knn_predict(feature = features, 
                                                       feature_bank=self.feature_bank, 
                                                       feature_labels= self.targets_bank, 
                                                       knn_k=knn_nhood, knn_t=0.1, classes=n_classes,
                                                       multi_label = not dataloader.dataset.is_multiclass)
                        knn_metric.add_preds(pred_labels, labels, using_knn=True)
                    if self.is_supervised:
                        loss = self.criterion(outputs, labels)
                        test_loss.append(loss.item())
                        metric.add_preds(outputs, labels)

            if self.log_embeddings:
                self.build_umaps(feature_bank, dataloader, labels = metric.truths if self.is_supervised else knn_metric.truths,
                            mode = f"li_{layer}", wandb_logging=False)                         

            self.test_loss = np.array(test_loss).mean() if test_loss else None
            test_metrics[layer] = {}
            if self.is_supervised:
                test_metrics[layer] = metric.get_value(use_dist=isinstance(dataloader,DS))
            if self.knn_eval or not self.is_supervised:
                test_metrics[layer].update(knn_metric.get_value(use_dist=isinstance(dataloader,DS)))
            if self.is_supervised:
                self.test_target = test_metrics[layer][f"test_{target_metric}"]
                test_metrics[layer]['test_loss'] = round(self.test_loss, 5)
            else:
                self.test_target = test_metrics[layer].knn_test_accuracy

        self.model.train()
        self.set_models_precision(self.use_mixed_precision)

        splitted = self.model_name.split("-run_")
        if len(splitted) == 1:
            layer_imp_mname = f"{splitted[0]}-{self.mode}"
        elif len(splitted) == 2:
            layer_imp_mname = f"{splitted[0]}-{self.mode}-run_{splitted[1]}"
        else:
            raise ValueError(f"Model name {self.model_name} does not follow the naming convention")
        layer_imp_results_dir = os.path.join(self.save_dir, 'results', layer_imp_mname)
        layer_imp_metrics_path = os.path.join(layer_imp_results_dir, "metrics_results.json")

        check_dir(layer_imp_results_dir)
        save_json(test_metrics, layer_imp_metrics_path)
        print(f'Saved test_metrics to: "{layer_imp_metrics_path}"')

        print('\n',"--"*5, "{} evaluated on the test set".format(self.model_name), "--"*5,'\n')
        for k, v in test_metrics.items():
            print('\n',"--"*5, "Evaluation of {} ".format(k), "--"*5,'\n')
            v = pd.DataFrame.from_dict(v, orient='index').T
            print(tabulate(v, headers = 'keys', tablefmt = 'psql'))
        print('\n',"--"*35, '\n')            
        
        
class L2Tester(LayerImportanceTester):
    def __init__(self, wraped_defs):
        super().__init__(wraped_defs)

    def test(self, dataloader=None, **kwargs):
        if not self.is_rank0: return
        
        self.load_session()        
        dist = self.get_distance()
        
        results_dir = os.path.join(self.save_dir, 'results', self.model_name)
        check_dir(results_dir)
        l2_path = os.path.join(results_dir, "L2_results.json")
        save_json(dist, l2_path)
        print(f"L2 distance results are saved at:\n {l2_path}")
        
    def get_distance(self, reduce_fn=np.mean):
        state1, state2 = self.original_state, self.model_state
        print(f"\n\033[32mEvaluating l2 distance between the original and final state for: {self.model_name} \033[0m")
        dist = {}
        for layer in self.get_target_modules():
            org_state = OrderedDict([(k,v) for k,v in state1.items() 
                                     if layer in k  and check_type(k)])
            fin_state = OrderedDict([(k,v) for k,v in state2.items() 
                                     if layer in k and check_type(k)])
            dist[layer] = get_block_dist(state1, state2)
        dist['total'] = reduce_fn(list(dist.values()))
        return dist              
        
def check_type(_key):
    excluded = ['norm', 'bn']
    return not bool(sum([1 if _k in _key.split('.')[-2] else 0  
                         for _k in excluded]))

def get_block_dist(state1, state2, reduce_fn=np.mean, norm_n_params=True):
    assert state1.keys() == state2.keys()
    L2s = []
    for l_key in state1.keys():
        w_org = state1[l_key].cpu().numpy()
        w_fin = state2[l_key].cpu().numpy()
        l2_ = np.linalg.norm(w_org - w_fin) 
        if norm_n_params:
            l2_ = l2_/ w_org.size
        L2s.append(l2_)
    return reduce_fn(L2s)


  
        