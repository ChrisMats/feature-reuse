from ablations.layer_importance.trainer import *
from ablations.layerwise_knn.trainer import *
from .similarity_functions import CKA
        
class WeightSimilatiryTester(LayerImportanceTester):
    def __init__(self, wraped_defs, cross_wtst=False):
        super().__init__(wraped_defs) 
        self.wraped_defs = wraped_defs
        self.cross_wtst = cross_wtst
        self.pool_features = False        
    
    def test(self, dataloader=None, **kwargs):
        if self.cross_wtst:
            self.cross_wtst_test(dataloader=dataloader, **kwargs)
        else:
            self.inner_model_test(dataloader=dataloader, **kwargs)
        
    def inner_model_test(self, dataloader=None, **kwargs):
        
        if not self.is_rank0: return
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.set_models_precision(False)
        
        if dataloader == None:
            dataloader=self.testloader
        
        # get named modules and define similarity modes
        layer_names = self.get_target_modules()
        similarity_res = edict({})
        
        # load the trained and the origiinal state
        self.load_session()
        states = {"init":self.original_state, "trained":self.model_state}
        
        # Defining intermediate layers and wrap model
        self.model = KnnHookedModel(self.model, layer_names, self.pool_features)        
        
        self.model.eval()
        iter_bar = tqdm(dataloader, desc='Calulating Similatiry', leave=True, total=len(dataloader))
        with torch.no_grad():
            for images, labels in iter_bar: 
                images = images.to(self.device_id, non_blocking=True)                   
                
                features = {"trained":{}, "init":{}}
                # loop through the the trained and the origiinal state
                for state_name, state in states.items():
                
                    # load state
                    self.model.model.load_state_dict(state)
                    
                    # run inference
                    if is_ddp(self.model):
                        raise NotImplementedError("LayerWise similarity does not support DDP")
                    else:
                        features[state_name] = self.model(images)
                        
                # get batch-wise similarity
                batched_sim = self.get_similarities(features, layer_names)
                for sim_mode in batched_sim.keys():
                    if sim_mode not in similarity_res:
                        similarity_res[sim_mode] = edict({})
                    for feature_type in batched_sim[sim_mode].keys():
                        if feature_type not in similarity_res[sim_mode]:
                            similarity_res[sim_mode][feature_type] = []                
                        similarity_res[sim_mode][feature_type].append(batched_sim[sim_mode][feature_type])
        
        # average similarity
        for sim_mode in similarity_res.keys():
            for feature_type in similarity_res[sim_mode]:
                similarity_res[sim_mode][feature_type] = torch.stack(
                    similarity_res[sim_mode][feature_type]).mean(0).cpu().numpy()

        self.model.train()
        self.set_models_precision(self.use_mixed_precision)

        splitted = self.model_name.split("-run_")
        if len(splitted) == 1:
            sim_mname = f"{splitted[0]}-weight_similatiry"
        elif len(splitted) == 2:     
            sim_mname = f"{splitted[0]}-weight_similatiry-run_{splitted[1]}"
        else:
            raise ValueError(f"Model name {self.model_name} does not follow the naming convention")
        sim_results_dir = os.path.join(self.save_dir, 'results', sim_mname)
        sim_path = os.path.join(sim_results_dir, "similarity_results.pickle")
        check_dir(sim_results_dir)   
        save_pickle(similarity_res, sim_path)
        print(f'Saved results to: {sim_path} \n')
        
    def cross_wtst_test(self, dataloader=None, **kwargs):

        if not self.is_rank0: return
        self.test_mode = True
        self.restore_session = True
        self.restore_only_model = True
        self.set_models_precision(False)
        
        if dataloader == None:
            dataloader=self.testloader

        # get named modules and define similarity modes
        layer_names = self.get_target_modules()
        similarity_res = edict({})
        
        # load the trained states for min and max checkpoint training mode (e.g. -2 to 12 for DeiT)
        
        # name handling
        self.get_saved_model_path()
        model_base = deepcopy(self.model_path)
        base_dir = os.path.dirname(model_base)
        model_name = os.path.basename(model_base)
        if "-run_" in model_name:
            run_id = model_name.split("run_")[-1]
        prefix = model_name.split("_tul_")[0]

        # check for name of min path
        min_path = os.path.join(base_dir, f"{prefix}-run_{run_id}")
        if not os.path.isfile(min_path):
            min_path = os.path.join(base_dir, f"{prefix}_tul_-2-run_{run_id}")
            if not os.path.isfile(min_path):
                raise FileNotFoundError(f" ST path not found: {min_path}")
        # check for name of max path
        candidates = [os.path.basename(p) for p in glob(os.path.join(base_dir, "*"))
                      if os.path.basename(p).startswith(prefix)]
        tuls = [c.split("_tul_")[-1].split("-run_")[0] for c in candidates]
        max_tul = max([int(c) for c in tuls  if c.isdigit()])
        max_path = os.path.join(base_dir, f"{prefix}_tul_{max_tul}-run_{run_id}")
        if not os.path.isfile(max_path):
            raise FileNotFoundError(f" WT path not found: {max_path}")        
        
        # load states
        states = {}
        self.load_session(model_path=min_path)
        states["random"] = deepcopy(self.model_state)
        self.load_session(model_path=max_path)
        states["pretrained"] = deepcopy(self.model_state)
        
        # Defining intermediate layers and wrap model
        self.model = KnnHookedModel(self.model, layer_names, self.pool_features)        
        
        self.model.eval()
        iter_bar = tqdm(dataloader, desc='Calulating Similatiry', leave=True, total=len(dataloader))
        with torch.no_grad():
            for images, labels in iter_bar: 
                images = images.to(self.device_id, non_blocking=True)                   
                
                features = {"random":{}, "pretrained":{}}
                # loop through the the trained and the origiinal state
                for state_name, state in states.items():
                
                    # load state
                    self.model.model.load_state_dict(state)
                    
                    # run inference
                    if is_ddp(self.model):
                        raise NotImplementedError("LayerWise similarity does not support DDP")
                    else:
                        features[state_name] = self.model(images)
                        
                # get batch-wise similarity
                batched_sim = self.compute_cka(features, layer_names, "pretrained", "random")
                for feature_type in batched_sim.keys():
                    if feature_type not in similarity_res:
                        similarity_res[feature_type] = []                
                    similarity_res[feature_type].append(batched_sim[feature_type])
        
        # average similarity
        for feature_type in similarity_res:
            similarity_res[feature_type] = torch.stack(
                similarity_res[feature_type]).mean(0).cpu().numpy()

        self.model.train()
        self.set_models_precision(self.use_mixed_precision)

        splitted = self.model_name.split("-run_")
        _mname = self.model_name.split("-wt")[0]
        if len(splitted) == 1:
            sim_mname = f"{_mname}-cross_weight_similatiry"
        elif len(splitted) == 2:     
            sim_mname = f"{_mname}-cross_weight_similatiry-run_{splitted[1]}"
        else:
            raise ValueError(f"Model name {self.model_name} does not follow the naming convention")
        sim_results_dir = os.path.join(self.save_dir, 'results', sim_mname)
        sim_path = os.path.join(sim_results_dir, "similarity_results.pickle")
        check_dir(sim_results_dir)   
        save_pickle(similarity_res, sim_path)
        print(f'Saved similarity results to: {sim_path}')
        
        
    def compute_cka(self, features, layer_order, mode1, mode2, calc_fc=False):
        
        depth = len(layer_order) if calc_fc else len(layer_order) - 1
        sim = {}
        
        for i, feat_name1 in enumerate(layer_order):
            if not calc_fc and feat_name1 == 'fc': continue 
                
            for j, feat_name2 in  enumerate(layer_order):
                if not calc_fc and feat_name2 == 'fc': continue 
                feature_types = features[mode1][feat_name1].keys() & features[mode1][feat_name2].keys()
                for f_t in feature_types:
                    if f_t not in sim:
                        sim[f_t] = torch.zeros(depth, depth)
                    feat1 = features[mode1][feat_name1][f_t].flatten(1)
                    feat2 = features[mode2][feat_name2][f_t].flatten(1)    
                    sim[f_t][i,j] = CKA(feat1.to(self.device_id), feat2.to(self.device_id))
                
        return sim
        
    
    def get_similarities(self, features, layer_order, calc_fc=False):

        init_to_init = self.compute_cka(features, layer_order, "init", "init", calc_fc)
        trained_to_init = self.compute_cka(features, layer_order, "trained", "init", calc_fc)
        trained_to_trained = self.compute_cka(features, layer_order, "trained", "trained", calc_fc)          

        return edict({"trained_to_trained": trained_to_trained,
                      "trained_to_init": trained_to_init,
                      "init_to_init": init_to_init })
                
                
        
