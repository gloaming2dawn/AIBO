
import math
import time
import sys
from copy import deepcopy
import subprocess

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

import numpy as np
import torch
from torch.quasirandom import SobolEngine
from sklearn.preprocessing import power_transform, PowerTransformer

from .gp import train_gp
#from utils import from_unit_cube, latin_hypercube, to_unit_cube

import cma

import botorch
from botorch.acquisition import qExpectedImprovement, ExpectedImprovement, UpperConfidenceBound, qUpperConfidenceBound, qLowerBoundMaxValueEntropy, AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_model
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize

import warnings
warnings.simplefilter("ignore", cma.evolution_strategy.InjectionWarning)
import gc

# #pymoo-0.6.0
# from pymoo.algorithms.soo.nonconvex.ga import GA
# from pymoo.core.problem import Problem
# from pymoo.core.evaluator import Evaluator
# from pymoo.core.termination import NoTermination

#pymoo-0.5.0
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import Problem
from pymoo.core.evaluator import set_cv
from pymoo.util.termination.no_termination import NoTermination

sys.path.append("..")
from functions.synthetic import dict_tracker




class AIBO:
    """Multi-level acquisition function optimization based Bayesian Optimization

    Parameters
    ----------
    f : function handle
    lb : Lower variable bounds, numpy.array, shape (d,).
    ub : Upper variable bounds, numpy.array, shape (d,).
    n_init : Number of initial points (2*dim is recommended), int.
    max_evals : Total evaluation budget, int.
    batch_size : Number of points in each batch, int.
    verbose : If you want to print information about the optimization progress, bool.
    use_ard : If you want to use ARD for the GP kernel.
    max_cholesky_size : Largest number of training points where we use Cholesky, int
    n_training_steps : Number of training steps for learning the GP hypers, int
    min_cuda : We use float64 on the CPU if we have this or fewer datapoints
    device : Device to use for GP fitting ("cpu" or "cuda")
    dtype : Dtype to use for GP fitting ("float32" or "float64")

    Example usage:
        aibo = AIBO(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
        AIBO.optimize()  # Run optimization
    """

    def __init__(
        self,
        f,
        fname,
        lb,
        ub,
        n_init,
        max_evals,
        batch_size=10,
        n_init_acq=256,
        max_acq_size=128,
        n_restarts_acq=1,
        acqf_mode ='EI', #'EI' or 'UCB'
        beta=1.96,
        acqf_maxmizer = 'grad', # 'none' or 'grad'
        acqf_initializer={"random":{},"cmaes":{'sigma0':0.2},"ga":{'pop_size':50}},#{"cmaes":{'iters':1, 'popsize':1000}}
        y_transform = 'power_transform', #'power_transform' or 'standardize'
        minimize=True,
        verbose=True,
        use_ard=True,
        max_cholesky_size=2000,
        n_training_steps=50,
        min_cuda=1024,
        device="cpu",
        dtype="float64",
        istrackcands = False,
        istrackAF = False,
        seed = None,
        initial_guess=None,
    ):

        # Very basic input checks
        assert lb.ndim == 1 and ub.ndim == 1
        assert len(lb) == len(ub)
        assert np.all(ub > lb)
        assert max_evals > 0 and isinstance(max_evals, int)
        assert n_init > 0 and isinstance(n_init, int)
        assert batch_size > 0 and isinstance(batch_size, int)
        assert n_init_acq > 0 and isinstance(n_init_acq, int)
        assert max_acq_size > 0 and isinstance(max_acq_size, int)
        assert isinstance(verbose, bool) and isinstance(use_ard, bool)
        assert max_cholesky_size >= 0 and isinstance(batch_size, int)
        assert n_training_steps >= 30 and isinstance(n_training_steps, int)
        assert max_evals > n_init and max_evals > batch_size
        assert device == "cpu" or device == "cuda"
        assert dtype == "float32" or dtype == "float64"
        if device == "cuda":
            assert torch.cuda.is_available(), "can't use cuda if it's not available"
        
        # Save function information
        self.f = f
        self.fname = fname
        self.dim = len(lb)
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        
        # need change
        self.istrackcands = istrackcands
        self.tracker_cands = dict_tracker(self.fname, filename='cands')
        self.istrackAF = istrackAF
        self.tracker_AF_value = dict_tracker(self.fname, filename='AFvalue')
        
        # Settings
        self.n_init = n_init
        self.n_evals = 0
        self.max_evals = max_evals
        self.batch_size = batch_size
        self.max_batch_size = 50
        self.n_init_acq = n_init_acq
        self.max_acq_size = max_acq_size
        self.n_restarts_acq = n_restarts_acq
        self.minimize = minimize
        self.verbose = verbose
        self.use_ard = use_ard
        self.max_cholesky_size = max_cholesky_size
        self.n_training_steps = n_training_steps
        self.min_cuda = min_cuda
        self.acqf_mode = acqf_mode
        self.acqf_maxmizer = acqf_maxmizer
        self.beta = beta
        self.y_transform = y_transform
        
        # Hyperparameters
        self.mean = np.zeros((0, 1))
        self.signal_var = np.zeros((0, 1))
        self.noise_var = np.zeros((0, 1))
        self.lengthscales = np.zeros((0, self.dim)) if self.use_ard else np.zeros((0, 1))
        


        # Save the full history
        self.X = np.zeros((0, self.dim))
        self.X_unit = np.zeros((0, self.dim))
        self.fX = np.zeros((0, 1))
        self.fX_transformed = np.zeros((0, 1))
        
        # Save the recent history
        self._X = np.zeros((0, self.dim))
        self._fX = np.zeros((0, 1))
        self._X_unit = np.zeros((0, self.dim))
        self._fX_transformed = np.zeros((0, 1))
        
        self.failcount = 0
        self.succcount = 0
        self.failtol = np.max([4.0 , 2*self.batch_size, self.dim])
        self.length = 0.5
        self.length_min = 0.5 ** 7
        self.local_modeling = True
        self.initialization = True
        self.max_fitting_samples = np.max([200, int(self.dim*5)])


        # Device and dtype for GPyTorch
        self.dtype = torch.float32 if dtype == "float32" else torch.float64
        self.device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
        if self.verbose:
            print("Using dtype = %s \nUsing device = %s" % (self.dtype, self.device))
            sys.stdout.flush()
        
        self.acqf_initializer=acqf_initializer
        self.gp = None
        self.es = None
        self.cma=True
        self.initial_guess=initial_guess
        self.hypers={}
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(1e5)
            
    
    def _train_gp(self, X, fX, hypers={}):
        '''
        X (numpy array):
        fX:
        '''
        # Figure out what device we are running on
        if len(X) < self.min_cuda:
            device, dtype = torch.device("cpu"), torch.float64
        else:
            device, dtype = self.device, self.dtype
        
        if self.y_transform == 'power_transform':
            if (fX>0).all():
                # self.pt=PowerTransformer(method='box-cox')
                # self.pt.fit(fX)
                # y=self.pt.transform(fX)
                y=power_transform(fX,method='box-cox')
            else:
                # self.pt=PowerTransformer(method='yeo-johnson')
                # self.pt.fit(fX)
                # y=self.pt.transform(fX)
                y=power_transform(fX,method='yeo-johnson')
        else:
            y=(fX-fX.mean())/fX.std()
                    
        
        # 
        
        X_torch = torch.tensor(X).to(device=device, dtype=dtype)
        y_torch = torch.tensor(y).to(device=device, dtype=dtype)
        self.train_y=-y_torch
        
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size), gpytorch.settings.max_cg_iterations(2000):
                # botorch by default is for maxmizing the objective function
                self.gp = train_gp(
                    train_x=X_torch, train_y = -y_torch if self.minimize else y_torch, use_ard=self.use_ard, use_input_warping=False, num_steps=self.n_training_steps, lr=0.1, hypers=hypers
                )
                
                
                # print(hypers.items())
        # Remove the torch variables
        del X_torch, y_torch
        hypers = self.gp.state_dict()
        return self.gp, hypers


    def _random(self, acqf):
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
            samples = draw_sobol_samples(
                bounds=torch.stack([torch.zeros(self.dim), torch.ones(self.dim)]),
                n=self.n_init_acq, 
                q=1
            )
            cands=np.squeeze(samples.numpy())

            if self.n_init_acq > self.max_acq_size:
                cands_list = np.array_split(cands, np.ceil(self.n_init_acq/self.max_acq_size))
            else:
                cands_list = [cands]
            y = np.zeros((0,1))
            for X in cands_list:
                # t1=time.time()
                X_cand_torch = torch.tensor(X).to(device=self.device, dtype=self.dtype)
                # y_tmp = self.gp(X_cand_torch).sample(torch.Size([1])).cpu().detach().numpy().reshape(-1, 1)
                y_cand_torch = -acqf(X_cand_torch.unsqueeze(-2))
                y_tmp = y_cand_torch.detach().cpu().numpy().reshape(-1, 1)
                y=np.vstack((y, y_tmp))
                # print('single time', time.time()-t1)
            ind = np.argsort(y, axis=0)[:self.n_restarts_acq]
            x0=np.take_along_axis(cands, ind, axis=0)
        return x0
    
    
    
    def _cmaes(self, acqf): 
        # self.es.ask()
        # es =deepcopy(self.es)
        # cands = np.zeros((0,self.dim))
        # y = np.zeros((0,1))
        # with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        #     for i in range (10):
        #         X=np.array(es.ask(number=50))
        #         X_cand_torch = torch.tensor(X).to(device=self.device, dtype=self.dtype)
        #         y_cand_torch = -acqf(X_cand_torch.unsqueeze(-2))
        #         y_tmp = y_cand_torch.detach().cpu().numpy().reshape(-1, 1)
        #         y_tmp=self.pt.inverse_transform(y_tmp)
        #         y=np.vstack((y, y_tmp))
        #         cands=np.vstack((cands, X))
        #         es.tell(X, y_tmp)
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
            cands=np.array(self.es.ask(number=self.n_init_acq))
            if self.n_init_acq > self.max_acq_size:
                cands_list = np.array_split(cands, np.ceil(self.n_init_acq/self.max_acq_size))
            else:
                cands_list = [cands]
            y = np.zeros((0,1))
            for X in cands_list:
                X_cand_torch = torch.tensor(X).to(device=self.device, dtype=self.dtype)
                # y_tmp = self.gp(X_cand_torch).sample(torch.Size([1])).cpu().detach().numpy().reshape(-1, 1)
                y_cand_torch = -acqf(X_cand_torch.unsqueeze(-2))
                y_tmp = y_cand_torch.detach().cpu().numpy().reshape(-1, 1)
                y=np.vstack((y, y_tmp))  
            ind = np.argsort(y, axis=0)[:self.n_restarts_acq]
            x0=np.take_along_axis(cands, ind, axis=0)
            
        return x0
    
    def _ga(self, acqf):
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
            pop = self.ga.ask()
            cands =pop.get("X")
            if self.n_init_acq > self.max_acq_size:
                cands_list = np.array_split(cands, np.ceil(self.n_init_acq/self.max_acq_size))
            else:
                cands_list = [cands]
            y = np.zeros((0,1))
            for X in cands_list:
                X_cand_torch = torch.tensor(X).to(device=self.device, dtype=self.dtype)
                # y_tmp = self.gp(X_cand_torch).sample(torch.Size([1])).cpu().detach().numpy().reshape(-1, 1)
                y_cand_torch = -acqf(X_cand_torch.unsqueeze(-2))
                y_tmp = y_cand_torch.detach().cpu().numpy().reshape(-1, 1)
                y=np.vstack((y, y_tmp))
            ind = np.argsort(y, axis=0)[:self.n_restarts_acq]
            x0=np.take_along_axis(cands, ind, axis=0)
            
        return x0
        
    def _select_candidates(self, gp_model, X_unit, fX, batch_size=1):
        """Select candidates."""
        # We may have to move the GP to a new device
        # device=torch.device("cpu")
        gp_model = gp_model.to(dtype=self.dtype, device=self.device)
        if self.acqf_mode == 'UCB':
            acqf = qUpperConfidenceBound(gp_model, beta=self.beta)
        elif self.acqf_mode == 'EI':
            acqf = qExpectedImprovement(gp_model, self.train_y.max().to(dtype=self.dtype, device=self.device))
        else:
            assert 1==0, 'please select a correct acqf_mode'
        
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
            X_next=[]
            candidate_list=[]
            base_X_pending = acqf.X_pending
            acqf_initializer_list=[]
            t_ms=0
            t_bfgs=0
            for i in range(batch_size):
                torch.cuda.empty_cache() 
                X0 = np.zeros((0, self.dim))
                ind2opt={}
                ind=0
                t0=time.time()
                if "ga" in self.acqf_initializer:
                    if len(self.X) >= self.acqf_initializer['ga']['pop_size']:
                        x0 = self._ga(acqf)
                    else:
                        x0 = self._random(acqf)
                    
                    X0 = np.vstack((X0, x0))
                    ind2opt[ind]='ga'
                    ind += 1
                
                if "cmaes" in self.acqf_initializer:
                    if self.cma:
                        x0 = self._cmaes(acqf)
                        X0 = np.vstack((X0, x0))
                        ind2opt[ind]='cmaes'
                        ind += 1
                    else:
                        self.es.ask()
                
                
                if "random" in self.acqf_initializer:
                    x0 = self._random(acqf)
                    # X_cand_torch = torch.tensor(x0[0]).to(device=self.device, dtype=self.dtype)
                    # print('acqf',acqf(X_cand_torch.unsqueeze(-2)).item())
                    X0 = np.vstack((X0, x0))
                    ind2opt[ind]='random'
                    ind += 1

                

                t_ms+=time.time()-t0
                
                
                if self.istrackAF:
                    AF_value = {}
                
                if self.acqf_maxmizer == 'none':
                    batch_candidates = torch.from_numpy(X0).to(dtype=self.dtype, device=self.device).unsqueeze(1)
                    batch_acq_values = acqf(batch_candidates)
                    indbest = torch.argmax(batch_acq_values.view(-1), dim=0)
                    best_cand = batch_candidates[indbest]
                    acqf_initializer_list.append(ind2opt[int(indbest.item()/self.n_restarts_acq)])
                    X_next.append(deepcopy(np.squeeze(best_cand.detach().cpu().numpy())))
                    
                    if self.istrackAF:
                        posterior = gp_model.posterior(batch_candidates)
                        means = posterior.mean.view(-1).cpu().detach().numpy()
                        variances = posterior.variance.view(-1).cpu().detach().numpy()
                        acq_values = batch_acq_values.view(-1).cpu().detach().numpy()
                        vecs = np.split(acq_values, len(acq_values)//self.n_restarts_acq)
                        for jj in range(len(vecs)):
                            indbest = np.argmax(vecs[jj])
                            ind1 = jj*self.n_restarts_acq + indbest
                            AF_value[ind2opt[jj]] = {'mean':means[ind1], 'variance':variances[ind1], 'value':acq_values[ind1]}
                
                
                    
                    
                elif self.acqf_maxmizer == 'grad':
                    t0=time.time()
                    with gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
                        batch_candidates, batch_acq_values = optimize_acqf(
                            acqf,
                            bounds=torch.stack(
                                [
                                    torch.zeros(self.dim, dtype=self.dtype, device=self.device),
                                    torch.ones(self.dim, dtype=self.dtype, device=self.device),
                                ]
                            ),
                            q=1,
                            num_restarts=len(X0),
                            batch_initial_conditions=torch.from_numpy(X0).to(dtype=self.dtype, device=self.device).unsqueeze(1),
                            options={'batch_limit': self.max_acq_size, 'maxiter': 40},#change#add maxiter to avoid too long time for optimizing acq func
                            return_best_only=False,
                            sequential=False,
                        )
                    t_bfgs+=time.time()-t0
                    
                    indbest = torch.argmax(batch_acq_values.view(-1), dim=0)
                    best_cand = batch_candidates[indbest]
                    acqf_initializer_list.append(ind2opt[int(indbest.item()/self.n_restarts_acq)])
                    X_next.append(deepcopy(np.squeeze(best_cand.detach().cpu().numpy())))
                    
                    if self.istrackAF:
                        posterior = gp_model.posterior(batch_candidates)
                        means = posterior.mean.view(-1).cpu().detach().numpy()
                        variances = posterior.variance.view(-1).cpu().detach().numpy()
                        acq_values = batch_acq_values.view(-1).cpu().detach().numpy()
                        vecs = np.split(acq_values, len(acq_values)//self.n_restarts_acq)
                        for jj in range(len(vecs)):
                            indbest = np.argmax(vecs[jj])
                            ind1 = jj*self.n_restarts_acq + indbest
                            AF_value[ind2opt[jj]] = {'mean':means[ind1], 'variance':variances[ind1], 'value':acq_values[ind1]}
                
                if self.istrackAF:
                    self.tracker_AF_value.track(AF_value)
                
                
                    
                    

                
                
                
                
                candidate_list.append(best_cand)
                candidates = torch.cat(candidate_list, dim=-2)
                acqf.set_X_pending(
                    torch.cat([base_X_pending, candidates], dim=-2)
                    if base_X_pending is not None
                    else candidates
                )
                
                

        print('t_ms:{}   t_bfgs:{}'.format(t_ms,t_bfgs))
        print(acqf_initializer_list)            
        X_next= np.array(X_next)
        return X_next, acqf_initializer_list
    

    
    def ask(self):
        if self.initialization:
            k=0 if self.initial_guess is None else 1
            sobol = SobolEngine(dimension=self.dim, scramble=True, seed=self.seed)
            X_next_unit_torch = sobol.draw(n=self.n_init-k) #torch.tensor
            X_next_torch = unnormalize(X_next_unit_torch, torch.from_numpy(np.array([self.lb, self.ub])) )
            X_next_unit=X_next_unit_torch.numpy()
            X_next=X_next_torch.numpy()
            if self.initial_guess is not None:
                x0=np.array(self.initial_guess)
                X_next=np.vstack((deepcopy(x0),X_next))
                self.initial_guess=None
            acqf_initializer_list=['random']*self.n_init
        else:
            t0=time.time()
            self.gp, _ = self._train_gp(self._X_unit, self._fX)
            print('training time:', time.time()-t0)
            X_next_unit, acqf_initializer_list = self._select_candidates(self.gp, self._X_unit, self._fX, batch_size=self.batch_size)
            

            X_next = X_next_unit * (self.ub - self.lb) + self.lb

        return X_next, acqf_initializer_list
        
    def tell(self, X_next, fX_next):
        X_next_unit=(X_next - self.lb) / (self.ub - self.lb)
         # Update budget
        self.n_evals += len(X_next)
        # Append data to the random history
        self.X = np.vstack((self.X, deepcopy(X_next)))
        self.fX = np.vstack((self.fX, deepcopy(fX_next)))
        self.X_unit = (self.X - self.lb) / (self.ub - self.lb)
        
        if self.initialization:
            self.initialization = False
            self._X_unit, self._X, self._fX = X_next_unit, X_next, fX_next
            ind = np.argsort(self._fX, axis=0)[:1]
            x0=np.take_along_axis(self._X_unit, ind, axis=0)
            if 'cmaes' in self.acqf_initializer:
                self.es = cma.CMAEvolutionStrategy(
                    x0=x0, #np.random.rand(self.dim),
                    sigma0=self.acqf_initializer['cmaes']['sigma0'],
                    inopts={'bounds': [0, 1], 'popsize': self.batch_size, 'verbose': -9},
                )#,'maxfevals': 5000, 'tolx':1e-4
                self.es.ask()
                self.es.tell(self._X_unit, self._fX.ravel().tolist())
            
            if 'ga' in self.acqf_initializer:
                # pymoo-0.6.0
                # problem = Problem(n_var=self.dim, n_obj=1, n_constr=0, xl=np.zeros(self.dim), xu=np.ones(self.dim))
                # popsize = self.acqf_initializer['ga']['pop_size']
                # self.ga = GA(pop_size=self.acqf_initializer['ga']['pop_size'], n_offsprings=self.n_init_acq)
                # self.ga.setup(problem, termination=NoTermination())
                # self.pop = self.ga.ask()
                # self.pop = self.pop[:self.n_init]
                # self.pop.set("X", self._X_unit[:popsize])
                # self.pop.set("F", self._fX[:popsize])
                # self.ga.tell(infills=self.pop)
                
                # pymoo-0.5.0
                problem = Problem(n_var=self.dim, n_obj=1, n_constr=0, xl=np.zeros(self.dim), xu=np.ones(self.dim))
                self.ga = GA(pop_size=5*self.batch_size, n_offsprings=self.n_init_acq)
                self.ga.setup(problem, termination=NoTermination())
                self.pop = self.ga.ask()
                self.pop = self.pop[:self.n_init]
                self.pop.set("X", self._X_unit)
                self.pop.set("F", self._fX)
                set_cv(self.pop)
                self.ga.tell(infills=self.pop)

        else:
            # Append data to the recent history after restarting which is used for training gp model
            self._X = np.vstack((self._X, deepcopy(X_next)))
            self._fX = np.vstack((self._fX, deepcopy(fX_next)))
            self._X_unit = (self._X - self.lb) / (self.ub - self.lb)
            if 'cmaes' in self.acqf_initializer:
                self.es.tell(X_next_unit, fX_next.ravel().tolist())
            if 'ga' in self.acqf_initializer:
                self.pop=self.pop[:self.batch_size]
                self.pop.set("X", X_next_unit)
                self.pop.set("F", fX_next)
                # set_cv only used for pymoo-0.5.0
                set_cv(self.pop)
                self.ga.tell(infills=self.pop)
                
    def check_point(self):
        pass         
        
    def optimize(self):
        self.check_point()
        """Run the full optimization process."""
        epochs=0
        self.failcount = 0
        while self.n_evals < self.max_evals:
            
            if self.n_init < self.batch_size:
                self.n_init = self.batch_size 
            t0=time.time()
            X_next, acqf_initializer_list = self.ask()
            # print('ask time:',time.time()-t0)
            # Evaluate batch
            fX_next = np.array([[self.f(x)] for x in X_next])
            
            if len(self._fX) >0:
                if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
                    self.failcount = 0
                else:
                    self.failcount += 1
            self.tell(X_next, fX_next)
            if self.verbose:
                print("{}) {} fbest = {:.4f} recent_fbest = {:.4f}".format(self.n_evals, self.fname, self.fX.min(), self._fX.min()))
                ind = np.argsort(fX_next, axis=0)[:1]
                print('this iteration best is from',acqf_initializer_list[ind.item()])
                # ind = np.argsort(self.fX, axis=0)[:1]
                # x0=np.take_along_axis(self.X, ind, axis=0)
                # print(np.round(x0,3))
            if 'cmaes' in self.acqf_initializer:    
                print('cmaes initializer sigma:',self.es.sigma)
            
            
            
            if self.failcount > np.max([int(self.dim/self.batch_size*10),1000]):
                ## not used actually
                self.initialization=True
                self.failcount = 0
                epochs += 1
                print('='*20)
                print('restart epoch {}'.format(epochs))
                
            
            #Avoid OOM
            gc.collect()
            torch.cuda.empty_cache()  
            
        #print final results
        print('best x:', self.X[np.argmin(self.fX),:])



                

