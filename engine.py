import os
import jax
import torch
import numpy as np
# import neural_tangents as nt

jax.config.update("jax_enable_x64", True)
from pathlib import Path
from absl import app, flags
from jax import numpy as jnp, random
from jax.numpy import linalg as jla
from jax.flatten_util import ravel_pytree
from dictionaries import data_dict, model_dict, criterion_dict, activation_dict
from loss import build_loss
from util import load_config, write_stats, flatten_tree
from taylor import find_instability, track_dynamics
import json
from functools import partial
import shutil
from torch.utils.tensorboard import SummaryWriter


class BaseEngine(object):

    def __init__(self, config_path, logger, save_dir):
        # Assign a logger
        self.logger = logger

        # Load configurations
        self.config = load_config(config_path)

        self.data_config = self.config['data']
        self.train_config = self.config['train']
        self.model_config = self.config['model']
        self.grokking_config = self.config['grokking']
        self.computation_config = self.config['computation']
        self.verbose = self.config['train'].get('verbose', False)
        self.experimental = self.config['train'].get('experimental', False)
        self.exp_period = self.config['train'].get('exp_period', 1000)

        # Determine which device to use
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.num_devices = torch.cuda.device_count()

        if device == 'cpu':
            self.logger.warn('GPU is not available.')
        else:
            self.logger.warn('GPU is available with {} devices.'.format(self.num_devices))
        self.logger.warn('CPU is available with {} devices.'.format(jax.device_count('cpu')))

        # Load a summary writer
        self.save_dir = save_dir
        log_dir = os.path.join(self.save_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def run(self):
        pass


class Engine(BaseEngine):

    def __init__(self, config_path, logger, save_dir):
        super(Engine, self).__init__(config_path, logger, save_dir)

    def _build(self):

        def build_data(data_config, model_config):
            return data_dict[data_config['name']](data_config, model_config['width'])
        self.train_data, self.test_data = build_data(self.data_config, self.model_config)
        self.is_quadratic = self.data_config.get('is_quadratic', False)
        self.is_classification = self.model_config['n_classes'] > 1
        if self.is_classification:
            self.Y = jax.nn.one_hot(self.train_data[1], self.model_config['n_classes'])
            self.Y_test = jax.nn.one_hot(self.test_data[1], self.model_config['n_classes'])
        else:
            self.Y = jnp.atleast_2d(self.train_data[1]).T
            self.Y_test = jnp.atleast_2d(self.test_data[1]).T

        def build_criterion(train_config):
            return criterion_dict[train_config['criterion']]
        self.criterion = build_criterion(self.train_config)  # Only fixed learning is supported for now

        mu = self.Y.sum(axis=0) / len(self.Y)
        std = jnp.sqrt(((self.Y - mu) ** 2).mean(axis=0))

        def build_model(model_config):
            width = model_config['width']
            num_classes = model_config['n_classes']
            activation = activation_dict[model_config['activation']]
            name = model_config['name']
            d_vocab = model_config['d_vocab']
            use_bias = model_config.get('use_bias', True)
            kwargs = {
                'activation': activation,
                'n_classes': num_classes,
                'width': width,
                'd_vocab': d_vocab,
                'use_bias': use_bias
            }
            if 'mlp' in name:
                kwargs.update({'depth': model_config['depth']})
                if name.startswith('normalized_'):
                    kwargs.update({
                        'normalization_scale': mu,
                        'normalization_bias': std
                    })
            return model_dict[name](**kwargs)
        self.model = build_model(self.model_config)
        self.is_linear_regression = self.model_config['depth'] < 2

        model_key, self.eig_key = random.split(random.PRNGKey(self.data_config['seed']))
        self.p, loss = build_loss(
            model=self.model,
            data=self.train_data,
            criterion=self.criterion,
            batch_size=None,
            # batch_size=min(self.train_config['ghost_batch_size'], len(self.train_data[1])),
            model_key=model_key,
            l2_reg=self.train_config.get('l2_reg', 0.0),
            is_classification=self.is_classification
        )
        dtype_dict = dict(f32=jnp.float32, f64=jnp.float64)
        self.loss = loss._replace(
            D=partial(loss.D, dtype=dtype_dict[self.computation_config.get('deriv_dtype', 'f32')]),
            eig=partial(
                loss.eig,
                tol=self.computation_config.get('solver_tol', 1e-9),
                hvp_dtype=dtype_dict[self.computation_config.get('hvp_dtype', 'f32')],
                solver_dtype=dtype_dict[self.computation_config.get('solver_dtype', 'f32')],
            ),
        )
        # self.e_loss = build_loss(
        #     model=self.model,
        #     data=self.test_data,
        #     criterion=self.criterion,
        #     # batch_size=min(self.train_config['ghost_batch_size'], len(self.test_data[1])),
        #     init=False,
        #     p_info=(self.p, self.t_loss.unravel_p),
        #     batch_size=None,
        #     model_key=model_key,
        #     l2_reg=self.train_config.get('l2_reg', 0.0)
        # )[1]
        # self.e_loss = self.e_loss._replace(
        #     D=partial(loss.D, dtype=dtype_dict[self.computation_config.get('deriv_dtype', 'f32')]),
        #     eig=partial(
        #         loss.eig,
        #         tol=self.computation_config.get('solver_tol', 1e-9),
        #         hvp_dtype=dtype_dict[self.computation_config.get('hvp_dtype', 'f32')],
        #         solver_dtype=dtype_dict[self.computation_config.get('solver_dtype', 'f32')],
        #     ),
        # )

        def apply_fn_trace(params, x):
            out = self.model.apply(params, x, mutable=['batch_stats'])[0]
            return jnp.sum(out, axis=-1) / out.shape[-1] ** 0.5

        # NTK stuff
        # kwargs = dict(
        #     f=apply_fn_trace,
        #     trace_axes=(),
        #     vmap_axes=0
        # )
        # self.ntk_fn = jax.jit(nt.empirical_ntk_fn(**kwargs, implementation=nt.NtkImplementation.JACOBIAN_CONTRACTION))

    def _build_quadratic_features(self):
        if hasattr(self, 'X'):
            return
        if self.is_quadratic:
            self.X = self.train_data[0]
            self.X_test = self.test_data[0]
        else:
            self.X = self._to_quadratic_data(self.train_data[0])
            self.X_test = self._to_quadratic_data(self.test_data[0])

    def _acc(self, X, Y):
        if self.is_classification:
            return (X.argmax(axis=1) == Y.argmax(axis=1)).sum() / Y.shape[0]
        else:
            return (jnp.round(X) == Y).sum() / Y.shape[0]

    def _mse_loss(self, X, Y):
        return ((X - Y) ** 2).sum() / np.prod(Y.shape)

    # Maybe we should change this to track before EOS? Should see what's happening in experiments and then decide
    def _should_track_measures(self, step):
        if self.grokking_config.get('warmup_track', True) \
                and step < self.grokking_config.get('warmup_track_epoch', 1000):
            return step % 50 == 0
        elif self.grokking_config.get('early_track', True) \
                and step < self.grokking_config.get('early_track_epoch', 5000):
            return step % 100 == 0

    def _kernel_regression_helper(self, kernel, ts, l2_regs):

        # kernel = lambda x, y: x @ y.T
        l2_regs = np.logspace(-3, 5)
        ts = np.concatenate([np.arange(0, 1, step=.001), np.arange(1, 10, step=.01), np.arange(10, 100, .1)])

        k_test_train = kernel(self.X_test, self.X)
        k_train_train = kernel(self.X, self.X)
        evals, evecs = jla.eigh(k_train_train)

        best_test_loss = 10
        best_loss_params = None
        best_test_acc = 0.0
        best_acc_params = None

        for l2 in l2_regs:
            inv = evecs @ jnp.diag(1. / (evals+l2)) @ evecs.T
            dest = inv @ self.Y
            lhs = inv @ evecs
            rhs = evecs.T @ self.Y
            for t in ts:
                inter = dest - lhs @ jnp.diag(-t * jnp.exp(evals+l2)) @ rhs
                # pred_t = k_train_train @ inter
                pred_e = k_test_train @ inter
                # train_loss = self._mse_loss(pred_t, self.Y)
                test_loss = self._mse_loss(pred_e, self.Y_test)
                # print('train loss:', train_loss)
                # print('test loss:', test_loss)
                # train_acc = self._acc(pred_t, self.Y)
                test_acc = self._acc(pred_e, self.Y_test)
                # print('train acc:', train_acc)
                # print('test acc:', test_acc)
                # print()
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    best_loss_params = (l2, t)
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_acc_params = (l2, t)
            print('Done with l2', l2)
            print('Best test loss:', best_test_loss, 'with params', best_loss_params)
            print('Best test acc:', best_test_acc, 'with params', best_acc_params)
            print()

        print('#########################')
        print('#########################')
        print('#########################')
        print('Finished!')
        print('Best test loss:', best_test_loss, 'with params', best_loss_params)
        print('Best test acc:', best_test_acc, 'with params', best_acc_params)

    def _ode_solver(self, p, ts):

        params = self.loss.unravel_p(p)
        W = torch.from_numpy(jax.device_get(params['params']['Dense_0']['kernel'].T.astype('float64'))).cuda()
        V = torch.from_numpy(jax.device_get(params['params']['Dense_1']['kernel'].T.astype('float64'))).cuda()
        X = torch.from_numpy(jax.device_get(self.train_data[0].astype('float64'))).cuda()  # n x d
        Y = torch.from_numpy(
            jax.device_get(jax.nn.one_hot(self.train_data[1], self.model_config['n_classes']).astype('float64'))
        ).cuda()  # n x c
        X_test = torch.from_numpy(jax.device_get(self.test_data[0].astype('float64'))).cuda()  # nt x d
        Y_test = torch.from_numpy(
            jax.device_get(jax.nn.one_hot(self.test_data[1], self.model_config['n_classes']).astype('float64'))
        ).cuda()  # nt x c
        from torchdiffeq import odeint

        def derivative(t, W):
            Z = W @ X.T
            return (V.T @ (V @ Z ** 2 - Y.T)) * Z @ X

        def mse(W, x, y):
            pred = (V @ (W @ x.T) ** 2).T
            return (((pred - y) ** 2).sum() / torch.prod(torch.tensor(y.shape))).item()

        def acc(W, x, y):
            pred = (V @ (W @ x.T) ** 2).T
            return ((pred.argmax(axis=1) == y.argmax(axis=1)).sum() / len(y)).item()

        answers = []
        for t in ts:
            ans = odeint(derivative, W, torch.linspace(0, t, steps=100))
            answers.append(ans)
            print(t, 'Done.')
            print('Train stats - loss:', mse(ans[-1], X, Y), 'acc:', acc(ans[-1], X, Y))
            print('Test stats - loss:', mse(ans[-1], X_test, Y_test), 'acc:', acc(ans[-1], X_test, Y_test))

    def _first_layer_train_solution(self, p):

        def null_space(A):
            u, s, vh = jla.svd(A, full_matrices=True)
            M, N = u.shape[0], vh.shape[1]
            tol = jnp.amax(s) * 0
            num = jnp.sum(s > tol, dtype=int)
            Q = vh[num:, :].T.conj()
            return Q

        params = self.loss.unravel_p(p)
        new_p = p.copy()
        new_params = self.loss.unravel_p(new_p).unfreeze()
        W = params['params']['Dense_0']['kernel'].T.astype('float64')
        V = params['params']['Dense_1']['kernel'].T.astype('float64')
        X = self.train_data[0].astype('float64')
        Y = jax.nn.one_hot(self.train_data[1], self.model_config['n_classes']).astype('float64')
        pvy = jla.pinv(V) @ Y.T
        nv = null_space(V)
        z = jla.pinv(nv) @ jnp.ones((nv.shape[0], pvy.shape[1]))  # Finding z for: nv z = 1 (matrix of ones)
        nvz = nv @ z
        mid = pvy + (nvz * jnp.abs(pvy.min() / nvz.min()))
        ans = (mid ** .5) @ jla.pinv(X.T)  # Answer for (W @ X.T)^2 = mid
        new_params['params']['Dense_0']['kernel'] = ans.T  # Note the transpose!

        new_p = ravel_pytree(new_params)[0]

        td, ed = self.train_data, self.test_data
        t_L, t_dL = self.loss.value_and_grad(new_p, data=td)
        t_acc = self.loss.acc(new_p, data=td)
        e_L, e_dL = self.loss.value_and_grad(new_p, data=ed)
        e_acc = self.loss.acc(new_p, data=ed)
        print('t loss:', t_L, 't acc:', t_acc, 't_dL.max():', t_dL.max())
        print('e loss:', e_L, 'e acc:', e_acc, 'e_dL.max():', e_dL.max())

    def _gd_step(self, p, U, Un, Ut, Unt, step, in_eos):
        lr = self.train_config['lr']
        t_L, t_dL = self.loss.value_and_grad(p, data=self.train_data)
        eig, eign, eigt, eignt, e_acc = self._track_prestep_stats(p, lr, t_L, t_dL, U, Un, Ut, Unt, step, in_eos)
        if self.train_config.get('first_layer_only', False):
            t_loss = self.loss.unravel_p(t_dL).unfreeze()
            t_loss['params']['Dense_1']['kernel'] = jnp.zeros_like(t_loss['params']['Dense_1']['kernel'])
            t_dL = ravel_pytree(t_loss)[0]
        elif self.train_config.get('second_layer_only', False):
            t_loss = self.loss.unravel_p(t_dL).unfreeze()
            t_loss['params']['Dense_0']['kernel'] = jnp.zeros_like(t_loss['params']['Dense_0']['kernel'])
            t_dL = ravel_pytree(t_loss)[0]
        p_next = p - lr * t_dL
        return p_next, lr, e_acc, eig, eign, eigt, eignt, jnp.any(jnp.isnan(t_L))

    def _track_prestep_stats(self, p, lr, t_L, t_dL, U, Un, Ut, Unt, step, in_eos):

        self._track_ntk_stats(p, step)

        if not self.experimental or step % self.exp_period == 0:

            td, ed = self.train_data, self.test_data
            normalized_p = p / jla.norm(p)

            stats = {'weights': {'avg_norm': jla.norm(p)}}
            p_list = list(flatten_tree(self.loss.unravel_p(p).unfreeze()))
            for name, param in p_list:
                stats['weights'][name] = jla.norm(param)

            t_acc = self.loss.acc(p, data=td)
            e_L, e_dL = self.loss.value_and_grad(p, data=ed)
            e_acc = self.loss.acc(p, data=ed)
            # Not tracking these for now as we don't have any regularization (vanilla GD)!
            # t_rL, t_rdL = self.loss.r_value_and_grad(p, data=td)
            # e_rL, e_rdL = self.loss.r_value_and_grad(p, data=ed)

            if self.train_config.get('skip_sharpness', False):
                S, Sn, St, Snt = 0, 0, 0, 0
            else:
                (S,), U = self.loss.eig(p, U[:, :1], data=td)
                (Sn,), Un = self.loss.eig(normalized_p, Un[:, :1], data=td)
                (St,), Ut = self.loss.eig(p, Ut[:, :1], data=ed)
                (Snt,), Unt = self.loss.eig(normalized_p, Unt[:, :1], data=ed)

            if self.is_linear_regression:
                A = 0
            else:
                A = self._to_linear_model(p)

            # import IPython; IPython.embed()  # todo: test the values used in stats to see if they're arrays

            stats['train'] = dict(
                lr=lr,
                s1=S, sn1=Sn,
                se1=St, sne1=Snt,
                t_L=t_L, e_L=e_L,
                # t_rL=t_rL, e_rL=e_rL,
                t_acc=t_acc, e_acc=e_acc,
                t_dL=jla.norm(t_dL), e_dL=jla.norm(e_dL),
                te_dl=jnp.inner(t_dL, e_dL),
                uut=U.T @ Ut,
                linear_norm=jla.norm(A),
                # t_rdL=jla.norm(t_rdL), e_rdL=jla.norm(e_rdL),
            )

            write_stats(stats, self.writer, step)
            if (not self.experimental and step % 100 == 0) or step % self.exp_period == 0 or self.verbose:
                print(f"St:{step}, L={t_L:.6f}, eL={e_L:.6f} S={S:.5f}/{2 / lr:.5f}, "f"Ac={t_acc:.4f}, EAc={e_acc:.4f}")

        else:
            S, Sn, e_acc = 0, 0, 0
            St, Snt = 0, 0

        if in_eos:
            self._track_eos_stats()

        return (S, U), (Sn, Un), (St, Ut), (Snt, Unt), e_acc

    def _track_ntk_stats(self, p, step):

        def track_stats():
            # Todo: track all ntk stats
            """
            We don't need to do it rn! We've seen these stuff already! The most interesting thing
            would probably be the trace of hessian, which we can't precisely compute!!
            """
            pass  # Fuck the load times dude
            # self._build_quadratic_features()
            # if self.is_quadratic:
            #     return
            # params = self.loss.unravel_p(p)
            # k_train_train = self.ntk_fn(self.train_data[0], None, params)
            # k_test_train = self.ntk_fn(self.test_data[0], self.train_data[0], params)
            # predict_fn = nt.predict.gradient_descent_mse(k_train_train, self.Y)
            # fx_train_0 = self.model.apply(params, self.train_data[0])
            # fx_test_0 = self.model.apply(params, self.test_data[0])
            # pred_train, pred_test = predict_fn(None, fx_train_0, fx_test_0, k_test_train)
            # stats = {'ntk': dict(
            #     t_L=self._mse_loss(pred_train, self.Y),
            #     e_L=self._mse_loss(pred_test, self.Y_test),
            #     t_acc=self._acc(pred_train, self.Y),
            #     e_acc=self._acc(pred_test, self.Y_test),
            #     norm_train=jla.norm(k_train_train),
            #     norm_test=jla.norm(k_test_train),
            # )}
            # write_stats(stats, self.writer, step)

        # if self._should_track_measures(step):
        #     track_stats()
        # else:
        #     if self.grokking_config.get('schedule') == 'periodic':
        #         if step % self.grokking_config['period'] == 0:
        #             track_stats()
        #     else:
        #         if step in self.grokking_config['target_epochs']:
        #             track_stats()

        if step == 0:
            track_stats()

    def _track_eos_stats(self):
        # Todo: track eos stats
        """
        Let's not track anything for now! We don't really need to check if it matches the predicted
        dynamics or not! What we need to realize is eventually why (if) weight decay causes convergence
        and no weight decay doesn't! Towards this, maybe the first step would be to see if theta_dagger
        actually has reduced test loss when using WD vs when not using it, and realizing why is that.
        We don't need to predict the dynamics, for now!
        """
        pass

    def _track_second_eig(self, p, U, step):
        eigs, U = self.loss.eig(p, U, data=self.train_data)
        stats = {'train': {'s1': eigs[0], 's2': eigs[1]}}
        write_stats(stats, self.writer, step)
        return eigs, U

    def _to_quadratic_data(self, X):
        return jnp.einsum('ij,ik->ijk', X, X).reshape(X.shape[0], -1)

    def _to_linear_model(self, p):
        params = self.loss.unravel_p(p)
        name = self.model_config['name']
        W = params['params']['Dense_0']['kernel'].T
        V = params['params']['Dense_1']['kernel'].T
        if name == 'gromov_mlp':  # NTK Parameterization, update scales
            W = W / jnp.sqrt(W.shape[1])  # Fixed on April 26, didn't have sqrt before
            V = V / jnp.sqrt(V.shape[0])
        elif name == "mlp":  # Standard Parameterization, don't do anything
            pass
        else:
            raise NotImplementedError()
        if self.is_quadratic:
            A = V @ W
        else:
            C = V.shape[0]
            D = W.shape[1]
            A = np.zeros((D**2, C))
            for i in range(C):
                A[:, i] = (W.T @ jnp.diag(V[i]) @ W).reshape(-1,)
            # A = np.zeros((C, D, D))
            # for k in range(D):
            #     A[:, k] = V @ (W.T[k] * W.T).T
            # A = A.reshape(C, D ** 2).T
        return A

    def track_linear_regression_stats(self):
        self._build_quadratic_features()
        w = np.linalg.pinv(self.X) @ self.Y
        pred_train = self.X @  w
        pred_test = self.X_test @  w
        stats = {'linear': dict(
            t_L=self._mse_loss(pred_train, self.Y),
            e_L=self._mse_loss(pred_test, self.Y_test),
            t_acc=self._acc(pred_train, self.Y),
            e_acc=self._acc(pred_test, self.Y_test),
            norm=jla.norm(w),
        )}
        write_stats(stats, self.writer, 0)

    def run(self):

        """
        Probably the best way would be to have a unified training loop and 
        track different things based on the stage of training. 
        """

        self._build()
        steps = self.train_config['steps']
        p = self.p.astype(jnp.float32)
        if self.train_config.get('first_layer_only', False) and self.train_config.get('frei_init', False):
            params = self.loss.unravel_p(p).unfreeze()
            V = params['params']['Dense_1']['kernel']
            V = ((np.random.uniform(-1, 1, V.shape) > 0).astype(int) * 2 - 1) / jnp.sqrt(max(V.shape))  # assuming h > c
            params['params']['Dense_1']['kernel'] = V
            p = ravel_pytree(params)[0]
        p0 = jnp.copy(p)

        U = random.normal(self.eig_key, (len(p), 2), dtype=p.dtype)
        Ut = random.normal(self.eig_key, (len(p), 2), dtype=p.dtype)
        Un = random.normal(self.eig_key, (len(p), 2), dtype=p.dtype)
        Unt = random.normal(self.eig_key, (len(p), 2), dtype=p.dtype)

        if self.train_config.get('manual_investigation', False):
            import IPython; IPython.embed()

            # Computing jacobians

            # N = len(self.train_data[0])
            # batch_size = 512
            # jacs = []
            # for i in range(N//batch_size + 1):
            #     begin_ind = i * batch_size
            #     end_ind = (i+1) * batch_size
            #     if begin_ind > N - 1:
            #         break
            #     data = self.train_data[0][begin_ind:end_ind], self.train_data[1][begin_ind:end_ind]
            #     f = partial(self.loss.individual_loss, data=data)
            #     jacs.append(jax.jacobian(f)(p))
            #
            # N = len(self.test_data[0])
            # batch_size = 512
            # jacs = []
            # for i in range(N//batch_size + 1):
            #     begin_ind = i * batch_size
            #     end_ind = (i+1) * batch_size
            #     if begin_ind > N - 1:
            #         break
            #     data = self.test_data[0][begin_ind:end_ind], self.test_data[1][begin_ind:end_ind]
            #     f = partial(self.loss.individual_loss, data=data)
            #     jacs.append(jax.jacobian(f)(p))

            # Computing linear model's norm

        # Don't have to! We know that it won't converge! We can check later if we want to
        # self.track_linear_regression_stats()

        step, t, in_eos = 0, 0, False
        next_sharpness_check, dt, check_sharpness = 0.0, 1.0, False
        """
        For now, let's track sharpness in every step and see how slow it is! 
        If it's not that slow, we shouldn't worry about it that much!
        """
        print('Starting training...')
        while step < steps:
            p, lr, e_acc, eig, eign, eigt, eignt, broken_training = \
                self._gd_step(p, U[:, :1], Un[:, :1], Ut[:, :1], Unt[:, :1], step, in_eos)
            U = U.at[:, :1].set(eig[1])
            Un = Un.at[:, :1].set(eign[1])
            Ut = Ut.at[:, :1].set(eigt[1])
            Unt = Unt.at[:, :1].set(eignt[1])
            t += lr
            # if e_acc >= .999: # Todo: enable later
            #     import IPython; IPython.embed()
            #     print('Model is generalizing, stopping training...')
            #
            #     break
            if broken_training:
                print(f'Broken training at step {step}, stopping training...')
                break
            # if not in_eos:
            #     if eig[0] >= 2. / lr:
            #         in_eos = True
            #         p = p.astype(jnp.float64)
            #         U = U.astype(jnp.float64)
            #         Un = Un.astype(jnp.float64)
            #         # Todo: is this enough? I thought we need to change type and dtype of flax computations as well!
            # else:
            #     # But let's check the second eigenvalue sometimes!
            #     # if step % self.train_config.get('second_eig_period', 50) == 0:
            #     #     _, U = self._track_second_eig(p, U, step)
            #     pass
            step += 1
