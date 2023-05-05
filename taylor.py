from collections import namedtuple
from util import tree_stack
from tqdm.auto import tqdm
from jax import numpy as jnp
from jax.numpy import linalg as jla
from tqdm import trange
from math import ceil
from util import power_iter, tree_save
from jax.tree_util import tree_map
import time
from functools import wraps


def find_instability(p, U, lr, loss, writer):
    pbar = tqdm()
    save_list = []
    t = 0
    prev = (t, p, jnp.inf)
    next_check = 0
    dt = 1
    while True:
        L, dL = loss.value_and_grad(p)
        acc = loss.acc(p)
        RL = loss.lossreg(p)
        RdL_norm = jnp.linalg.norm(loss.RD(p, 1))
        if t >= next_check:
            p_next = p - lr * dL
            (S,), U = loss.eig((p + p_next) / 2, U)
            save_list.append(dict(t=t, L=L, S=S))
            pbar.set_description(
                f"t={t:.2f}, next={next_check:.2f}, L={L:.6f}, S={S:.2f}/{2/lr:.2f}, "
                f"Acc={acc:.2f}, RL: {RL:.6f}, RdL: {RdL_norm:.6f}"
            )
            pbar.refresh()
            if S >= 2 / lr or jnp.abs(L - 0.0) < 5e-4:
                if dt <= lr:
                    break
                else:
                    print(
                        f"\nBacktracking to t={prev[0]:.2f} where sharpness was {prev[2]:.2f}"
                    )
                    t, p, _ = prev
                    dt = 0
                    next_check = t
                    continue
            if S >= 1.5 / lr:
                dt = min(dt, 0.01)
            if S >= 1 / lr:
                dt = min(dt, 0.1)
            next_check = t + dt
            prev = (t, p, S)
        pbar.set_description(f"t={t:.2f}, next={next_check:.2f}, L={L:.6f}, S={S:.2f}/{2/lr:.2f}, "
                             f"Acc={acc:.2f}, RL: {RL:.6f}, RdL: {RdL_norm:.6f}")
        t += lr
        p = p - lr * dL
        pbar.update()
    pbar.set_description(f"t={t:.2f}, L={L:.6f}, S={S:.2f}/{2/lr:.2f}")
    pbar.close()
    return p, U, tree_stack(save_list)


TaylorCenter = namedtuple("TaylorCenter", "p L dL S u dS")
EigenSystem = namedtuple("EigenSystem", "p S u")


def track_dynamics(
    p, ref_U, ref_Ub, lr, loss, steps, num_proj_steps, generalized_pred=False, save_dir=None, writer=None
):
    def check_eigengap(T: TaylorCenter, ref_U):
        p, L, dL, S, u, dS = T
        U = ref_U.at[:, 0].set(u)
        eigs, U = loss.eig(p, U)
        return eigs, U

    def taylor_center(p, ref_u):
        L, dL = loss.value_and_grad(p)
        S, u = loss.eig(p, ref_u)
        dS = loss.D(p, 3, u, u)
        return TaylorCenter(p, L, dL, S, u, dS)

    def linear_project(T: TaylorCenter):
        p, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)  # What about the scale?
        S_proj = -max(0, S - 2 / lr) * dS_perp / (dS_perp @ dS_perp)
        newton_proj = -u * ((u @ dL) / S)
        return taylor_center(p + S_proj + newton_proj, T.u)

    def project(T: TaylorCenter):
        for _ in range(num_proj_steps):
            T = linear_project(T)
        return T

    def constrained_step(T: TaylorCenter):
        return project(taylor_center(T.p - lr * T.dL, T.u))

    def gd_step(p):
        return p - lr * loss.grad(p)

    def gf_step(E: EigenSystem):
        p, S, u = E
        sub_steps = int(ceil(lr * S))
        for _ in range(sub_steps):
            p = p - (lr / sub_steps) * loss.grad(p.astype(jnp.float32)).astype(
                jnp.float64
            )
        S, u = loss.eig(p, u)
        return EigenSystem(p, S, u)

    def predicted_step(v, T: TaylorCenter):
        p, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        alpha = -dL @ dS_perp
        beta = dS_perp @ dS_perp
        delta = jnp.sqrt(2 * alpha / beta)

        x = u @ v
        y = dS_perp @ v
        v_perp = v - x * u
        if not generalized_pred:
            grad_terms = [
                (x**2 - delta**2) * dS_perp / 2,
                loss.D(p, 2, v),
                u * x * y,
            ]
        else:
            delta_dL = u * (u @ loss.grad(p + x * u))
            grad_terms = [
                (x**2 - delta**2) * dS_perp / 2,
                loss.D(p, 2, v_perp),
                delta_dL,
                u * x * y,
            ]
        v = v - lr * sum(grad_terms)
        return v

    def taylor_stats_fn(p, T: TaylorCenter):
        ref, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        v = p - ref

        def d2L_ratio(*vs):
            vs = tree_map(lambda v: v - u * (u @ v), vs)
            vs = tree_map(lambda v: v / jla.norm(v), vs)
            return jla.norm(loss.D(ref, 2, *vs))

        def d3L_ratio(*vs):
            vs = tree_map(lambda v: v - u * (u @ v), vs)
            vs = tree_map(lambda v: v / jla.norm(v), vs)
            return jla.norm(loss.D(ref, 3, *vs))

        output = dict(
            L=loss.value(p),
            S_taylor=S + dS @ v,
            S_avg=S + dS_perp @ v,
            S_exact=loss.eig(p, T.u)[0],
            x=u @ v,
            # d2L_v_ratio=d2L_ratio(v),
            # d2L_dL_ratio=d2L_ratio(dL),
            # d2L_dS_ratio=d2L_ratio(dS),
            d2L_v_v_ratio=d2L_ratio(v, v),
            # d2L_v_dL_ratio=d2L_ratio(v, dL),
            # d2L_v_dS_ratio=d2L_ratio(v, dS),
            # d2L_dL_dL_ratio=d2L_ratio(dL, dL),
            # d2L_dL_dS_ratio=d2L_ratio(dL, dS),
            # d2L_dS_dS_ratio=d2L_ratio(dS, dS),
            d3L_v_ratio=d3L_ratio(v, v),
            d3L_dL_ratio=d3L_ratio(dL, dL),
            dist=jla.norm(v),
        )
        return output

    def gf_stats_fn(E: EigenSystem, T: TaylorCenter):
        output = dict(
            L=loss.value(E.p),
            S_exact=E.S,
            dist=jla.norm(E.p - T.p),
        )
        return output

    def constants_fn(T: TaylorCenter):
        ref, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        alpha = -dL @ dS_perp
        beta = dS_perp @ dS_perp
        gamma = alpha / (jla.norm(dL) * jla.norm(dS_perp))
        delta = jnp.sqrt(2 * alpha / beta)
        dL_perp = dL - u * (u @ dL) - dS_perp * (dS_perp @ dL) / beta
        newton_x = u @ dL / S
        output = dict(
            L=L,
            S=S,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            newton_x=newton_x,
            dL_norm=jla.norm(dL),
            dL_perp_norm=jla.norm(dL_perp),
        )
        return output

    def update_betas(betas, T: TaylorCenter, step, save_step=False):
        p, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        for t, (rot, l) in betas.items():
            l.append(dS_perp @ rot)
            rot = rot - u * (u @ rot)
            rot = rot - lr * loss.D(T.p, 2, rot)
            betas[t] = (rot, l)
        if save_step:
            betas[step] = (dS_perp, [])
        return betas

    def slow_stats_fn(T: TaylorCenter):
        p, L, dL, S, u, dS = T
        dS_perp = dS - u * (u @ dS)
        alpha = -dL @ dS
        beta = dS_perp @ dS_perp
        delta = jnp.sqrt(2 * alpha / beta)
        xs = jnp.linspace(-5 * delta, 5 * delta, 100)
        F = jnp.array([loss.value(p + x * u) for x in xs])
        rho3 = power_iter(lambda v: loss.D(p, 3, v, v), u)[0]
        rho4 = power_iter(lambda v: loss.D(p, 4, v, v, v), u)[0]
        return dict(F=(xs, F), rho3=rho3, rho4=rho4)

    def write_stats(d, writer, step):
        for top_k in d.keys():
            for k in d[top_k].keys():
                val = d[top_k][k]
                val = val.item() if hasattr(val, 'item') else val
                writer.add_scalar(top_k + '/' + k, val, global_step=step)

    ref_u = ref_U[:, 0]
    dagger = project(taylor_center(p, ref_u))
    gd = p
    gf = EigenSystem(p, 2 / lr, ref_u)
    v = gd - dagger.p
    save_list = []
    betas = {}
    slow_stats = {}
    pbar = trange(steps)
    for i in pbar:
        stats = dict(
            gd=taylor_stats_fn(gd, dagger),
            pred=taylor_stats_fn(dagger.p + v, dagger),
            gf=gf_stats_fn(gf, dagger),
            dagger=constants_fn(dagger),
        )
        stats['gd'].update(dict(
            acc=loss.acc(gd),
            zero_reg_loss=loss.lossreg(gd),
            p_norm=jla.norm(gd),
            dl_norm=jla.norm(loss.D(gd, 1)),
            rdl_norm=jla.norm(loss.RD(gd, 1)),
        ))
        stats['dagger'].update(dict(
            rdl_norm=jla.norm(loss.RD(dagger.p, 1)),
        ))
        write_stats(stats, writer, i)
        save_list.append(stats)
        if i % 50 == 0:
            ref_U = ref_U.at[:, 0].set(dagger.u)
            eigs, ref_U = check_eigengap(dagger, ref_U)
            betas = update_betas(betas, dagger, step=i, save_step=True)
            slow_stats[i] = slow_stats_fn(dagger)
            writer.add_scalar('gd/rho3', slow_stats[i]['rho3'].item(), global_step=i)
            writer.add_scalar('gd/rho4', slow_stats[i]['rho4'].item(), global_step=i)
            # import IPython; IPython.embed()
            if eigs[1] > 1.9 / lr:
                print(f"Step = {i}, eigenvalues at theta dagger: {eigs}")
                print("Second largest eigenvalue exceeded threshold")
                break
            # dynamics = tree_stack(save_list)
            # dynamics["beta"] = {key: jnp.array(val[1]) for key, val in betas.items()}
            # dynamics["slow_stats"] = slow_stats
            # tree_save(dynamics, save_dir + "/dynamics.pytree", overwrite=True)
        else:
            betas = update_betas(betas, dagger, step=i, save_step=False)


        ## My stats:
        def project_onto(a, b):
            return b * (b.T @ a) / (b.T @ b)

        t = taylor_center(gd, ref_u)
        v_pu = project_onto(v, dagger.u)
        v_cu = v - v_pu
        v_pp = project_onto(v, dagger.p)
        v_cp = v - v_pp
        v_pdS = project_onto(v, dagger.dS)
        v_cdS = v - v_pdS
        p_pu = project_onto(dagger.p, dagger.u)
        p_cu = dagger.p - p_pu
        dL = loss.grad(dagger.p)
        dL_pp = project_onto(dL, dagger.p)
        dL_cp = dL - dL_pp
        dL_pu = project_onto(dL, dagger.u)
        dL_cu = dL - dL_pu
        d3L_vpu = loss.D(dagger.p, 3, v_pu, v_pu)
        d3L_vcu = loss.D(dagger.p, 3, v_cu, v_cu)
        d3L_vpp = loss.D(dagger.p, 3, v_pp, v_pp)
        d3L_vcp = loss.D(dagger.p, 3, v_cp, v_cp)
        dS_pu = project_onto(dagger.dS, dagger.u)
        dS_cu = dagger.dS - dS_pu
        dS_pp = project_onto(dagger.dS, dagger.p)
        dS_cp = dagger.dS - dS_pp
        stats['in'] = dict(
            v=v,
            v_pu=v_pu,
            v_cu=v_cu,
            v_pp=v_pp,
            v_cp=v_cp,
            v_pdS=v_pdS,
            v_cdS=v_cdS,
            p_pu=p_pu,
            p_cu=p_cu,
            dL_pp=dL_pp,
            dL_cp=dL_cp,
            dL_pu=dL_pu,
            dL_cu=dL_cu,
            d3L_vpu=d3L_vpu,
            d3L_vcu=d3L_vcu,
            d3L_vpp=d3L_vpp,
            d3L_vcp=d3L_vcp,
            dS_pu=dS_pu,
            dS_cu=dS_cu,
            dS_pp=dS_pp,
            dS_cp=dS_cp
        )
        for k in stats['in'].keys():
            stats['in'][k] = jla.norm(stats['in'][k])

        u_align = t.u @ dagger.u
        ds_align = t.dS @ dagger.dS / (jla.norm(t.dS) * jla.norm(dagger.dS))
        d2L_vpu = loss.D(dagger.p, 2, v_pu, v_pu)
        d2L_cpu = loss.D(dagger.p, 2, v_cu, v_cu)
        d2L_vpp = loss.D(dagger.p, 2, v_pp, v_pp)
        d2L_vcp = loss.D(dagger.p, 2, v_cp, v_cp)
        stats['in'].update(dict(
            u_align=u_align,
            d2L_vpu=d2L_vpu,
            d2L_cpu=d2L_cpu,
            d2L_vpp=d2L_vpp,
            d2L_vcp=d2L_vcp,
            ds_align=ds_align,
        ))

        ref_U = ref_U.at[:, 0].set(dagger.u)
        eigs, ref_U = check_eigengap(dagger, ref_U)

        p_pu2 = project_onto(dagger.p, ref_U[:, 1])
        p_cu2 = dagger.p - p_pu2
        v_pu2 = project_onto(v, ref_U[:, 1])
        v_cu2 = v - v_pu2
        d3L_vpu2 = jla.norm(loss.D(dagger.p, 3, v_pu2, v_pu2))
        d3L_vcu2 = jla.norm(loss.D(dagger.p, 3, v_cu2, v_cu2))
        d2L_vpu2 = loss.D(dagger.p, 2, v_pu2, v_pu2)
        d2L_vcu2 = loss.D(dagger.p, 2, v_cu2, v_cu2)

        p_pu3 = project_onto(dagger.p, ref_U[:, 2])
        p_cu3 = dagger.p - p_pu3
        v_pu3 = project_onto(v, ref_U[:, 2])
        v_cu3 = v - v_pu3
        d3L_vpu3 = jla.norm(loss.D(dagger.p, 3, v_pu3, v_pu3))
        d3L_vcu3 = jla.norm(loss.D(dagger.p, 3, v_cu3, v_cu3))
        d2L_vpu3 = loss.D(dagger.p, 2, v_pu3, v_pu3)
        d2L_vcu3 = loss.D(dagger.p, 2, v_cu3, v_cu3)
        stats['eigs'] = dict(
            p_pu2=jla.norm(p_pu2),
            p_cu2=jla.norm(p_cu2),
            v_pu2=jla.norm(v_pu2),
            v_cu2=jla.norm(v_cu2),
            d3L_vpu2=d3L_vpu2,
            d3L_vcu2=d3L_vcu2,
            d2L_vpu2=d2L_vpu2,
            d2L_vcu2=d2L_vcu2,
            p_pu3=jla.norm(p_pu3),
            p_cu3=jla.norm(p_cu3),
            v_pu3=jla.norm(v_pu3),
            v_cu3=jla.norm(v_cu3),
            d3L_vpu3=d3L_vpu3,
            d3L_vcu3=d3L_vcu3,
            d2L_vpu3=d2L_vpu3,
            d2L_vcu3=d2L_vcu3,
            s1=eigs[0],
            s2=eigs[1],
            s3=eigs[2]
        )

        eigsb, ref_Ub = loss.eig(dagger.p, ref_Ub, largest=False)

        p_pub = project_onto(dagger.p, ref_Ub[:, 0])
        p_cub = dagger.p - p_pub
        v_pub = project_onto(v, ref_Ub[:, 0])
        v_cub = v - v_pub
        d3L_vpub = jla.norm(loss.D(dagger.p, 3, v_pub, v_pub))
        d3L_vcub = jla.norm(loss.D(dagger.p, 3, v_cub, v_cub))
        d2L_vpub = loss.D(dagger.p, 2, v_pub, v_pub)
        d2L_vcub = loss.D(dagger.p, 2, v_cub, v_cub)

        p_pub2 = project_onto(dagger.p, ref_Ub[:, 1])
        p_cub2 = dagger.p - p_pub2
        v_pub2 = project_onto(v, ref_Ub[:, 1])
        v_cub2 = v - v_pub2
        d3L_vpub2 = jla.norm(loss.D(dagger.p, 3, v_pub2, v_pub2))
        d3L_vcub2 = jla.norm(loss.D(dagger.p, 3, v_cub2, v_cub2))
        d2L_vpub2 = loss.D(dagger.p, 2, v_pub2, v_pub2)
        d2L_vcub2 = loss.D(dagger.p, 2, v_cub2, v_cub2)
        stats['small_eigs'] = dict(
            p_pub=jla.norm(p_pub),
            p_cub=jla.norm(p_cub),
            v_pub=jla.norm(v_pub),
            v_cub=jla.norm(v_cub),
            d3L_vpub=d3L_vpub,
            d3L_vcub=d3L_vcub,
            d2L_vpub=d2L_vpub,
            d2L_vcub=d2L_vcub,
            p_pub2=jla.norm(p_pub2),
            p_cub2=jla.norm(p_cub2),
            v_pub2=jla.norm(v_pub2),
            v_cub2=jla.norm(v_cub2),
            d3L_vpub2=d3L_vpub2,
            d3L_vcub2=d3L_vcub2,
            d2L_vpub2=d2L_vpub2,
            d2L_vcub2=d2L_vcub2,
            sb1=eigsb[0],
            sb2=eigsb[1],
        )

        # v_pu = project_onto(v, t.u)
        # v_cu = v - v_pu
        # v_pp = project_onto(v, t.p)
        # v_cp = v - v_pp
        # v_pdS = project_onto(v, t.dS)
        # v_cdS = v - v_pdS
        # p_pu = project_onto(t.p, dagger.u)
        # p_cu = t.p - p_pu
        # dL = loss.grad(dagger.p)
        # dL_pp = project_onto(dL, t.p)
        # dL_cp = dL - dL_pp
        # dL_pu = project_onto(dL, t.u)
        # dL_cu = dL - dL_pu
        # d3L_vpu = loss.D(dagger.p, 3, v_pu, v_pu)
        # d3L_vcu = loss.D(dagger.p, 3, v_cu, v_cu)
        # d3L_vpp = loss.D(dagger.p, 3, v_pp, v_pp)
        # d3L_vcp = loss.D(dagger.p, 3, v_cp, v_cp)
        # dS_pu = project_onto(dagger.dS, dagger.u)
        # dS_cu = dagger.dS - dS_pu
        # dS_pp = project_onto(dagger.dS, dagger.p)
        # dS_cp = dagger.dS - dS_pp
        # stats['in'] = dict(
        #     v=v,
        #     v_pu=v_pu,
        #     v_cu=v_cu,
        #     v_pp=v_pp,
        #     v_cp=v_cp,
        #     v_pdS=v_pdS,
        #     v_cdS=v_cdS,
        #     p_pu=p_pu,
        #     p_cu=p_cu,
        #     dL_pp=dL_pp,
        #     dL_cp=dL_cp,
        #     dL_pu=dL_pu,
        #     dL_cu=dL_cu,
        #     d3L_vpu=d3L_vpu,
        #     d3L_vcu=d3L_vcu,
        #     d3L_vpp=d3L_vpp,
        #     d3L_vcp=d3L_vcp,
        #     dS_pu=dS_pu,
        #     dS_cu=dS_cu,
        #     dS_pp=dS_pp,
        #     dS_cp=dS_cp
        # )
        # for k in stats['in'].keys():
        #     stats['in'][k] = jla.norm(stats['in'][k])


        write_stats(stats, writer, i)

        def timed(f):
            @wraps(f)
            def _f(*args, **kwargs):
                start_time = time.perf_counter()
                out = f(*args, **kwargs)
                # tree_map(lambda x: x.block_until_ready(), out)
                end_time = time.perf_counter()
                return out, end_time - start_time

            return _f

        gd, gd_time = timed(gd_step)(gd)
        gf, gf_time = timed(gf_step)(gf)
        v, pred_time = timed(predicted_step)(v, dagger)
        dagger, dagger_time = timed(constrained_step)(dagger)
        pbar.set_description(
            f"gd: {gd_time:.1f}, gf: {gf_time:.1f}, pred: {pred_time:.1f}, dagger: {dagger_time:.1f}, "
            f"acc: {stats['gd']['acc']:.2f}, loss: {stats['gd']['L']:.4f}, regless_loss: {stats['gd']['zero_reg_loss']:.4f}"
        )
    # output = tree_stack(save_list)
    # output["beta"] = {key: jnp.array(val[1]) for key, val in betas.items()}
    # output["slow_stats"] = slow_stats
    # return output
