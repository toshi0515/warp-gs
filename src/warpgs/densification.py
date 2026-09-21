"""Small adapter around gsplat's densification strategies."""

from typing import Any

import torch

from warpgs.model import GaussianModel, OptimizerDict
from warpgs.renderers import RenderOutput, ensure_gsplat_available


class Densification:
    def __init__(self, renderer: str, kind: str, args) -> None:
        if renderer == "warp" and kind == "default" and args.absgrad:
            raise ValueError(
                "warp + default does not support --absgrad. "
                "Use the standard gradient or select the gsplat renderer."
            )

        ensure_gsplat_available()
        self.kind = kind
        self.name = f"gsplat-{kind}"
        self.packed = args.packed if renderer == "gsplat" else False
        self.opacity_reg = args.opacity_reg if kind == "mcmc" else 0.0
        self.scale_reg = args.scale_reg if kind == "mcmc" else 0.0
        self.noise_until = args.mcmc_noise_until
        self.strategy: Any

        if kind == "default":
            from gsplat.strategy import DefaultStrategy

            self.strategy = DefaultStrategy(
                prune_opa=args.opacity_threshold,
                grow_grad2d=args.grad_threshold,
                refine_start_iter=max(0, args.densify_from - 1),
                refine_stop_iter=args.densify_until + 1,
                refine_every=args.densify_interval,
                absgrad=args.absgrad,
                verbose=True,
            )
            self.state = self.strategy.initialize_state(scene_scale=1.5)
        else:
            from gsplat.strategy import MCMCStrategy

            self.strategy = MCMCStrategy(
                cap_max=args.max_gaussians,
                noise_lr=args.mcmc_noise_lr,
                refine_start_iter=max(0, args.densify_from - 1),
                refine_stop_iter=args.densify_until + 1,
                refine_every=args.densify_interval,
                min_opacity=args.opacity_threshold,
                verbose=True,
            )
            self.state = self.strategy.initialize_state()

    def check_sanity(self, model: GaussianModel, optimizers: OptimizerDict) -> None:
        self.strategy.check_sanity(model.params, optimizers)

    def regularization(self, model: GaussianModel) -> torch.Tensor:
        if self.kind == "default":
            return model.means.new_zeros(())
        return self.opacity_reg * model.opacities.mean() + self.scale_reg * model.scales.mean()

    def before_backward(
        self,
        model: GaussianModel,
        optimizers: OptimizerDict,
        step: int,
        output: RenderOutput,
    ) -> None:
        if self.kind == "default":
            self.strategy.step_pre_backward(model.params, optimizers, self.state, step, output.info)

    def after_backward(
        self,
        model: GaussianModel,
        optimizers: OptimizerDict,
        step: int,
        output: RenderOutput,
    ) -> None:
        if self.kind == "default":
            self.strategy.step_post_backward(
                model.params,
                optimizers,
                self.state,
                step,
                output.info,
                packed=self.packed,
            )
        else:
            means_lr = float(optimizers["means"].param_groups[0]["lr"])
            self.strategy.step_post_backward(
                model.params,
                optimizers,
                self.state,
                step,
                output.info,
                lr=means_lr if step <= self.noise_until else 0.0,
            )
