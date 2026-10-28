from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from .actor import EVANDataParallelPPOActor


class EVANActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            self.actor = EVANDataParallelPPOActor(
                config=self.actor.config,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )
