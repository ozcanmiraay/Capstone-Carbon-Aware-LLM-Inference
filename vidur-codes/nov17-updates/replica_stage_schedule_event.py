from typing import List

from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

logger = init_logger(__name__)


class ReplicaStageScheduleEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, stage_id: int):
        super().__init__(time, EventType.REPLICA_STAGE_SCHEDULE)

        self._replica_id = replica_id
        self._stage_id = stage_id

        self._batch = None
        self._batch_stage = None
        self._is_last_stage = None

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.batch_stage_end_event import BatchStageEndEvent

        stage_scheduler = scheduler._replica_schedulers[
            self._replica_id
        ]._replica_stage_schedulers[self._stage_id]

        # Attempt to schedule the batch and batch stage
        self._batch, self._batch_stage, execution_time = stage_scheduler.on_schedule()

        # Check for uninitialized batch or batch stage
        if not self._batch:
            logger.warning(
                f"Skipping event for replica {self._replica_id}, stage {self._stage_id}: Batch is not initialized."
            )
            return []

        if not self._batch_stage:
            logger.warning(
                f"Skipping event for replica {self._replica_id}, stage {self._stage_id}: BatchStage is not initialized."
            )
            return []

        # Debug log for successful initialization
        logger.debug(
            f"Replica ID: {self._replica_id}, Stage ID: {self._stage_id}, Batch: {self._batch}, Batch Stage: {self._batch_stage}"
        )

        # Trigger scheduling logic for the batch stage
        self._batch_stage.on_schedule(self.time)

        # Log MFU metrics via metrics_store
        metrics_store.on_replica_stage_schedule(
            self.time,
            self._replica_id,
            self._stage_id,
            self._batch_stage,
            execution_time,
        )

        # Determine if this is the last stage in the pipeline
        self._is_last_stage = stage_scheduler.is_last_stage

        # Schedule the next event
        return [
            BatchStageEndEvent(
                self.time + self._batch_stage.execution_time,
                self._replica_id,
                self._stage_id,
                self._is_last_stage,
                self._batch,
                self._batch_stage,
            ),
        ]

    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "replica_id": self._replica_id,
            "stage_id": self._stage_id,
            "batch_id": self._batch.id if self._batch else None,
            "batch_stage_id": self._batch_stage.id if self._batch_stage else None,
            "is_last_stage": self._is_last_stage,
        }
