import json

import wandb

from vidur.logger import init_logger

logger = init_logger(__name__)


class SeriesAverageMeter:
    def __init__(
        self,
        x_name: str,
        y_name: str,
        use_weighted_mean: bool = True,
        save_table_to_wandb: bool = True,
    ) -> None:
        # column names of x, y datatpoints for data collection
        self._distribution = []  # Add a list to store the distribution
        self._x_name = x_name
        self._y_name = y_name
        self._use_weighted_mean = use_weighted_mean
        self._save_table_to_wandb = save_table_to_wandb

        self._denom_sum = 0
        self._numer_sum = 0
        self._min_y = float("inf")
        self._max_y = float("-inf")
        # most recently collected y datapoint for incremental updates
        # to aid incremental updates to y datapoints
        self._last_data_y = None
        self._last_data_x = None

    def _update_simple_mean(self, data_y: float) -> None:
        self._denom_sum += 1
        self._numer_sum += data_y

    def _update_weighted_mean(self, data_x: float) -> None:
        if not self._last_data_x:
            return

        # compute the segment length
        x_diff = data_x - self._last_data_x
        # Add the weighted value multiplied by the time difference to the total weighted sum
        self._numer_sum += self._last_data_y * x_diff
        # Add the time difference to the total time
        self._denom_sum += x_diff

    # add a new x, y datapoint
    def put(self, data_x: float, data_y: float, batch_id: int = None, batch_stage_id: int = None) -> None:
        if self._use_weighted_mean:
            self._update_weighted_mean(data_x)
        else:
            self._update_simple_mean(data_y)

        # Log data into the distribution list
        if batch_id is None or batch_stage_id is None:
            logger.warning(f"Skipping logging due to uninitialized batch or batch stage.")
            return
        logger.debug(f"Logging data: time={data_x}, mfu={data_y}, batch_id={batch_id}, batch_stage_id={batch_stage_id}")

        self._distribution.append({
            "time": data_x,
            "mfu": data_y,
            "batch_id": batch_id,
            "batch_stage_id": batch_stage_id
        })
        
        self._last_data_y = data_y
        self._last_data_x = data_x
        self._min_y = min(self._min_y, data_y)
        self._max_y = max(self._max_y, data_y)


    # get most recently collected y datapoint
    def _peek_y(self):
        return self._last_data_y

    # add a new x, y datapoint as an incremental (delta) update to
    # recently collected y datapoint
    def put_delta(self, data_x: float, data_y_delta: float) -> None:
        last_data_y = self._peek_y()
        data_y = last_data_y + data_y_delta
        self.put(data_x, data_y)

    def print_stats(self, name: str, path: str) -> None:
        if self._denom_sum == 0:
            return

        weighted_mean = self._numer_sum / self._denom_sum

        logger.debug(
            f"{name}: {self._y_name} stats:"
            f" min: {self._min_y},"
            f" max: {self._max_y},"
            f" weighted_mean: {weighted_mean}"
        )

        stats_dict = {
            f"{name}_min": self._min_y,
            f"{name}_max": self._max_y,
            f"{name}_weighted_mean": weighted_mean,
            f"{name}_distribution": [
                {
                    "time": data["time"],
                    "mfu": data["mfu"],
                    "batch_id": data.get("batch_id"),
                    "batch_stage_id": data.get("batch_stage_id")
                }
                for data in self._distribution
            ]
        }

        with open(f"{path}/{name}.json", "w") as f:
            json.dump(stats_dict, f)

        if wandb.run:
            wandb.log(
                {
                    f"{name}_min": self._min_y,
                    f"{name}_max": self._max_y,
                    f"{name}_weighted_mean": weighted_mean,
                },
                step=0,
            )