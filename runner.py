import os

from catalyst import dl


class MultiheadClassificationRunner(dl.Runner):
    def __init__(self, num_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def handle_batch(self, batch):
        x, targets = batch
        logits = self.model(x)
        
        batch_dict = { "features": x }
        for i in range(self.num_heads):
            batch_dict[f"targets{i}"] = targets[i]
        for i in range(self.num_heads):
            batch_dict[f"logits{i}"] = logits[i]
        
        self.batch = batch_dict

def get_runner_callbacks(num_heads, num_classes_per_head, class_names, logdir):
    cbs = [
        *[
            dl.CriterionCallback(
                metric_key=f"loss{i}",
                input_key=f"logits{i}",
                target_key=f"targets{i}"
            )
            for i in range(num_heads)
        ],
        dl.MetricAggregationCallback(
            metric_key="loss",
            metrics=[f"loss{i}" for i in range(num_heads)],
            mode="mean"
        ),
        dl.OptimizerCallback(metric_key="loss"),
        dl.SchedulerCallback(),
        *[
            dl.AccuracyCallback(
                input_key=f"logits{i}",
                target_key=f"targets{i}",
                num_classes=num_classes_per_head,
                suffix=f"{i}"
            )
            for i in range(num_heads)
        ],
        dl.CheckpointCallback(
            logdir=os.path.join(logdir, "checkpoints"),
            loader_key="valid",
            metric_key="loss",
            minimize=True,
            save_n_best=1
        )
    ]
    
    return cbs
