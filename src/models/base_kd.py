from .modules.optimizer import OptimizerMixin
from .modules import loss_functions
from pytorch_lightning import LightningModule
import torch
import torchmetrics
from .modules import pooling


class GeneralDistillation(OptimizerMixin, LightningModule):
    def __init__(
        self,
        student: LightningModule,
        teacher: LightningModule,
        temperature: float = 1.0,
        alpha: float = 0.5,
        loss: str = "loss_fn_kd",
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.teacher = self.hparams.teacher
        self.student = self.hparams.student

        self.loss_function = getattr(loss_functions, self.hparams.loss)

        self.val_mse = torchmetrics.MeanSquaredError()
        self.test_mse = torchmetrics.MeanSquaredError()

    def forward(self, batch: dict) -> torch.Tensor:
        """Used for inference only.

        Args:
            batch ():

        Returns:

        """

        return None

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training Loop.

        Args:
            batch ():
            batch_idx ():

        Returns:

        """

        # Calculate Student Outputs
        student_outputs = self.student(batch)  # (bs, seq_length, voc_size)

        # Calculate Teacher Outputs (don't need gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher(batch)  # (bs, seq_length, voc_size)

        # Get Logits and Hidden States
        student_logits, student_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        teacher_logits, teacher_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]

        #student_embedding = self.student.get_embedding(student_outputs["hidden_states"], batch["attention_mask"])
        #teacher_embedding = self.teacher.get_embedding(teacher_outputs["hidden_states"], batch["attention_mask"])

        # Check output sizes
        assert student_logits.size() == teacher_logits.size()

        # Calculate the Loss
        loss = self.loss_function(student_logits, teacher_logits, batch["labels"], self.hparams)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):

        # Calculate Student Outputs
        student_outputs = self.student(batch)  # (bs, seq_length, voc_size)

        # Calculate Teacher Outputs (don't need gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher(batch)  # (bs, seq_length, voc_size)

        # Get Logits and Hidden States
        student_logits, student_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        teacher_logits, teacher_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]

        #student_embedding = self.get_embedding(student_outputs["hidden_states"], batch["attention_mask"])
        #teacher_embedding = self.get_embedding(teacher_outputs["hidden_states"], batch["attention_mask"])

        # Check output sizes
        assert student_logits.size() == teacher_logits.size()

        # TODO: Change Evaluation Metric
        self.val_mse(student_logits, teacher_logits)

    def validation_epoch_end(self, outputs):
        val_mse = self.val_mse.compute()

        # TODO: Change Evaluation Metric
        self.log("val/mse", val_mse, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # Calculate Student Outputs
        student_outputs = self.student(batch)  # (bs, seq_length, voc_size)

        # Calculate Teacher Outputs (don't need gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher(batch)  # (bs, seq_length, voc_size)

        # Get Logits and Hidden States
        student_logits, student_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        teacher_logits, teacher_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]

        # student_embedding = self.get_embedding(student_outputs["hidden_states"], batch["attention_mask"])
        # teacher_embedding = self.get_embedding(teacher_outputs["hidden_states"], batch["attention_mask"])

        # Check output sizes
        assert student_logits.size() == teacher_logits.size()

        # TODO: Change Evaluation Metric
        self.test_mse(student_logits, teacher_logits)

    def test_epoch_end(self, outputs):
        test_mse = self.test_mse.compute()

        # TODO: Change Evaluation Metric 
        self.log("test/mse", test_mse, prog_bar=True)


