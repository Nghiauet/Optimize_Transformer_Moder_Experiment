from transformers import TrainingArguments 
import torch.nn as nn
import torch.nn.functional as F 
from transformers import Trainer

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature = 2.0, **kwargs): 
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
class DistillationTrainer(Trainer):
    def __init__(self,*arg, teacher_model = None, **kwargs):
        super().__init__(*arg, **kwargs)
        self.teacher_model = teacher_model 
    def compute_loss(self, model, inputs, return_outputs=False):
        # extract loss from stuent 
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits
        student_loss = student_outputs.loss
        # extract loss from teacher
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        klloss = nn.KLDivLoss(reduction="batchmean", log_target=False)
        loss_kd = klloss(
            F.log_softmax(student_logits / self.args.temperature, dim=-1),
            F.softmax(teacher_logits / self.args.temperature, dim=-1),
        ) * (self.args.temperature ** 2) # scale the loss by temperature
        # return weight student loss and kd loss
        loss = (1 - self.args.alpha) * student_loss + self.args.alpha * loss_kd
        return (loss, student_outputs) if return_outputs else loss