import torchvision.models as models
import torch

model = models.resnet152()
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

