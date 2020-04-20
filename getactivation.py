import numpy as np
from dnnbrain.dnn.core import Stimulus, Mask
from dnnbrain.dnn import models as db_models


model = eval('db_models.{}()'.format('alexnet'))


stim = Stimulus()
stim.load(r"C:\Users\xuwei\Desktop\test\test.dmask.csv")

dmask = Mask(r"C:\Users\xuwei\Desktop\test\test.stim.csv")

activation = model.compute_activation(stim, dmask)


activation.layers

activation.get('conv5')